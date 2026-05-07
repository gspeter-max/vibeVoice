"""
brain.py — Parakeet Flow v2 (Streaming Edition)
=================================================
Receives chunk and commit commands via Unix socket, decodes chunks as they arrive,
and pastes the stitched final text after session commit.
"""

import os
import sys
import socket
import time
import threading
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from rich.prompt import Prompt
from src.text_refiner.llm_router import refine_text_with_fallbacks, set_primary_provider, PROVIDERS
from src.utils.env_manager import check_and_ask_for_api_key
from src.streaming.streaming_shared_logic import (
    analyze_duplicate_chunk_prefix,
    remove_duplicate_chunk_prefix,
)
from src.backend.data_record.telemetry import StreamingSessionTelemetryRecorder
from src.utils.env_utils import get_float_from_environment
from src.utils.bootstrap import fix_macos_library_paths
from src import log
from rich.console import Console
from rich.table import Table
from rich import box

from src.backend.state import (
    backend_info, backend_lock,
    session_store, session_store_lock,
    RecordingState, SessionState
)

from src.backend.data_record.telemetry import (
    _telemetry_seed,
    _telemetry_recorder_for_session,
    _update_chunk_telemetry_summary,
    _update_session_telemetry_summary,
    _handle_session_telemetry_event,
    STREAMING_TELEMETRY_ENABLED,
    STREAMING_TELEMETRY_DIR,
    RECORDING_MODE
)

NO_STREAMING_MODE = "no_streaming"

try:
    from pynput.keyboard import Controller
except ImportError:  # pragma: no cover

    class Controller:  # type: ignore[override]
        def type(self, _text):
            return None


# Constants
SOCKET_PATH = "/tmp/parakeet.sock"
HUD_HOST, HUD_PORT = "127.0.0.1", 57234
keyboard = Controller()

# Environment Config
BACKEND = os.environ.get("BACKEND", "parakeet").lower().strip()

def _is_no_streaming_mode() -> bool:
    return RECORDING_MODE == NO_STREAMING_MODE


def _is_online(timeout=0.5) -> bool:
    """
    Checks for internet connectivity by attempting a quick connection to a reliable DNS server.
    Used to prevent the app from hanging on LLM requests when the user is offline.
    """
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=timeout)
        return True
    except OSError:
        return False


def load_transcription_engine(model_name="parakeet-tdt-0.6b-v3"):
    """
    Loads the requested AI model into memory wrapped in our clean TranscriptionEngine interface.
    """
    log.info(f"[Brain] Loading engine: {model_name}")
    
    if "nemotron" in model_name.lower():
        from src.engines.nemotron import NemotronEngine
        return NemotronEngine()
        
    from src.engines.parakeet import ParakeetEngine
    return ParakeetEngine(model_name)


def send_hud(cmd: str):
    """
    Communicates a state change to the always-on-top HUD (Heads-Up Display).
    The Brain sends simple text commands like 'listen', 'process', or 'done'
    to the HUD over a local network socket. The HUD then updates its visual
    state (such as changing the color or icon) to give the user immediate
    feedback on whether the AI is currently listening or transcribing audio.
    """
    try:
        with socket.create_connection((HUD_HOST, HUD_PORT), timeout=0.2) as s:
            s.sendall(cmd.encode())
    except Exception:
        pass


def _get_or_create_session(session_id: str) -> SessionState:
    """
    Returns the state object for a specific recording session, creating it if needed.
    This function manages the shared 'session_store' in a thread-safe manner,
    ensuring that audio chunks arriving from the Ear are mapped to the correct
    session. If it's a new session, it also initializes the telemetry recorder
    to ensure every new recording has a dedicated log file from the very first chunk.
    """
    with session_store_lock:
        if session := session_store.get(session_id):
            if STREAMING_TELEMETRY_ENABLED and session.telemetry_recorder is None:
                session.telemetry_recorder = StreamingSessionTelemetryRecorder(
                    session_id=session_id,
                    output_dir=STREAMING_TELEMETRY_DIR,
                    summary_seed=_telemetry_seed(),
                )
            return session

        with backend_lock:
            engine = backend_info.get("engine")

        telemetry_recorder = None
        if STREAMING_TELEMETRY_ENABLED:
            telemetry_recorder = StreamingSessionTelemetryRecorder(
                session_id=session_id,
                output_dir=STREAMING_TELEMETRY_DIR,
                summary_seed=_telemetry_seed(),
            )

        session = SessionState(
            engine=engine, telemetry_recorder=telemetry_recorder
        )
        session_store[session_id] = session
        return session


def _show_summary_table(session_id: str, raw_text: str, cleaned_text: str, stt_timing: float, refiner_timing: float):
    """
    Displays a formatted summary of the completed session in the terminal.
    """
    # Clear the previous meter line from Ear
    sys.stdout.write(f"\r\033[K")
    sys.stdout.flush()

    table = Table(title=f"📋 Session: {session_id[:8]}", box=box.ROUNDED, expand=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Status", "[bold green]DONE[/bold green]")
    table.add_row("Raw Text", f"[dim white]{raw_text}[/dim white]")
    table.add_row("Refined", f"[bold white]{cleaned_text}[/bold white]")
    llm_timing_str = f"{refiner_timing:.2f}s" if refiner_timing > 0 else "bypassed"
    table.add_row("Timing", f"STT: {stt_timing:.2f}s | LLM: {llm_timing_str} | Total: {(stt_timing + refiner_timing):.2f}s")
    table.add_row("Stats", f"Before: {len(raw_text)} chars | After: {len(cleaned_text)} chars")

    Console().print("\n", table, "\n")


def _finalize_recording_if_ready(session_id: str, rec_idx: int) -> None:
    """
    Stitches all chunks for ONE button press and pastes the final text if ready.
    A recording is ready to finalize when:
      - The Ear has committed it (rec.closed == True), AND
      - Every chunk that was received has been decoded (rec.done_count == rec.received_count)
    """
    with session_store_lock:
        session = session_store.get(session_id)
    if not session:
        return

    with session.lock:
        rec = session.recordings.get(rec_idx)
        if rec is None:
            return
        if rec.finalized or not rec.closed or rec.done_count != rec.received_count:
            return

        # Combine all non-empty transcript parts in sequential order
        parts = (part for _, part in sorted(rec.transcript_parts.items()) if part)
        text = " ".join(parts).strip()
        rec.finalized = True

    short_id = session_id[:8]
    if text:
        stt_time = rec.stt_time
        # log.info replaced by _show_summary_table for Zen mode
        
        send_hud("process")
        
        # Smart online/offline detection: skip LLM if no internet to ensure instant pasting
        t_refine_start = time.perf_counter()
        if _is_online():
            cleaned_text = refine_text_with_fallbacks(text)
            refine_time = time.perf_counter() - t_refine_start
        else:
            log.info("[Brain] 🔌 Offline mode detected — bypassing LLM refiner for instant paste")
            cleaned_text = text
            refine_time = 0.0
        
        _update_session_telemetry_summary(
            session_id,
            {
                "final_text": text,
                "cleaned_text": cleaned_text,
                "total_chunks_received": rec.received_count,
                "total_decode_seconds": round(stt_time, 2),
                "refine_seconds": round(refine_time, 2),
            },
        )
        _show_summary_table(session_id, text, cleaned_text, stt_time, refine_time)
        
        paste_instantly(cleaned_text + " ")
        _update_session_telemetry_summary(session_id, {"final_paste_success": True})
        send_hud("done")
    else:
        # log.info("[Brain] 🔇 Nothing detected") # Silenced for Zen
        _update_session_telemetry_summary(
            session_id,
            {
                "total_chunks_received": rec.received_count,
                "total_decode_seconds": round(rec.stt_time, 2),
            },
        )
        send_hud("hide")

    # Clear memory using our clean interface
    with session.lock:
        # log.info(f"[Brain] 🔄 Clearing engine memory...") # Silenced for Zen
        session.engine.clear_internal_memory()


# Keep the old name as an alias so any test code referencing it still works during migration
def _finalize_session_if_ready(session_id: str) -> None:
    """Deprecated: finalizes recording index 0 only. Use _finalize_recording_if_ready."""
    _finalize_recording_if_ready(session_id, 0)


def _handle_audio_chunk(
    session_id: str, rec_idx: int, seq: int, audio_bytes: bytes
) -> None:
    """
    Processes a single audio chunk: converts it to text and deduplicates against the previous chunk.
    rec_idx = which button press this chunk belongs to
    seq     = sequence number within that button press
    """
    session = _get_or_create_session(session_id)
    with session.lock:
        rec = session.get_or_create_recording(rec_idx)
        rec.received_count += 1
        
        if seq == 0 and rec.received_count == 1:
            log.info("[Brain] 🎙️  Recording started...")

    elapsed = 0.0
    audio_int16 = np.frombuffer(
        audio_bytes[: len(audio_bytes) // 2 * 2], dtype=np.int16
    )
    audio = _normalize_audio(audio_int16)

    text = ""
    dedup_analysis = None
    last_chunk_text = ""

    if audio is not None:
        try:
            engine = session.engine
            if not engine:
                log.info("[Brain] ⚠️  No engine loaded — skipping chunk")
            else:
                t_start = time.perf_counter()
                
                # 1. Transcribe blindly using the Rulebook
                text = engine.transcribe_chunk(audio)
                elapsed = time.perf_counter() - t_start
                
                with session.lock:
                    rec = session.get_or_create_recording(rec_idx)
                    
                    # 2. Check Rulebook to see how to handle the output
                    if engine.is_stateful():
                        # For cumulative engines, we store the full text in part 0
                        rec.transcript_parts = {0: text}
                        analysis_cleaned_text = text
                    else:
                        # Stateless backends transcribe chunks independently
                        last_chunk_text = rec.transcript_parts.get(seq - 1, "")
                        # Deduplicate the new chunk against the last chunk text
                        from src.streaming.streaming_shared_logic import analyze_duplicate_chunk_prefix
                        dedup_analysis = analyze_duplicate_chunk_prefix(last_chunk_text, text)

                        rec.transcript_parts[seq] = dedup_analysis.cleaned_text
                        
                    rec.stt_time += elapsed
                    session.stt_time += elapsed
                        
        except Exception as e:
            log.info(f"[Brain] ❌ Chunk decode error: {e}")
            import traceback
            traceback.print_exc()

    with session.lock:
        rec = session.get_or_create_recording(rec_idx)
        rec.done_count += 1

    audio_file_path = None
    recorder = _telemetry_recorder_for_session(session_id)
    if recorder:
        audio_file_path = recorder.save_chunk_audio(rec_idx, seq, audio_bytes)

    _update_chunk_telemetry_summary(
        session_id,
        rec_idx,
        seq,
        {
            "audio_file_path": audio_file_path,
            "decode_seconds": round(elapsed, 2),
            "last_chunk_text": last_chunk_text,
            "raw_text": text,
            "cleaned_text_after_dedup": dedup_analysis.cleaned_text if dedup_analysis else text,
            "dedup_stats": {
                "overlap_word_count": dedup_analysis.overlap_word_count if dedup_analysis else 0,
                "trim_applied": dedup_analysis.trim_applied if dedup_analysis else False,
                "combined_score": round(dedup_analysis.combined_score, 4) if dedup_analysis else 0.0,
                "char_score": round(dedup_analysis.char_score, 4) if dedup_analysis else 0.0,
                "token_score": round(dedup_analysis.token_score, 4) if dedup_analysis else 0.0,
                "skipped_too_small": dedup_analysis.skipped_because_result_too_small if dedup_analysis else False,
            },
        },
    )
    _update_session_telemetry_summary(
        session_id,
        {
            "total_chunks_received": sum(
                r.received_count for r in session.recordings.values()
            ),
            "total_decode_seconds": round(session.stt_time, 2),
            "flags": {"dedup_trim_applied": dedup_analysis.trim_applied if dedup_analysis else False},
        },
    )

    _finalize_recording_if_ready(session_id, rec_idx)


def _mark_session_closed(session_id: str, rec_idx: int) -> None:
    """
    Marks a specific recording as closed.
    """
    session = _get_or_create_session(session_id)
    with session.lock:
        rec = session.get_or_create_recording(rec_idx)
        rec.closed = True
        
    _finalize_recording_if_ready(session_id, rec_idx)


def _handle_session_event(blob: bytes) -> None:
    """
    Extracts and processes telemetry events from raw socket data.
    Header format: CMD_SESSION_EVENT:SESSION_ID:RECORDING_INDEX
    """
    try:
        header, payload_blob = blob.split(b"\n\n", 1)
        # Split into 3 parts: CMD, session_id, recording_index
        _, session_id, rec_idx_str = header.decode("utf-8").strip().split(":", 2)

        payload = json.loads(payload_blob.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("session event payload must be a JSON object")

        # Inject the recording index from the header into the payload
        # so _handle_session_telemetry_event routes it to the correct recording slot.
        payload["recording_index"] = int(rec_idx_str)

        log.info(
            "📡 Session event received",
            session=session_id[:8],
            recording=rec_idx_str,
            kind=payload.get("type", "session_event"),
        )
        _handle_session_telemetry_event(session_id, payload)
    except Exception as e:
        log.warning("⚠️  Bad session event command", error=str(e))


def _transcribe_raw_connection_audio(blob: bytes, t_connect: float) -> None:
    """
    Handles legacy 'one-shot' audio transcription for non-streaming modes.
    In this mode, the Brain receives the entire audio recording as a single
    block of data. It normalizes the entire block, runs a single transcription
    pass, and pastes the result. This path is used as a fallback or for
    situations where real-time streaming feedback is not required by the user.
    """
    with backend_lock:
        engine = backend_info.get("engine")

    if not engine:
        log.info("[Brain] ⚠️  No engine loaded — skipping")
        send_hud("hide")
        return

    try:
        audio_int16 = np.frombuffer(blob[: len(blob) // 2 * 2], dtype=np.int16)
        audio = _normalize_audio(audio_int16)
        if audio is None:
            log.info("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return

        log.info(f"[Brain] 🎙️  Final utterance decode")
        send_hud("process")
        final_text = engine.transcribe_chunk(audio)
        if not final_text:
            log.info("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return
    except Exception as e:
        log.info(f"[Brain] ❌ Audio decode error: {e}")
        send_hud("hide")
        return

    stt_time = time.perf_counter() - t_connect
    
    # Smart online/offline detection: skip LLM if no internet to ensure instant pasting
    t_refine_start = time.perf_counter()
    if _is_online():
        cleaned_text = refine_text_with_fallbacks(final_text)
        refine_time = time.perf_counter() - t_refine_start
        llm_log_str = f"{refine_time:.2f}s"
    else:
        log.info("[Brain] 🔌 Offline mode detected — bypassing LLM refiner for instant paste")
        cleaned_text = final_text
        refine_time = 0.0
        llm_log_str = "bypassed"

    log.info(
        f'[Brain] 📝 [STT: {stt_time:.2f}s | LLM: {llm_log_str}] → "{cleaned_text}"'
    )
    
    # We do not use telemetry for the non-streaming path currently, but we can show the table.
    _show_summary_table("one-shot", final_text, cleaned_text, stt_time, refine_time)
    
    paste_instantly(cleaned_text + " ")
    send_hud("done")


def paste_instantly(text: str):
    """
    Simulates a 'Cmd+V' paste operation to insert text into the active app.
    It first saves the current clipboard content, copies the new transcribed
    text to the clipboard, triggers the system's paste command via AppleScript,
    and then restores the original clipboard. If this fast method fails, it
    falls back to simulating individual keystrokes, which is slower but more reliable.
    """
    try:
        # Save old clipboard
        old = subprocess.check_output(["pbpaste"], stderr=subprocess.DEVNULL).decode(
            "utf-8", errors="ignore"
        )

        # Copy new text
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(input=text.encode("utf-8"))

        # Paste via AppleScript
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True,
        )

        # Restore clipboard
        time.sleep(0.05)
        if old:
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(input=old.encode("utf-8"))

    except Exception as e:
        log.info(f"[Brain] ⚠️  Paste failed: {e}. Falling back to slow typing.")
        keyboard.type(text)


def _normalize_audio(int16_audio: np.ndarray) -> np.ndarray | None:
    """
    Converts raw 16-bit integer audio into normalized 32-bit floating point.
    This function scales the audio samples to a range between -1.0 and 1.0,
    which is the format expected by the AI transcription models. It also
    performs a basic silence check to reject empty chunks and applies a
    peak normalization to ensure that quiet recordings are boosted to a
    consistent volume level before they are processed by the backend.
    """
    if len(int16_audio) == 0:
        return None
    audio = int16_audio.astype(np.float32) / 32768.0
    max_val = float(np.max(np.abs(audio)))
    if max_val < 0.001:
        return None
    return audio / max_val * 0.9 if max_val > 0.01 else audio


def handle_connection(conn):
    """
    Manages an incoming socket connection from the Ear or another client.
    This function reads all data from the socket into a buffer and then
    dispatches it to the appropriate handler based on the command prefix.
    It can handle model switches, session commits, telemetry events, or
    raw audio data. By running in a separate thread for each connection,
    the Brain can handle multiple incoming data streams simultaneously.
    """
    t_connect = time.perf_counter()
    raw_audio = bytearray()
    try:
        conn.settimeout(0.1)
        while True:
            try:
                data = conn.recv(32768)
                if not data:
                    break
                raw_audio.extend(data)
            except socket.timeout:
                continue
    except Exception as e:
        log.info(f"[Brain] ⚠️  Recv error: {e}")
    finally:
        conn.close()

    if not raw_audio:
        return
    blob = bytes(raw_audio)
    # log.info("📥 Utterance received", size=len(blob)) # Silenced for Zen mode

    if blob.startswith(b"CMD_SWITCH_MODEL:"):
        _handle_switch_model(blob)
    elif blob.startswith(b"CMD_SESSION_COMMIT:"):
        _handle_session_commit(blob)
    elif blob.startswith(b"CMD_SESSION_EVENT:") and b"\n\n" in blob:
        _handle_session_event(blob)
    elif blob.startswith(b"CMD_AUDIO_CHUNK:") and b"\n\n" in blob:
        _handle_chunk_command(blob)
    else:
        _transcribe_raw_connection_audio(blob, t_connect)


def _handle_switch_model(blob: bytes):
    """
    Processes a command to switch the active AI transcription model.
    It extracts the new model name from the command string, unloads the
    current model to free up memory (using garbage collection), and then
    loads the new model into the backend. This allows users to quickly
    swap between faster but less accurate models and slower but more
    powerful ones without needing to restart the entire application.
    """
    try:
        new_model = blob.decode("utf-8").strip().split(":", 1)[1]
        log.info("🔄 Switching model", model=new_model)
        with backend_lock:
            import gc

            backend_info["engine"] = None
            gc.collect()
            backend_info["engine"] = load_transcription_engine(new_model)
            
            with session_store_lock:
                session_store.clear()
                
            log.info("✅ Model switched", model=new_model)
    except Exception as e:
        log.error("❌ Switching failed", error=str(e))


def _handle_session_commit(blob: bytes):
    """
    Handles the commit command: signals that one button press is complete.
    Header format: CMD_SESSION_COMMIT:SESSION_ID:RECORDING_INDEX
    """
    try:
        # Split into 3 parts: CMD, session_id, rec_idx
        _, session_id, rec_idx_str = blob.decode("utf-8").strip().split(":", 2)
        rec_idx = int(rec_idx_str)
        log.info(
            "✅ Session commit received", session=session_id[:8], recording=rec_idx
        )
        _mark_session_closed(session_id, rec_idx)
    except Exception:
        log.warning("⚠️  Bad commit command")


def _handle_chunk_command(blob: bytes):
    """
    Parses and routes an individual audio chunk command from the Ear.
    Header format: CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ
    """
    header, audio_bytes = blob.split(b"\n\n", 1)
    try:
        # Split into 4 parts: CMD, session_id, rec_idx, seq
        _, session_id, rec_idx_str, seq_text = (
            header.decode("utf-8").strip().split(":", 3)
        )
        log.info(
            "🎙️  Audio chunk received",
            rec=rec_idx_str,
            seq=seq_text,
            size=len(audio_bytes),
            session=session_id[:8],
        )
        _handle_audio_chunk(session_id, int(rec_idx_str), int(seq_text), audio_bytes)
    except Exception as e:
        log.warning("⚠️  Bad chunk header", error=str(e))


def start_server():
    """
    Initializes the Brain server and begins listening for incoming connections.
    """
    # 1. Auto-fix environment issues (e.g., macOS library paths)
    fix_macos_library_paths()

    # 2. Setup Provider from environment (configured by wizard.py)
    # We no longer prompt here to avoid EOFError in background.
    provider_idx_str = os.environ.get("VIBEVOICE_PROVIDER_INDEX", "0")
    try:
        provider_idx = int(provider_idx_str)
    except ValueError:
        provider_idx = 0

    set_primary_provider(provider_idx)
    log.info(f"[Brain] Text refiner set to: {PROVIDERS[provider_idx]['name']}")

    # 3. Load the STT engine
    # Use STT_MODEL from .env (configured by wizard_tui.py)
    stt_model = os.environ.get("STT_MODEL", "parakeet-tdt-0.6b-v3")
    backend_info["engine"] = load_transcription_engine(stt_model)
    log.info(f"[Brain] Warming up model: {stt_model}...")
    try:
        backend_info["engine"].transcribe_chunk(np.zeros(8000, dtype=np.float32))
    except Exception:
        pass
    log.info("[Brain] Warm-up done ✓")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(10)
    log.info(f"[Brain] ✅ Streaming server ready at {SOCKET_PATH}")

    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(
                target=handle_connection, args=(conn,), daemon=True
            ).start()
    except KeyboardInterrupt:
        log.info("\n[Brain] Shutting down...")
        try:
            with session_store_lock:
                for session_id, session_state in session_store.items():
                    _update_session_telemetry_summary(
                        session_id,
                        {
                            "total_chunks_received": sum(
                                r.received_count
                                for r in session_state.recordings.values()
                            ),
                            "total_decode_seconds": round(session_state.stt_time, 2),
                        },
                    )
        except Exception as e:
            log.warning(f"⚠️ Failed to write shutdown telemetry: {e}")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
            
        # Close the LLM router connection pool safely
        try:
            from src.text_refiner.llm_router import global_http_client
            global_http_client.close()
            log.info("[Brain] 🌐 LLM Router connection pool closed")
        except Exception as e:
            log.warning(f"⚠️ Failed to close LLM router client: {e}")


if __name__ == "__main__":
    start_server()
