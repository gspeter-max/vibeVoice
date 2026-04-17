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
from streaming_shared_logic import (
    analyze_duplicate_chunk_prefix,
    remove_duplicate_chunk_prefix,
)
from streaming_session_telemetry import StreamingSessionTelemetryRecorder
from src import log
from rich.console import Console
from rich.table import Table
from rich import box

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
BACKEND = os.environ.get("BACKEND", "faster_whisper").lower().strip()
RECORDING_MODE = os.environ.get("RECORDING_MODE", "silence_streaming").strip().lower()
STREAMING_TELEMETRY_ENABLED = (
    os.environ.get("STREAMING_TELEMETRY_ENABLED", "0").strip() == "1"
)
STREAMING_TELEMETRY_DIR = Path(
    os.environ.get("STREAMING_TELEMETRY_DIR", "logs/streaming_sessions")
)

# Global state
backend_info = {"backend": None, "model": None}
backend_lock = threading.Lock()
session_store = {}
session_store_lock = threading.Lock()


@dataclass
class RecordingState:
    """
    Holds the transcription state for a single button press.
    One SessionState contains many RecordingState objects — one per button press.
    """

    received_count: int = 0
    done_count: int = 0
    closed: bool = False
    finalized: bool = False
    transcript_parts: dict = field(default_factory=dict)
    stt_time: float = 0.0


@dataclass
class SessionState:
    """
    State object for an entire application run (one Brain process lifetime).
    A session survives multiple button presses; each press is one RecordingState
    stored in the recordings dict keyed by its recording_index integer.
    """

    backend: object
    model: object
    # recordings[0] = first button press, recordings[1] = second, …
    recordings: dict = field(default_factory=dict)
    stt_time: float = 0.0
    telemetry_recorder: StreamingSessionTelemetryRecorder | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create_recording(self, rec_idx: int) -> "RecordingState":
        """Returns the RecordingState for rec_idx, creating it if needed."""
        if rec_idx not in self.recordings:
            self.recordings[rec_idx] = RecordingState()
        return self.recordings[rec_idx]


def _model_name_for_telemetry() -> str | None:
    """
    Returns the active model name for labeling telemetry sessions.
    This function checks the currently loaded backend and model to extract
    a human-readable name, such as 'base.en' or 'parakeet-tdt'. This ensures
    that every saved telemetry file can clearly show which AI model was used
    to generate the transcriptions during that specific session.
    """
    model_name = getattr(backend_info.get("backend"), "CURRENT_MODEL_NAME", None)
    if model_name:
        return model_name
    return getattr(backend_info.get("model"), "model_name", None)


def _telemetry_seed() -> dict:
    """
    Constructs the initial configuration data for a new telemetry session.
    This dictionary includes the recording mode, current backend type, and model name,
    as well as all the VAD (Voice Activity Detection) parameters used in the session.
    By seeding this data at the start, we create a comprehensive 'header' in the
    telemetry file that documents the exact settings and hardware state for debugging.
    """
    return {
        "recording_mode": RECORDING_MODE,
        "backend": BACKEND,
        "model": _model_name_for_telemetry(),
        "telemetry_enabled": STREAMING_TELEMETRY_ENABLED,
        "flags": {
            "vad_no_speech_warning_seen": False,
            "dedup_trim_applied": False,
        },
        "config": {
            "vad_threshold": float(os.environ.get("VAD_THRESHOLD", "0.5")),
            "silence_timeout_seconds": float(
                os.environ.get(
                    "VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT",
                    "1.0",
                )
            ),
            "energy_threshold": float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.05")),
            "energy_ratio": float(os.environ.get("VAD_ENERGY_RATIO", "2.5")),
            "overlap_seconds": float(os.environ.get("OVERLAP_SECONDS", "1")),
            "minimum_chunk_seconds": float(os.environ.get("MIN_CHUNK_SECONDS", "8.0")),
        },
    }


def _telemetry_recorder_for_session(
    session_id: str,
) -> StreamingSessionTelemetryRecorder | None:
    """
    Retrieves or initializes a telemetry recorder for a given session ID.
    If telemetry is globally disabled, it returns None immediately. Otherwise,
    it looks for an existing recorder within the session store or creates a new one,
    ensuring that a session state object exists to hold the recorder. This acts
    as a centralized factory for managing logging objects during streaming.
    """
    if not STREAMING_TELEMETRY_ENABLED:
        return None

    with session_store_lock:
        session = session_store.get(session_id)
        if session and session.telemetry_recorder is not None:
            return session.telemetry_recorder

        recorder = StreamingSessionTelemetryRecorder(
            session_id=session_id,
            output_dir=STREAMING_TELEMETRY_DIR,
            summary_seed=_telemetry_seed(),
        )
        if session is None:
            session = SessionState(
                backend=backend_info["backend"],
                model=backend_info["model"],
                telemetry_recorder=recorder,
            )
            session_store[session_id] = session
        else:
            session.telemetry_recorder = recorder
        return recorder


def _update_chunk_telemetry_summary(
    session_id: str, recording_index: int, chunk_index: int, fields: dict
) -> None:
    """Updates the summary data for a specific chunk inside a given recording."""
    recorder = _telemetry_recorder_for_session(session_id)
    if recorder is None:
        return
    recorder.update_chunk_summary(recording_index, chunk_index, fields)


def _update_session_telemetry_summary(session_id: str, fields: dict) -> None:
    """
    Merges top-level session information into the overall telemetry report.
    This is used to record final session outcomes such as the total processing time,
    the final combined transcript, and whether the final paste was successful.
    It provides a high-level overview of the entire session's performance and
    success rate without needing to dig into every individual audio chunk.
    """
    recorder = _telemetry_recorder_for_session(session_id)
    if recorder is None:
        return
    recorder.update_session_summary(fields)


def _handle_session_telemetry_event(session_id: str, payload: dict) -> None:
    """
    Routes incoming telemetry commands from the Ear to the correct recorder logic.
    Updates the session-wide 'summary' flags, such as flagging that a VAD warning
    was seen, so the final report highlights potential microphone sensitivity issues.
    """
    event_type = str(payload.get("type", "session_event"))
    chunk_index = payload.get("chunk_index")
    recording_index = payload.get("recording_index", 0)

    if event_type == "vad_no_speech_warning":
        _update_session_telemetry_summary(
            session_id,
            {
                "flags": {
                    "vad_no_speech_warning_seen": True,
                    "dedup_trim_applied": False,
                }
            },
        )
        return

    fields = {
        key: value
        for key, value in payload.items()
        if key not in {"type", "chunk_index", "recording_index"}
    }

    if event_type == "chunk_sent_to_brain":
        _update_chunk_telemetry_summary(
            session_id, recording_index, int(chunk_index), fields
        )
        return

    if event_type == "silence_threshold_hit":
        _update_chunk_telemetry_summary(
            session_id, recording_index, int(chunk_index), fields
        )
        return


def _is_no_streaming_mode() -> bool:
    return RECORDING_MODE == NO_STREAMING_MODE


def load_backend(model_name="base.en"):
    """
    Loads the requested ASR (Automatic Speech Recognition) model into memory.
    This function acts as a factory that chooses between the faster-whisper
    engine or the NVIDIA Parakeet-TDT engine depending on the model name.
    It returns a tuple containing the backend module and the loaded model
    object, allowing the Brain to switch models dynamically during runtime
    whenever the user requests a change from the terminal.
    """
    if "parakeet-tdt" in model_name:
        log.info(f"[Brain] Backend: NVIDIA Parakeet-TDT (via sherpa-onnx)")
        import backend_parakeet as backend

        return backend, backend.load_model(model_name)

    log.info(f"[Brain] Backend: faster-whisper + {model_name} + INT8 (CPU)")
    import backend_faster_whisper as backend

    return backend, backend.load_model(model_name)


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
            backend = backend_info["backend"]
            model = backend_info["model"]

        telemetry_recorder = None
        if STREAMING_TELEMETRY_ENABLED:
            telemetry_recorder = StreamingSessionTelemetryRecorder(
                session_id=session_id,
                output_dir=STREAMING_TELEMETRY_DIR,
                summary_seed=_telemetry_seed(),
            )

        session = SessionState(
            backend=backend, model=model, telemetry_recorder=telemetry_recorder
        )
        session_store[session_id] = session
        return session


def _show_summary_table(session_id: str, text: str, stt_timing: float):
    """
    Displays a formatted summary of the completed session in the terminal.
    Using the 'rich' library, this function prints a colorful table showing
    the final transcribed text, the time it took to transcribe (STT), and
    the character count. This provides developers with an easy-to-read
    confirmation of the session results immediately after the text is pasted.
    """
    table = Table(title=f"📋 Session: {session_id[:8]}", box=box.ROUNDED, expand=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Status", "[bold green]DONE[/bold green]")
    table.add_row("Text", f"[bold white]{text}[/bold white]")
    table.add_row("Timing", f"STT: {stt_timing:.2f}s")
    table.add_row("Stats", f"{len(text)} chars")

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
        log.info(
            "🏁 Finalizing recording",
            session=short_id,
            recording=rec_idx,
            chunks=rec.received_count,
            size=len(text),
            stt_time=f"{stt_time:.2f}s",
            text=text,
        )
        _update_session_telemetry_summary(
            session_id,
            {
                "final_text": text,
                "total_chunks_received": rec.received_count,
                "total_decode_seconds": round(stt_time, 2),
            },
        )
        _show_summary_table(session_id, text, stt_time)
        send_hud("process")
        paste_instantly(text + " ")
        _update_session_telemetry_summary(session_id, {"final_paste_success": True})
        send_hud("done")
    else:
        log.info("🔇 Nothing detected", session=short_id, recording=rec_idx)
        _update_session_telemetry_summary(
            session_id,
            {
                "total_chunks_received": rec.received_count,
                "total_decode_seconds": round(rec.stt_time, 2),
            },
        )
        send_hud("hide")


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

    elapsed = 0.0
    audio_int16 = np.frombuffer(
        audio_bytes[: len(audio_bytes) // 2 * 2], dtype=np.int16
    )
    audio = _normalize_audio(audio_int16)

    text = ""
    if audio is not None:
        try:
            backend, model = session.backend, session.model
            if not backend or not model:
                log.info("[Brain] ⚠️  No model loaded — skipping chunk")
            else:
                t_start = time.perf_counter()
                text = backend.transcribe(model, audio).strip()
                elapsed = time.perf_counter() - t_start
                log.info(
                    f"[Brain] 🎙️  [{session_id[:8]} rec={rec_idx} chunk={seq}] decode took {elapsed:.2f}s"
                )
                with session.lock:
                    rec = session.get_or_create_recording(rec_idx)
                    rec.stt_time += elapsed
                    session.stt_time += elapsed
        except Exception as e:
            log.info(f"[Brain] ❌ Chunk decode error: {e}")

    with session.lock:
        rec = session.get_or_create_recording(rec_idx)
        prev_text = rec.transcript_parts.get(seq - 1, "")
        analysis = analyze_duplicate_chunk_prefix(prev_text, text)
        rec.transcript_parts[seq] = analysis.cleaned_text
        rec.done_count += 1

    _update_chunk_telemetry_summary(
        session_id,
        rec_idx,
        seq,
        {
            "decode_seconds": round(elapsed, 2),
            "previous_chunk_text": prev_text,
            "raw_text": text,
            "cleaned_text_after_dedup": analysis.cleaned_text,
            "dedup_stats": {
                "overlap_word_count": analysis.overlap_word_count,
                "trim_applied": analysis.trim_applied,
                "combined_score": round(analysis.combined_score, 4),
                "char_score": round(analysis.char_score, 4),
                "token_score": round(analysis.token_score, 4),
                "skipped_too_small": analysis.skipped_because_result_too_small,
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
            "flags": {"dedup_trim_applied": analysis.trim_applied},
        },
    )

    _finalize_recording_if_ready(session_id, rec_idx)


def _mark_session_closed(session_id: str, rec_idx: int) -> None:
    """
    Marks a specific recording (button press) as closed — no more chunks are coming for it.
    This fires when the user releases the record hotkey. The Brain then waits for all
    in-flight chunks for that recording to finish decoding before pasting the result.
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
        backend, model = backend_info["backend"], backend_info["model"]

    if not backend or not model:
        log.info("[Brain] ⚠️  No model loaded — skipping")
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
        final_text = backend.transcribe(model, audio).strip()
        if not final_text:
            log.info("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return
    except Exception as e:
        log.info(f"[Brain] ❌ Audio decode error: {e}")
        send_hud("hide")
        return

    log.info(
        f'[Brain] 📝 [{time.perf_counter() - t_connect:.2f}s total] → "{final_text}"'
    )
    paste_instantly(final_text + " ")
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
    log.info("📥 Utterance received", size=len(blob))

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

            backend_info["model"] = None
            gc.collect()
            backend_info["backend"], backend_info["model"] = load_backend(new_model)
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
    It first loads the default AI model and performs a 'warm-up' transcription
    to ensure everything is ready for the first real user input. Then, it
    sets up a Unix domain socket and enters a loop to accept new connections,
    spawning a new thread for each one. This server is the central 'nervous system'
    that coordinates all audio processing and text output for the app.
    """
    backend_info["backend"], backend_info["model"] = load_backend("base.en")
    log.info("[Brain] Warming up model...")
    try:
        backend_info["backend"].transcribe(
            backend_info["model"], np.zeros(8000, dtype=np.float32)
        )
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


if __name__ == "__main__":
    start_server()
