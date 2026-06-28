"""Brain — Streaming Transcription Server
=========================================

The Brain is the central processing unit of the VibeVoice pipeline.
It runs as a persistent Unix-socket server that the Ear connects to in
order to deliver audio data.  For every incoming connection the Brain
spawns a dedicated thread so that multiple audio chunks can be decoded
in parallel without blocking the accept loop.

Responsibilities
----------------
* Load and hot-swap the AI transcription model (Parakeet / Nemotron).
* Receive raw audio bytes and route them to the correct handler based
  on the command prefix embedded in the message.
* Decode each audio chunk through the transcription engine, deduplicate
  overlapping text between consecutive chunks, and stitch the parts
  together when a recording session is committed.
* Refine the stitched transcript through the LLM text-refiner, then
  insert_transcripte the final result into the active application.
* Push simple state strings (``listen``, ``process``, ``done``, ``hide``)
  to the HUD overlay over a local TCP socket so the user always has
  visual feedback on what the pipeline is doing.

Wire Protocol (Ear → Brain, Unix socket)
-----------------------------------------
Every message the Ear sends is a byte string that starts with one of
these prefixes:

``CMD_SWITCH_MODEL:<model_name>``
    Unload the current engine and load ``<model_name>`` instead.

``CMD_SESSION_COMMIT:<session_id>:<rec_idx>``
    Signal that a button press is finished.  The Brain will finalize
    the recording and insert_transcripte the result once all chunks are decoded.

``CMD_SESSION_EVENT:<session_id>:<rec_idx>\\n\\n<json>``
    Deliver a structured telemetry event for the given recording slot.

``CMD_AUDIO_CHUNK:<session_id>:<rec_idx>:<seq>\\n\\n<raw_pcm>``
    Deliver a single audio chunk.  ``rec_idx`` identifies the button
    press; ``seq`` is the zero-based sequence number of this chunk
    within that press.

Any other bytes are treated as a legacy one-shot recording and the
entire buffer is transcribed as a single pass.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from sound import PlaySound
from src import log
from src.backend.data_record.telemetry import (
    _handle_session_telemetry_event,
    _update_chunk_telemetry_summary,
    build_recorder,
    update_summary,
)
from src.backend.state import (
    BackendState,
    SessionState,
    SessionStates,
)
from src.ipc.client import SocketConfig, create_socket, send_message
from src.text_refiner.llm_router import (
    PROVIDERS,
    refine,
    set_provider,
)
from src.utils.bootstrap import fix_macos_library_paths
from src.utils.settings import settings

# ---------------------------------------------------------------------------
# HUD communication
# ---------------------------------------------------------------------------


def send_hud(cmd: str) -> None:
    """Push a state-change command to the HUD overlay over a local TCP socket.

    The HUD listens on ``settings.hud_host:settings.hud_port``.  Valid
    commands are short strings such as ``"listen"``, ``"process"``,
    ``"done"``, and ``"hide"``.  The connection is fire-and-forget with a
    200 ms timeout; if the HUD is not running the error is silently ignored
    so the Brain never blocks on UI feedback.

    Args:
        cmd: The state command to send, e.g. ``"listen"`` or ``"done"``.
    """
    try:
        cfg = SocketConfig(
            family=socket.AF_UNIX,
            socket_type=socket.SOCK_STREAM,
            address=settings.brain_to_hud_socket_path,
            timeout=0.2,
        )
        s = create_socket(cfg)
        send_message(message_bytes=cmd.encode(), sock=s)
    except (OSError, ConnectionRefusedError):
        pass  # HUD is optional — Brain never blocks on UI feedback


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------


def _get_or_create_session(session_id: str, state: SessionStates) -> SessionStates:
    """Return the :class:`~src.backend.state.SessionState` for a session.

    If the session does not exist yet it is created, registered in the
    shared ``session_store``, and a telemetry recorder is attached when
    telemetry is enabled.  Thread-safe: the function acquires
    ``session_store_lock`` for all reads and writes to the store.

    Args:
        session_id: Opaque string that uniquely identifies the user's
            recording session.  Generated once per app launch by the Ear.

    Returns:
        The existing or newly created :class:`~src.backend.state.SessionState`
        for ``session_id``.
    """
    with state.lock:
        if session := state.sessions.get(session_id):
            if settings.streaming_telemetry_enabled and session.telemetry_recorder is None:
                session.telemetry_recorder = build_recorder(session_id=session_id, state=state)
                state.sessions[session_id] = session
            return state

        telemetry_recorder = None
        if settings.streaming_telemetry_enabled:
            telemetry_recorder = build_recorder(session_id=session_id, state=state)

        session = SessionState(telemetry_recorder=telemetry_recorder)
        state.sessions[session_id] = session
        return state


# ---------------------------------------------------------------------------
# Terminal summary table
# ---------------------------------------------------------------------------


def _show_summary_table(
    session_id: str,
    raw_text: str,
    cleaned_text: str,
    stt_timing: float,
    refiner_timing: float,
) -> None:
    """Print a Rich-formatted summary table for a completed recording session.

    The table shows the raw STT output, the LLM-refined output, timing for
    each stage, and before/after character counts.  It clears the live voice
    level meter line printed by the Ear before rendering.

    Args:
        session_id: Session identifier (first 8 characters shown as title).
        raw_text: The stitched transcript before LLM refinement.
        cleaned_text: The transcript after LLM refinement.
        stt_timing: Total seconds spent on speech-to-text decoding.
        refiner_timing: Total seconds spent on LLM refinement (0 = bypassed).
    """
    # Clear the previous meter line from Ear
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    table = Table(title=f"📋 Session: {session_id[:8]}", box=box.ROUNDED, expand=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Status", "[bold green]DONE[/bold green]")
    table.add_row("Raw Text", f"[dim white]{raw_text}[/dim white]")
    table.add_row("Refined", f"[bold white]{cleaned_text}[/bold white]")
    llm_timing_str = f"{refiner_timing:.2f}s" if refiner_timing > 0 else "bypassed"
    table.add_row(
        "Timing",
        (
            f"STT: {stt_timing:.2f}s | LLM: {llm_timing_str} | "
            f"Total: {(stt_timing + refiner_timing):.2f}s"
        ),
    )
    table.add_row("Stats", f"Before: {len(raw_text)} chars | After: {len(cleaned_text)} chars")

    Console().print("\n", table, "\n")


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_recording(session_id: str, rec_idx: int, state: SessionStates) -> None:
    """Stitch, refine, and insert_transcripte the final text for one button press.

    A recording slot is ready when **both** conditions are true:

    * ``rec.closed == True`` — the Ear sent ``CMD_SESSION_COMMIT`` for this
      recording index, meaning no more chunks are coming.
    * ``rec.done_count == rec.received_count`` — every chunk that arrived has
      finished decoding (which may happen on multiple threads).

    When ready the function:

    1. Joins all non-empty :attr:`~src.backend.state.RecordingState.transcript_parts`
       in sequence order.
    2. Sends ``"process"`` to the HUD while the LLM refiner runs.
    3. Calls :func:`insert_transcripte` with the refined text.
    4. Logs the session summary table in the terminal.
    5. Sends ``"done"`` (or ``"hide"`` when the transcript is empty) to the HUD.

    This function is a no-op when the recording is already finalized, not yet
    closed, or still has outstanding decode threads.

    Args:
        session_id: Session identifier passed through from the chunk command.
        rec_idx: Zero-based index of the button press to finalize.
    """
    with state.lock:
        session = state.sessions.get(session_id)

    if session is None:
        return

    with state.lock:
        rec = session.recordings.get(rec_idx)
        if rec is None:
            return
        if rec.finalized or not rec.closed or rec.done_count != rec.received_count:
            return

        # Combine all non-empty transcript parts in sequential order
        parts = (part for _, part in sorted(rec.transcript_parts.items()) if part)
        text = " ".join(parts).strip()
        rec.finalized = True

    if text:
        stt_time = rec.stt_time

        send_hud("process")

        t_refine_start = time.perf_counter()
        cleaned_text = refine(text)
        refine_time = time.perf_counter() - t_refine_start

        update_summary(
            session_id,
            {
                "final_text": text,
                "cleaned_text": cleaned_text,
                "total_chunks_received": rec.received_count,
                "total_decode_seconds": round(stt_time, 2),
                "refine_seconds": round(refine_time, 2),
            },
            state,
        )
        insert_transcripte(cleaned_text + " ")
        _show_summary_table(session_id, text, cleaned_text, stt_time, refine_time)
        update_summary(session_id, {"final_insert_transcripte_success": True}, state)
        send_hud("done")

    else:
        update_summary(
            session_id,
            {
                "total_chunks_received": rec.received_count,
                "total_decode_seconds": round(rec.stt_time, 2),
            },
            state,
        )
        send_hud("hide")

    engine = state.backend.get_engine()
    if engine is not None:
        engine.clear_internal_memory()


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


def handle_streaming_audio_chunk(
    session_id: str,
    rec_idx: int,
    seq: int,
    audio_bytes: bytes,
    state: SessionStates,
) -> None:
    """Decode one audio chunk and store the transcript fragment.

    This is the hot path that runs on a dedicated thread for every chunk the
    Ear delivers.  The steps are:

    1. Increment the received-chunk counter inside the session lock.
    2. Convert the raw PCM bytes to a normalised ``float32`` array.
    3. Run the transcription engine on the float array.
    4. For **stateful** engines (e.g. Nemotron), replace the single
       accumulated transcript part with the latest full output.
       For **stateless** engines (e.g. Parakeet), deduplicate the new text
       against the previous chunk's text and store the trimmed result.
    5. Increment the done-chunk counter, then call
       :func:`finalize_recording` which is a no-op unless all
       chunks are decoded and the session is committed.

    Args:
        session_id: Opaque session identifier from the Ear.
        rec_idx: Which button press this chunk belongs to (0-based).
        seq: Sequence number of this chunk within the current button press
            (0-based).  Used to order transcript parts during stitching.
        audio_bytes: Raw 16-bit little-endian PCM audio at
            ``settings.rate`` Hz, mono channel.
    """
    _get_or_create_session(session_id, state)
    with state.lock:
        session = state.sessions.get(session_id)
        rec = session.get_or_create_recording(rec_idx)
        rec.received_count += 1

        if seq == 0 and rec.received_count == 1:
            log.info("[Brain] 🎙️  Recording started...")

    elapsed = 0.0
    audio_int16 = np.frombuffer(audio_bytes[: len(audio_bytes) // 2 * 2], dtype=np.int16)
    audio = _normalize_audio(audio_int16)

    text = ""
    dedup_analysis = None
    last_chunk_text = ""
    engine = state.backend.get_engine()

    if audio is not None:
        try:
            if not engine:
                log.info("[Brain] ⚠️  No engine loaded — skipping chunk")
            else:
                t_start = time.perf_counter()

                text = engine.transcribe_chunk(audio)
                time_taken = time.perf_counter() - t_start

                with state.lock:
                    rec = session.get_or_create_recording(rec_idx)

                    if engine.is_stateful():
                        rec.transcript_parts = {0: text}
                    else:
                        last_chunk_text = rec.transcript_parts.get(seq - 1, "")
                        from src.streaming.session import dedup_prefix

                        dedup_analysis = dedup_prefix(last_chunk_text, text)
                        rec.transcript_parts[seq] = dedup_analysis.text

                    rec.stt_time += time_taken
                    session.stt_time += time_taken

        except (OSError, RuntimeError, ValueError) as e:
            log.info("[Brain] Chunk decode error: %s", e, exc_info=True)

    with state.lock:
        rec = session.get_or_create_recording(rec_idx)
        rec.done_count += 1

    audio_file_path = None
    # recorder = build_recorder(session_id)
    recorder = session.telemetry_recorder
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
            "cleaned_text_after_dedup": dedup_analysis.text if dedup_analysis else text,
            "dedup_stats": {
                "overlap_word_count": (dedup_analysis.overlap_words if dedup_analysis else 0),
                "trim_applied": dedup_analysis.trimmed if dedup_analysis else False,
                "combined_score": (
                    round(dedup_analysis.combined_score, 4) if dedup_analysis else 0.0
                ),
                "char_score": round(dedup_analysis.char_score, 4) if dedup_analysis else 0.0,
                "token_score": round(dedup_analysis.token_score, 4) if dedup_analysis else 0.0,
                "skipped_too_small": (dedup_analysis.skipped if dedup_analysis else False),
            },
        },
        state,
    )
    update_summary(
        session_id,
        {
            "total_chunks_received": sum(r.received_count for r in session.recordings.values()),
            "total_decode_seconds": round(session.stt_time, 2),
            "flags": {"dedup_trim_applied": (dedup_analysis.trimmed if dedup_analysis else False)},
        },
        state,
    )

    finalize_recording(session_id, rec_idx, state)


def _mark_session_closed(session_id: str, rec_idx: int, state: SessionStates) -> None:
    """Mark a recording slot as closed and trigger finalization.

    Called when the Brain receives ``CMD_SESSION_COMMIT``.  Setting
    ``rec.closed = True`` tells :func:`finalize_recording` that
    no more chunks will arrive for this recording index, so it can safely
    stitch and insert_transcripte as soon as all outstanding decode threads finish.

    Args:
        session_id: Session identifier from the commit command header.
        rec_idx: Index of the button press being committed.
    """
    _get_or_create_session(session_id, state)
    with state.lock:
        session = state.sessions.get(session_id)
        if session is None:
            return
        rec = session.get_or_create_recording(rec_idx)
        rec.closed = True

    finalize_recording(session_id, rec_idx, state)


# ---------------------------------------------------------------------------
# Command parsers
# ---------------------------------------------------------------------------


def _handle_session_event(audio: bytes, state: SessionStates) -> None:
    """Parse and route a ``CMD_SESSION_EVENT`` message to the telemetry layer.

    Wire format::

        CMD_SESSION_EVENT:<session_id>:<rec_idx>\\n\\n<json_object>

    The recording index is extracted from the header and injected into the
    JSON payload so the telemetry recorder can file the event under the
    correct recording slot.

    Args:
        audio: The full raw bytes of the incoming message, including header
            and payload, exactly as read from the socket.
    """
    try:
        header, payload_audio = audio.split(b"\n\n", 1)
        # Split into 3 parts: CMD, session_id, recording_index
        _, session_id, rec_idx_str = header.decode("utf-8").strip().split(":", 2)

        payload = json.loads(payload_audio.decode("utf-8"))
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
        _handle_session_telemetry_event(session_id, payload, state)
    except (KeyError, ValueError, TypeError) as e:
        log.warning("Bad session event command", error=str(e))


def _transcribe_raw_connection_audio(audio: bytes, t_connect: float, state: SessionStates) -> None:
    """Transcribe a legacy one-shot audio recording in a single pass.

    This path is used when the Ear sends the entire recording as a plain byte
    stream without any ``CMD_`` prefix.  The whole buffer is normalised,
    decoded once, refined by the LLM, and insert_transcripted.  No deduplication or
    chunk stitching is performed.

    Args:
        audio: The entire raw PCM recording as a flat byte buffer (16-bit
            little-endian, mono, ``settings.rate`` Hz).
        t_connect: Timestamp (from :func:`time.perf_counter`) captured when
            the connection was accepted, used to calculate total STT latency.
    """
    engine = state.backend.get_engine()

    if not engine:
        log.info("[Brain] ⚠️  No engine loaded — skipping")
        send_hud("hide")
        return

    try:
        audio_int16 = np.frombuffer(audio[: len(audio) // 2 * 2], dtype=np.int16)
        audio = _normalize_audio(audio_int16)

        log.info("[Brain] 🎙️  Final utterance decode")
        send_hud("process")
        final_text = engine.transcribe_chunk(audio)
        if not final_text:
            log.info("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return
    except (OSError, RuntimeError, ValueError) as e:
        log.info("[Brain] Audio decode error: %s", e)
        send_hud("hide")
        return

    stt_time = time.perf_counter() - t_connect

    t_refine_start = time.perf_counter()
    cleaned_text = refine(final_text)
    refine_time = time.perf_counter() - t_refine_start
    llm_log_str = f"{refine_time:.2f}s"

    log.info(f'[Brain] 📝 [STT: {stt_time:.2f}s | LLM: {llm_log_str}] → "{cleaned_text}"')

    insert_transcripte(cleaned_text + " ")
    _show_summary_table("one-shot", final_text, cleaned_text, stt_time, refine_time)

    send_hud("done")


# ---------------------------------------------------------------------------
# Paste
# ---------------------------------------------------------------------------


def _copy_to_clipboard(txt: bytes):
    with subprocess.Popen(
        args="pbcopy", shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        _, err = p.communicate(input=txt)
        if err:
            log.error(f"[copy_to_clipboard] FAIL : {err.decode()}")
            raise subprocess.SubprocessError(err.decode())


def _insert_transcripte_to_clipboard():
    try:
        complete_process = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True,
        )
        return f"status code : {complete_process.returncode}"

    except subprocess.CalledProcessError:
        raise


def insert_transcripte(text: str) -> None:
    """Insert ``text`` into the currently focused application.

    On **macOS** the fast path is:

    1. Save the current clipboard with ``pbinsert_transcripte``.
    2. Copy ``text`` into the clipboard with ``pbcopy``.
    3. Simulate ``Cmd+V`` via AppleScript (zero latency, no per-character typing).
    4. Restore the original clipboard after a 50 ms delay.

    On **Linux / Windows** or when the macOS path fails, falls back to
    :meth:`pynput.keyboard.Controller.type` which simulates individual
    keystrokes — slower but universally compatible.

    After insert_transcripting, the session-finished sound effect is played via
    :func:`_play_finish_sound`.

    Args:
        text: The final transcript string to insert.  A trailing space is
            typically appended by the caller so the cursor is positioned
            after the insert_transcripted text.
    """
    if platform.system() == "Darwin":
        try:
            old = subprocess.check_output(["pbinsert_transcripte"], stderr=subprocess.DEVNULL)

            _copy_to_clipboard(text.encode("utf-8"))
            _insert_transcripte_to_clipboard()

            time.sleep(0.05)
            _copy_to_clipboard(old)

            play = PlaySound("FINISH")
            play()

        except (subprocess.SubprocessError, OSError) as e:
            log.info("[Brain] macOS Fast-insert_transcripte failed: %s", e)


# ---------------------------------------------------------------------------
# Audio normalisation
# ---------------------------------------------------------------------------


def _normalize_audio(int16_audio: np.ndarray) -> Optional[np.ndarray]:
    """Convert raw 16-bit PCM to a normalised ``float32`` array.

    The transcription engines expect audio in the range ``[-1.0, 1.0]``.
    This function:

    * Returns ``None`` for empty arrays or arrays whose peak amplitude is
      below the silence threshold (< 0.001), so silent chunks are dropped
      before hitting the engine.
    * Peak-normalises louder chunks to 0.9 to ensure consistent input
      levels across different microphone gains.
    * Leaves very quiet but non-silent chunks (peak 0.001–0.01) unnormalised
      to avoid amplifying noise into the engine.

    Args:
        int16_audio: 1-D NumPy array of raw 16-bit PCM samples.

    Returns:
        A ``float32`` NumPy array scaled to ``[-1.0, 1.0]``, or ``None``
        when the chunk is empty or completely silent.
    """
    if len(int16_audio) == 0:
        return None
    audio = int16_audio.astype(np.float32) / 32768.0
    max_val = float(np.max(np.abs(audio)))
    if max_val < 0.001:
        return None
    return audio / max_val * 0.9 if max_val > 0.01 else audio


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


def handle_connection(conn: socket.socket, state: SessionStates) -> None:
    """Read a full incoming message and dispatch it to the correct handler.

    Each connection from the Ear is handled on its own thread.  The function
    reads until the remote side closes the connection, assembles the bytes
    into a single buffer, then dispatches based on the command prefix:

    ==============================  ===================================
    Prefix                          Handler
    ==============================  ===================================
    ``CMD_SWITCH_MODEL:``           :func:`_handle_switch_model`
    ``CMD_SESSION_COMMIT:``         :func:`_handle_session_commit`
    ``CMD_SESSION_EVENT:`` + ``\\n\\n``  :func:`_handle_session_event`
    ``CMD_AUDIO_CHUNK:`` + ``\\n\\n``    :func:`_handle_chunk_command`
    *(anything else)*               :func:`_transcribe_raw_connection_audio`
    ==============================  ===================================

    Args:
        conn: An accepted :class:`socket.socket` from the Unix domain server.
            Ownership of this socket is transferred to this function; it will
            be closed before the function returns.
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
    except OSError as e:
        log.info("[Brain] Recv error: %s", e)
    finally:
        conn.close()

    if not raw_audio:
        return
    audio = bytes(raw_audio)

    if audio.startswith(b"CMD_SWITCH_MODEL:"):
        _handle_switch_model(audio, state)
    elif audio.startswith(b"CMD_SESSION_COMMIT:"):
        _handle_session_commit(audio, state)
    elif audio.startswith(b"CMD_SESSION_EVENT:") and b"\n\n" in audio:
        _handle_session_event(audio, state)
    elif audio.startswith(b"CMD_AUDIO_CHUNK:") and b"\n\n" in audio:
        _handle_chunk_command(audio, state)
    else:
        _transcribe_raw_connection_audio(audio, t_connect, state)


# ---------------------------------------------------------------------------
# High-level command handlers
# ---------------------------------------------------------------------------


def _handle_switch_model(audio: bytes, state: SessionStates) -> None:
    """Swap the active transcription engine for a different model.

    Parses the new model name from the command bytes, unloads the current
    engine (calling ``gc.collect()`` to immediately free GPU/CPU memory),
    loads the new engine, and clears the session store so stale session
    state from the old model does not leak into new recordings.

    Wire format::

        CMD_SWITCH_MODEL:<model_name>

    Args:
        audio: The full raw bytes of the ``CMD_SWITCH_MODEL`` message.
    """
    try:
        new_model = audio.decode("utf-8").strip().split(":", 1)[1]
        log.info("🔄 Switching model", model=new_model)
        state.backend.set_engine(None)  # unload first to free memory
        gc.collect()
        state.backend.load_tts(new_model)

        with state.lock:
            state.sessions.clear()

        log.info("✅ Model switched", model=new_model)
    except (KeyError, ValueError, RuntimeError) as e:
        log.error("Switching failed", error=str(e))


def _handle_session_commit(audio: bytes, state: SessionStates) -> None:
    """Signal that a button press is complete and trigger finalization.

    Parses the session ID and recording index from the commit command, then
    calls :func:`_mark_session_closed` which sets ``rec.closed = True`` and
    attempts to finalize the recording immediately if all chunks are done.

    Wire format::

        CMD_SESSION_COMMIT:<session_id>:<rec_idx>

    Args:
        audio: The full raw bytes of the ``CMD_SESSION_COMMIT`` message.
    """
    try:
        _, session_id, rec_idx_str = audio.decode("utf-8").strip().split(":", 2)
        rec_idx = int(rec_idx_str)
        log.info("✅ Session commit received", session=session_id[:8], recording=rec_idx)
        _mark_session_closed(session_id, rec_idx, state)
    except (KeyError, ValueError, TypeError):
        log.warning("Bad commit command")


def _handle_chunk_command(audio: bytes, state: SessionStates) -> None:
    """Parse a ``CMD_AUDIO_CHUNK`` message and route it to the chunk handler.

    Splits the header from the raw PCM payload, extracts the session ID,
    recording index, and sequence number, then calls
    :func:`handle_streaming_audio_chunk` which runs the transcription engine.

    Wire format::

        CMD_AUDIO_CHUNK:<session_id>:<rec_idx>:<seq>\\n\\n<raw_pcm_bytes>

    Args:
        audio: The full raw bytes of the ``CMD_AUDIO_CHUNK`` message,
            including the ``\\n\\n`` separator and trailing PCM payload.
    """
    header, audio_bytes = audio.split(b"\n\n", 1)
    try:
        _, session_id, rec_idx_str, seq_text = header.decode("utf-8").strip().split(":", 3)
        log.info(
            "🎙️  Audio chunk received",
            rec=rec_idx_str,
            seq=seq_text,
            size=len(audio_bytes),
            session=session_id[:8],
        )
        handle_streaming_audio_chunk(
            session_id, int(rec_idx_str), int(seq_text), audio_bytes, state
        )
    except (ValueError, TypeError) as e:
        log.warning("Bad chunk header", error=str(e))


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def start_server() -> None:
    """Boot the Brain server and block while serving incoming connections.

    Startup sequence
    ~~~~~~~~~~~~~~~~
    1. Fix macOS dynamic library paths via :func:`~src.utils.bootstrap.fix_macos_library_paths`.
    2. Configure the LLM text-refiner provider from ``settings.vibevoice_provider_index``.
    3. Load the STT engine specified by ``settings.stt_model``.
    4. Warm the engine with a silent audio frame so the first real request
       has no cold-start latency.
    5. Bind a Unix domain socket at ``settings.ear_to_brain_socket_path``
       and enter the accept loop, spawning a daemon thread per connection.

    Shutdown
    ~~~~~~~~
    ``KeyboardInterrupt`` (Ctrl-C) breaks the accept loop.  The server
    flushes telemetry for every open session, closes the Unix socket file,
    and shuts down the LLM router's HTTP connection pool before exiting.
    """
    # 1. Auto-fix environment issues (e.g., macOS library paths)
    breakpoint()
    fix_macos_library_paths()

    safe_provider_index = min(settings.vibevoice_provider_index, len(PROVIDERS) - 1)
    set_provider(safe_provider_index)
    log.info(f"[Brain] Text refiner set to: {PROVIDERS[safe_provider_index]['name']}")
    state = SessionStates()
    backend = BackendState(model_name=settings.stt_model)
    state.backend = backend
    log.info(f"[Brain] Warming up model: {settings.stt_model}...")
    try:
        state.backend.get_engine().transcribe_chunk(np.zeros(8000, dtype=np.float32))
    except (RuntimeError, OSError) as e:
        log.error("[Brain] Warm-up failed %s", e)
        sys.exit(1)
    log.info("[Brain] Warm-up done ✓")

    if os.path.exists(settings.ear_to_brain_socket_path):
        os.remove(settings.ear_to_brain_socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(settings.ear_to_brain_socket_path)
    server.listen(10)
    log.info(f"[Brain] ✅ Streaming server ready at {settings.ear_to_brain_socket_path}")

    try:
        while True:
            readable, _, _ = select.select([server], [], [], 0.4)
            if server in readable:
                conn, _ = server.accept()
                threading.Thread(target=handle_connection, args=(conn, state), daemon=True).start()
            continue

    except KeyboardInterrupt:
        log.info("\n[Brain] Shutting down...")
        try:
            with state.lock:
                for session_id, session_state in state.sessions.items():
                    update_summary(
                        session_id,
                        {
                            "total_chunks_received": sum(
                                r.received_count for r in session_state.recordings.values()
                            ),
                            "total_decode_seconds": round(session_state.stt_time, 2),
                        },
                        state,
                    )
        except (OSError, ValueError, TypeError) as e:
            log.warning("Failed to write shutdown telemetry: %s", e)
    finally:
        server.close()
        if os.path.exists(settings.ear_to_brain_socket_path):
            os.remove(settings.ear_to_brain_socket_path)

        # Close the LLM router connection pool safely
        try:
            from src.text_refiner.llm_router import global_http_client

            global_http_client.close()
            log.info("[Brain] 🌐 LLM Router connection pool closed")
        except (ImportError, AttributeError) as e:
            log.warning("Failed to close LLM router client: %s", e)


if __name__ == "__main__":
    start_server()
