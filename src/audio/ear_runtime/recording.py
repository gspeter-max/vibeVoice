"""Recording mechanics for the Ear runtime.

This module owns the execution-heavy parts of Ear recording flow. The
controller keeps the `Ear` object and decides when to call these helpers,
while this module performs the lower-level work.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.audio.ear_runtime.controller import Ear


import numpy as np
import pyaudio

from sound import PlaySound
from src import log
from src.audio.ear_runtime.analysis import (
    analyze_frequency_bands,
    boost_audio_chunk,
)
from src.audio.ear_runtime.analysis import (
    get_rms as runtime_get_rms,
)
from src.audio.ear_runtime.system_audio import play_start_sound
from src.ipc.client import (
    close_raw_audio_stream_to_brain,
    send_message,
    send_raw_audio_stream_chunk_or_close,
)
from src.ipc.protocol_message_formats import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
)
from src.ui.hud_client import change_ui_status
from src.utils.settings import settings


def begin_recording_session(ear: Ear) -> None:
    """Start a new recording session on the CaptureSession manager.

    This function marks the start time of a recording event and sends a telemetry
    signal to the Brain backend. It initializes the session telemetry logging
    based on the current recording mode configured in settings.

    Args:
        ear: The Ear controller instance.
    """
    ear._capture_session.begin_recording(time.time())
    send_session_event_to_telemetry_brain(
        ear,
        "session_started",
        {"recording_mode": settings.recording_mode},
    )


def send_session_event_to_telemetry_brain(
    ear: Ear,
    event_type: str,
    fields: dict | None = None,
) -> bool:
    """Send an Ear runtime telemetry event over the Telemetry Brain socket.

    If telemetry is disabled or there is no active session ID, it returns early.
    Otherwise, it packages the event metadata along with the session details
    and sends them to the Brain via an IPC socket.

    Args:
        ear: The Ear controller instance.
        event_type: The name/type of the telemetry event.
        fields: Optional dictionary containing event details.

    Returns:
        True if the event was successfully sent, False otherwise.
    """
    if not ear._telemetry_enabled or not ear._capture_session.current_session_id:
        return False

    payload = {"type": event_type}
    if fields:
        payload.update(fields)
    message_bytes = format_session_event_message(
        ear._capture_session.current_session_id,
        ear._capture_session.current_recording_index,
        payload,
    )
    sent = send_message(
        message_bytes,
        timeout_seconds=5.0,
        socket_factory=socket.socket,
    )
    if not sent:
        log.info(f"[Ear] ❌ Failed to send telemetry event '{event_type}' to telemetry brain")
    return sent


def send_audio_chunk_to_brain(ear: Ear, utterance_bytes: bytes) -> bool:
    """Format and send a processed chunk of audio data to the Brain backend.

    This function formats the audio bytes into the IPC message protocol, increments
    the sequence counter on the capture session, and streams it over the socket.
    It also sends a corresponding telemetry event to trace chunk latency and size.

    Args:
        ear: The Ear controller instance.
        utterance_bytes: Raw audio data bytes to send.

    Returns:
        True if the chunk was successfully sent, False otherwise.
    """
    if not utterance_bytes or not ear._capture_session.current_session_id:
        return False

    session_id = ear._capture_session.current_session_id
    recording_index = ear._capture_session.current_recording_index
    sequence_number = ear._capture_session.mark_chunk_sent()
    message_bytes = format_audio_chunk_message(
        session_id,
        recording_index,
        sequence_number,
        utterance_bytes,
    )
    sent = send_message(
        message_bytes,
        timeout_seconds=5.0,
        socket_factory=socket.socket,
    )
    if sent:
        send_session_event_to_telemetry_brain(
            ear,
            "chunk_sent_to_brain",
            {
                "chunk_index": sequence_number,
                "audio_bytes": len(utterance_bytes),
            },
        )
        return True

    log.error("❌ Failed to send chunk")
    return False


def commit_session_recording_stoped(ear: Ear) -> bool:
    """Commit the end of the current recording session to the Brain backend.

    This function sends the final commit message to close the session's stream,
    logs the final recording stats, and marks the session as fully committed
    in the CaptureSession model to prevent further chunk transmissions.

    Args:
        ear: The Ear controller instance.

    Returns:
        True if the commit was successfully sent, False otherwise.
    """
    if not ear._capture_session.current_session_id:
        return False

    message_bytes = format_session_commit_message(
        ear._capture_session.current_session_id,
        ear._capture_session.current_recording_index,
    )
    sent = send_message(
        message_bytes,
        timeout_seconds=5.0,
        socket_factory=socket.socket,
    )
    if sent:
        log.info(
            "✅ Session recording stop committed",
            session=ear._capture_session.current_session_id[:8],
            recording=ear._capture_session.current_recording_index,
        )
        ear._capture_session.mark_recording_committed()
        return True

    log.error("❌ Failed to commit session recording stop")
    return False


def reset_chunk_tracking(ear: Ear) -> None:
    """Reset the temporary status logging flags on the Ear controller.

    These boolean flags are used to throttle repetitive logs (like VAD speech
    detected warnings or silence alerts) during a recording loop tick.
    Resetting them ensures clean logs for the next recording session.

    Args:
        ear: The Ear controller instance.
    """
    ear._chunk_speech_logged = False
    ear._silence_pending_logged = False
    ear._vad_no_speech_warned = False


def flush_current_chunk(ear: Ear, *, stop_session: bool) -> bool:
    """Finalize the active utterance and send it to the Brain backend.

    This function extracts the buffered audio bytes from the utterance gate,
    applies the volume gain boost, appends overlap data from the previous
    chunk, and transmits the resulting package over the IPC socket."""

    now_seconds = time.time()
    silence_len = ear._utterance_gate.silence_len(now_seconds)

    with ear._lock:
        if not ear.is_recording:
            return False
        total_frames = ear._total_frames
        if stop_session:
            ear.is_recording = False
        ear._total_frames = 0
        ear.last_rms = 0.0
        reset_chunk_tracking(ear)

    utterance_bytes = ear._utterance_gate.flush
    if not utterance_bytes:
        if stop_session:
            ear._capture_session.mark_recording_stopped()
            log.info("[Ear] 🔇 No speech captured; stopping recording")
            commit_session_recording_stoped(ear)
        return False

    boosted_utterance_bytes = boost_audio_chunk(utterance_bytes, ear.gain_multiplier)
    previous_chunk_tail_bytes = ear._capture_session.last_chunk_tail_bytes
    overlapped_utterance_bytes = ear._capture_session.prepare_chunk_for_send(
        boosted_utterance_bytes,
        stop_session=stop_session,
        silence_seconds=silence_len if not stop_session else 0.0,
    )
    overlap_seconds_added = len(previous_chunk_tail_bytes) / 2.0 / settings.rate
    chunk_age_seconds = ear._capture_session.current_chunk_age_seconds(now_seconds)
    duration_seconds = (total_frames * settings.chunk) / settings.rate

    if stop_session:
        log.info(
            f"\r\n⏹️  Streamed {duration_seconds:.1f}s ({total_frames} chunks) — Brain transcribing...\n"
        )
        change_ui_status("process", socket_factory=socket.socket)
    else:
        log.info(
            f"\r[Ear] ✂️  Silence boundary hit ({silence_len:.2f}s) — sending chunk "
            f"{duration_seconds:.1f}s ({total_frames} chunks)"
        )

    sent = send_audio_chunk_to_brain(ear, overlapped_utterance_bytes)
    if sent:
        send_session_event_to_telemetry_brain(
            ear,
            "silence_threshold_hit" if not stop_session else "session_stopped",
            {
                "chunk_index": ear._capture_session.current_chunk_sequence_number - 1,
                "chunk_age_seconds": round(chunk_age_seconds, 2),
                "silence_elapsed_seconds": round(silence_len, 2),
                "split_reason": "silence_threshold_hit" if not stop_session else "session_stop",
                "overlap_seconds_added": round(overlap_seconds_added, 4),
                "audio_bytes": len(overlapped_utterance_bytes),
            },
        )

    if stop_session:
        ear._capture_session.mark_recording_stopped()
        commit_session_recording_stoped(ear)
        close_mic_stream(ear)
    else:
        ear._capture_session.mark_nonfinal_chunk_sent()
    return sent


def open_mic_stream(ear: Ear) -> None:
    """Open the system microphone input stream using the PyAudio library.

    This function releases any existing microphone stream before initializing
    a new PyAudio stream. It binds the stream to the designated mic device index
    and configures the real-time audio callback to process incoming frames.

    Args:
        ear: The Ear controller instance.
    """
    if ear.stream is not None:
        try:
            ear.stream.stop_stream()
            ear.stream.close()
        except OSError:
            pass

    ear.stream = ear.pyaudio_inst.open(
        format=settings.audio_format,
        channels=settings.channels,
        rate=settings.rate,
        input=True,
        input_device_index=ear.input_device_index,
        frames_per_buffer=settings.chunk,
        stream_callback=ear._audio_callback,
    )
    log.info("[Ear] 🎤 Mic stream opened")


def close_mic_stream(ear: Ear) -> None:
    """Close and release the active PyAudio microphone input stream.

    If a stream is currently active, it stops audio capture, closes the stream
    interface, and sets the stream attribute back to None. It catches and
    silences any OS-level errors during the teardown process.

    Args:
        ear: The Ear controller instance.
    """
    if ear.stream is None:
        return

    try:
        ear.stream.stop_stream()
        ear.stream.close()
    except OSError:
        pass
    ear.stream = None


def start_recording_state(ear: Ear, *, from_hold: bool) -> None:
    """Reset and prepare the Ear controller state for a new recording run.

    This function plays the audible start notification sound, initializes the
    microphone input stream, and sets the active recording flags. If the app
    is in silence-streaming mode, it also clears the VAD utterance gate.

    Args:
        ear: The Ear controller instance.
        from_hold: If True, indicates recording was started via mouse hold.
    """
    del from_hold
    play = PlaySound("STARTING")
    play()
    open_mic_stream(ear)

    with ear._lock:
        ear.is_recording = True
        ear.last_rms = 0.0
        ear._total_frames = 0
        reset_chunk_tracking(ear)
        ear._recording_level_log_time = 0.0

    ear._capture_session.clear_overlap_tail()
    if settings.is_silence_streaming_mode:
        ear._utterance_gate.reset
        begin_recording_session(ear)


def process_audio_callback(
    ear: Ear,
    in_data: bytes,
    frame_count: int,
    time_info: dict[str, float],
    status: int,
) -> tuple[bytes | None, int]:
    """Process a single real-time microphone callback tick from PyAudio.

    This is the core audio intake function. It boosts the audio gain, computes
    root-mean-square volume, runs VAD checks in streaming mode, or feeds raw
    audio directly to the Brain connection socket in non-streaming mode.

    Args:
        ear: The Ear controller instance.
        in_data: The raw recorded audio chunk bytes.
        frame_count: Number of frames in this chunk.
        time_info: Dict containing timing metadata from PyAudio.
        status: Status flags from PyAudio.

    Returns:
        A tuple of (None, continue_flag) indicating stream state.
    """
    del frame_count, time_info, status

    with ear._lock:
        if not ear.is_recording:
            return (None, pyaudio.paContinue)

        boosted_chunk_bytes = boost_audio_chunk(in_data, ear.gain_multiplier)
        boosted_samples = np.frombuffer(boosted_chunk_bytes, dtype=np.int16)
        ear.last_rms = runtime_get_rms(boosted_chunk_bytes)
        ear._total_frames += 1
        ear.last_frequency_bands = analyze_frequency_bands(
            boosted_samples,
            sample_rate=settings.rate,
        )

        if settings.is_no_streaming_mode:
            with ear._brain_sock_lock:
                ear._brain_sock = send_raw_audio_stream_chunk_or_close(
                    ear._brain_sock,
                    boosted_chunk_bytes,
                )
                raw_stream_socket_alive = ear._brain_sock is not None
            if not raw_stream_socket_alive:
                log.info("\r⚠️  Brain disconnected — will transcribe on release\n")
            return (None, pyaudio.paContinue)

        now_seconds = time.time()
        speech_now = ear._utterance_gate.push(
            audio_chunk=in_data,
            now=now_seconds,
            analysis_chunk=boosted_chunk_bytes,
        )

        if speech_now and not ear._chunk_speech_logged:
            ear._chunk_speech_logged = True
            ear._silence_pending_logged = False
            log.info("[Ear] 🗣️  VAD speech detected")

        if now_seconds - ear._vad_state_log_time >= settings.vad_status_log_interval:
            try:
                score = ear._utterance_gate.last_score
                energy = ear._utterance_gate.last_energy
                dynamic_threshold = ear._utterance_gate.last_dynamic_threshold
                started = ear._utterance_gate.has_speech_started
                silence_len = ear._utterance_gate.silence_len(now_seconds) if started else 0.0
                log.debug(
                    "[Ear] 🔎 "
                    f"VAD score={score:.3f} "
                    f"threshold={settings.vad_score_threshold:.2f} "
                    f"started={started} silence={silence_len:.2f}s "
                    f"rms={ear.last_rms:.4f} "
                    f"energy={energy:.4f} energy_threshold={dynamic_threshold:.4f}",
                )
            except (OSError, ValueError, TypeError):
                pass
            ear._vad_state_log_time = now_seconds

    return (None, pyaudio.paContinue)
