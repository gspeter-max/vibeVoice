"""Recording mechanics for the Ear runtime.

This module owns the execution-heavy parts of Ear recording flow. The
controller keeps the `Ear` object and decides when to call these helpers,
while this module performs the lower-level work.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pyaudio

from src import log
from src.audio.ear_runtime.analysis import (
    analyze_frequency_bands,
    boost_audio_chunk,
    get_rms as runtime_get_rms,
)
from src.audio.ear_runtime.system_audio import play_start_sound
from src.ipc.client import (
    close_raw_audio_stream_and_forget,
    send_message_to_brain,
    send_raw_audio_stream_chunk_or_close,
)
from src.ipc.protocol import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
)
from src.streaming.session import should_split_chunk_after_silence
from src.ui.hud_client import start_hud_command_thread
from src.utils.settings import settings


def begin_recording_session(ear) -> None:
    """Start one controller-managed recording session."""

    ear._capture_session.begin_recording(time.time())
    send_session_event_to_brain(
        ear,
        "session_started",
        {"recording_mode": settings.recording_mode},
    )


def send_session_event_to_brain(
    ear,
    event_type: str,
    fields: dict | None = None,
) -> bool:
    """Send one Ear session event over the Brain socket."""

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
    sent = send_message_to_brain(
        message_bytes,
        timeout_seconds=5.0,
        socket_path=settings.socket_path,
        socket_factory=socket.socket,
    )
    if not sent:
        log.info(f"[Ear] ❌ Failed to send telemetry event '{event_type}'")
    return sent


def send_audio_chunk_to_brain(ear, utterance_bytes: bytes) -> bool:
    """Format and send one processed audio chunk to Brain."""

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
    sent = send_message_to_brain(
        message_bytes,
        timeout_seconds=5.0,
        socket_path=settings.socket_path,
        socket_factory=socket.socket,
    )
    if sent:
        send_session_event_to_brain(
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


def commit_recording_session(ear) -> bool:
    """Send the final commit signal for the current recording."""

    if not ear._capture_session.current_session_id:
        return False

    message_bytes = format_session_commit_message(
        ear._capture_session.current_session_id,
        ear._capture_session.current_recording_index,
    )
    sent = send_message_to_brain(
        message_bytes,
        timeout_seconds=5.0,
        socket_path=settings.socket_path,
        socket_factory=socket.socket,
    )
    if sent:
        log.info(
            "✅ Session committed",
            session=ear._capture_session.current_session_id[:8],
            recording=ear._capture_session.current_recording_index,
        )
        ear._capture_session.mark_recording_committed()
        return True

    log.error("❌ Failed to commit session")
    return False


def reset_chunk_tracking(ear) -> None:
    """Reset per-chunk logging flags on the Ear instance."""

    ear._chunk_speech_logged = False
    ear._silence_pending_logged = False
    ear._vad_no_speech_warned = False


def flush_current_chunk(ear, *, stop_session: bool) -> bool:
    """Finalize the active utterance and send it to Brain if present."""

    now_seconds = time.time()
    silence_elapsed_seconds = ear._utterance_gate.silence_elapsed(now_seconds)

    with ear._lock:
        if not ear.is_recording:
            return False
        total_frames = ear._total_frames
        if stop_session:
            ear.is_recording = False
        ear._total_frames = 0
        ear.last_rms = 0.0
        reset_chunk_tracking(ear)

    utterance_bytes = ear._utterance_gate.flush()
    if not utterance_bytes:
        if stop_session:
            ear._capture_session.mark_recording_stopped()
            log.info("[Ear] 🔇 No speech captured; stopping recording")
            commit_recording_session(ear)
        return False

    boosted_utterance_bytes = boost_audio_chunk(utterance_bytes, ear.gain_multiplier)
    previous_chunk_tail_bytes = ear._capture_session.last_chunk_tail_bytes
    overlapped_utterance_bytes = ear._capture_session.prepare_chunk_for_send(
        boosted_utterance_bytes,
        stop_session=stop_session,
        silence_seconds=silence_elapsed_seconds if not stop_session else 0.0,
    )
    overlap_seconds_added = len(previous_chunk_tail_bytes) / 2.0 / settings.rate
    chunk_age_seconds = ear._capture_session.current_chunk_age_seconds(now_seconds)
    duration_seconds = (total_frames * settings.chunk) / settings.rate

    if stop_session:
        log.info(
            f"\r\n⏹️  Streamed {duration_seconds:.1f}s ({total_frames} chunks) — Brain transcribing...\n"
        )
        start_hud_command_thread("process", socket_factory=socket.socket)
    else:
        log.info(
            f"\r[Ear] ✂️  Silence boundary hit ({silence_elapsed_seconds:.2f}s) — sending chunk "
            f"{duration_seconds:.1f}s ({total_frames} chunks)"
        )

    sent = send_audio_chunk_to_brain(ear, overlapped_utterance_bytes)
    if sent:
        send_session_event_to_brain(
            ear,
            "silence_threshold_hit" if not stop_session else "session_stopped",
            {
                "chunk_index": ear._capture_session.current_chunk_sequence_number - 1,
                "chunk_age_seconds": round(chunk_age_seconds, 2),
                "silence_elapsed_seconds": round(silence_elapsed_seconds, 2),
                "split_reason": "silence_threshold_hit" if not stop_session else "session_stop",
                "overlap_seconds_added": round(overlap_seconds_added, 4),
                "audio_bytes": len(overlapped_utterance_bytes),
            },
        )

    if stop_session:
        ear._capture_session.mark_recording_stopped()
        commit_recording_session(ear)
        close_mic_stream(ear)
    else:
        ear._capture_session.mark_nonfinal_chunk_sent()
    return sent


def stop_no_streaming(ear) -> None:
    """Stop a no-streaming recording and close the raw Brain stream."""

    with ear._lock:
        if not ear.is_recording:
            return
        ear.is_recording = False
        total_frames = ear._total_frames
        ear._total_frames = 0
        ear.last_rms = 0.0

    duration_seconds = (total_frames * settings.chunk) / settings.rate
    log.info(
        f"\r\n⏹️  Streamed {duration_seconds:.1f}s ({total_frames} chunks) — Brain transcribing...\n"
    )
    start_hud_command_thread("process", socket_factory=socket.socket)
    with ear._brain_sock_lock:
        raw_stream_socket = ear._brain_sock
        ear._brain_sock = None
    threading.Thread(
        target=close_raw_audio_stream_and_forget,
        args=(raw_stream_socket,),
        daemon=True,
    ).start()
    close_mic_stream(ear)


def open_mic_stream(ear) -> None:
    """Open the PyAudio input stream for the current Ear instance."""

    if ear.stream is not None:
        try:
            ear.stream.stop_stream()
            ear.stream.close()
        except OSError:
            pass

    ear.stream = ear.pyaudio_library_for_capturing_audio.open(
        format=settings.audio_format,
        channels=settings.channels,
        rate=settings.rate,
        input=True,
        input_device_index=ear.input_device_index,
        frames_per_buffer=settings.chunk,
        stream_callback=ear._audio_callback,
    )
    log.info("[Ear] 🎤 Mic stream opened")


def close_mic_stream(ear) -> None:
    """Close the active PyAudio input stream if one exists."""

    if ear.stream is None:
        return

    try:
        ear.stream.stop_stream()
        ear.stream.close()
    except OSError:
        pass
    ear.stream = None


def start_recording_state(ear, *, from_hold: bool) -> None:
    """Reset Ear state for a fresh recording."""

    del from_hold
    play_start_sound()
    open_mic_stream(ear)

    with ear._lock:
        ear.is_recording = True
        ear.last_rms = 0.0
        ear._total_frames = 0
        reset_chunk_tracking(ear)
        ear._recording_level_log_time = 0.0

    ear._capture_session.clear_overlap_tail()
    if settings.is_silence_streaming_mode:
        ear._utterance_gate.reset()
        begin_recording_session(ear)


def process_audio_callback(ear, in_data, frame_count, time_info, status):
    """Handle one microphone callback for the active Ear mode."""

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
                score = ear._utterance_gate.last_score()
                energy = ear._utterance_gate.last_energy()
                dynamic_threshold = ear._utterance_gate.last_dynamic_threshold()
                started = ear._utterance_gate.has_speech_started()
                silence_elapsed_seconds = (
                    ear._utterance_gate.silence_elapsed(now_seconds) if started else 0.0
                )
                log.debug(
                    "[Ear] 🔎 "
                    f"VAD score={score:.3f} "
                    f"threshold={settings.vad_score_threshold:.2f} "
                    f"started={started} silence={silence_elapsed_seconds:.2f}s "
                    f"rms={ear.last_rms:.4f} "
                    f"energy={energy:.4f} energy_threshold={dynamic_threshold:.4f}",
                )
            except (OSError, ValueError, TypeError):
                pass
            ear._vad_state_log_time = now_seconds

    return (None, pyaudio.paContinue)


def record_loop_tick(ear, input_trigger=None) -> None:
    """Run one controller-managed recording loop tick."""

    with ear._lock:
        recording = ear.is_recording
        rms = ear.last_rms

    if input_trigger is not None:
        input_trigger.check_mouse_hold_threshold()

    if not recording:
        return

    now_seconds = time.time()
    if now_seconds - ear._recording_level_log_time >= settings.recording_level_log_interval:
        meter_width = 30
        level = min(int(rms * 300), meter_width)
        meter = "█" * level + "░" * (meter_width - level)
        ear._recording_level_log_time = now_seconds
        print(f"\r  Voice Level: [{meter}] ", end="", flush=True)

    if not settings.is_silence_streaming_mode:
        return

    if "nemotron" in ear.current_model.lower():
        time_since_last_chunk = ear._capture_session.current_chunk_age_seconds(now_seconds)
        if time_since_last_chunk >= 1.12:
            ear._stop_and_send(stop_session=False)
        return

    if ear._utterance_gate.has_speech_started() and not ear._silence_pending_logged:
        silence_elapsed_seconds = ear._utterance_gate.silence_elapsed(now_seconds)
        if silence_elapsed_seconds > 0.0:
            ear._silence_pending_logged = True

    silence_elapsed_seconds = (
        ear._utterance_gate.silence_elapsed(now_seconds)
        if ear._utterance_gate.has_speech_started()
        else ear._utterance_gate.finalize_elapsed(now_seconds)
    )
    split_decision = should_split_chunk_after_silence(
        chunk_started_at_seconds=ear._capture_session.chunk_started_at_seconds,
        now_seconds=now_seconds,
        minimum_chunk_age_before_silence_split_seconds=(
            settings.minimum_chunk_age_before_silence_split_seconds
        ),
        utterance_gate_should_finalize_now=ear._utterance_gate.should_finalize(now_seconds),
        silence_duration_seconds=silence_elapsed_seconds,
    )
    if split_decision.should_split_now:
        ear._stop_and_send(stop_session=False)
