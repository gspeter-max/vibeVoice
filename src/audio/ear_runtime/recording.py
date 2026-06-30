"""Recording mechanics for the Ear runtime.

This module owns the execution-heavy parts of Ear recording flow. The
controller keeps the `Ear` object and decides when to call these helpers,
while this module performs the lower-level work.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

from src.audio.vad_segmenter import SileroUtteranceGate
from src.interfaces import EarProtocol
from src.streaming.capture_session import CaptureSession


@dataclass
class LogState:
    """Throttle flags and timers for recording loop debug output."""

    chunk_speech_logged: bool = False
    silence_pending_logged: bool = False
    vad_no_speech_warned: bool = False
    vad_state_log_time: float = 0.0
    level_log_time: float = 0.0

    @property
    def reset(self) -> "LogState":
        """Reset all flags and timers back to their initial state."""
        self.chunk_speech_logged = False
        self.silence_pending_logged = False
        self.vad_no_speech_warned = False
        self.vad_state_log_time = 0.0
        self.level_log_time = 0.0
        return self


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

from src.ipc.client import (
    SocketConfig,
    commit_stop,
    send_audio,
    send_event,
    send_message,
)
from src.ui.hud_client import change_ui_status
from src.utils.settings import settings


def begin_recording_session(session: CaptureSession, telemetry_enabled: bool) -> None:
    """Start a new recording session on the CaptureSession manager.

    This function marks the start time of a recording event and sends a telemetry
    signal to the Brain backend. It initializes the session telemetry logging
    based on the current recording mode configured in settings.

    Args:
        ear: The Ear controller instance.
    """
    session.begin(time.time())
    send_event(
        session=session,
        telemetry_enabled=telemetry_enabled,
        event_type="session_started",
        fields={"recording_mode": settings.recording_mode},
    )





def reset_chunk_tracking(log_state: LogState) -> None:
    """Reset all throttle flags on the LogState for the next recording session."""
    log_state.reset


def flush_current_chunk(
    ear: EarProtocol,
    session: CaptureSession,
    utr_gate: SileroUtteranceGate,
    log_state: LogState,
    stop_session: bool,
    telemetry_enabled: bool,
) -> bool:
    """Finalize the active utterance and send it to the Brain backend.

    This function extracts the buffered audio bytes from the utterance gate,
    applies the volume gain boost, appends overlap data from the previous
    chunk, and transmits the resulting package over the IPC socket."""

    now_seconds = time.time()
    silence_len = utr_gate.silence_len(now_seconds)

    with ear.lock:
        if not ear.is_recording:
            return False
        total_frames = ear.total_frames
        if stop_session:
            ear.is_recording = False
        ear.total_frames = 0
        ear.last_rms = 0.0
        reset_chunk_tracking(log_state)

    if not utr_gate.flush:
        if stop_session:
            session.stop()
            log.info("[Ear] 🔇 No speech captured; stopping recording")
            commit_stop(session=session)
        return False

    overlapped_utterance_bytes = session.prep_chunk(
        boost_audio_chunk(utr_gate.flush, ear.gain_multiplier),
        stop=stop_session,
        silence_seconds=silence_len if not stop_session else 0.0,
    )
    overlap_seconds_added = len(session.tail) / 2.0 / settings.rate
    duration_seconds = (total_frames * settings.chunk) / settings.rate

    if stop_session:
        log.info(
            f"\r\n⏹️  Streamed {duration_seconds:.1f}s ({total_frames} chunks) — Brain transcribing...\n"
        )
        change_ui_status("process")
    else:
        log.info(
            f"\r[Ear] ✂️  Silence boundary hit ({silence_len:.2f}s) — sending chunk "
            f"{duration_seconds:.1f}s ({total_frames} chunks)"
        )

    sent = send_audio(
        session=session, audio_bytes=overlapped_utterance_bytes, telemetry_enabled=telemetry_enabled
    )
    if sent:
        send_event(
            session=session,
            telemetry_enabled=telemetry_enabled,
            event_type="silence_threshold_hit" if not stop_session else "session_stopped",
            fields={
                "chunk_index": session.chunk_seq - 1,
                "chunk_age_seconds": round(session.chunk_age, 2),
                "silence_elapsed_seconds": round(silence_len, 2),
                "split_reason": "silence_threshold_hit" if not stop_session else "session_stop",
                "overlap_seconds_added": round(overlap_seconds_added, 4),
                "audio_bytes": len(overlapped_utterance_bytes),
            },
        )

    if stop_session:
        session.stop()
        commit_stop(session)
        ear.close_mic_stream
    else:
        session.mark_next()
    return sent


def open_mic_stream(ear: EarProtocol, utr_gate: SileroUtteranceGate, log_state: LogState) -> None:
    """Open the system microphone input stream using the PyAudio library."""
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
        # PyAudio's background thread expects a specific 4-argument callback signature:
        # (in_data, frame_count, time_info, status). The lambda satisfies this requirement
        # and forwards only the necessary components to our processing function.
        stream_callback=lambda in_data, frame_count, time_info, status: process_audio_callback(
            ear, utr_gate, log_state, in_data
        ),
    )
    log.info("[Ear] 🎤 Mic stream opened")


def start_recording_state(
    ear: EarProtocol,
    session: CaptureSession,
    utr_gate: SileroUtteranceGate,
    log_state: LogState,
    telemetry_enabled: bool,
) -> None:
    """Reset and prepare the Ear controller state for a new recording run.

    This function plays the audible start notification sound, initializes the
    microphone input stream, and sets the active recording flags. If the app
    is in silence-streaming mode, it also clears the VAD utterance gate.

    Args:
        ear: The Ear controller instance.
        from_hold: If True, indicates recording was started via mouse hold.
    """
    play = PlaySound("STARTING")
    play()
    open_mic_stream(ear, utr_gate, log_state)

    with ear.lock:
        ear.is_recording = True
        ear.last_rms = 0.0
        ear.total_frames = 0
        reset_chunk_tracking(log_state)
        log_state.level_log_time = 0.0

    session.clear_tail()
    if settings.is_silence_streaming_mode:
        utr_gate.reset
        begin_recording_session(session, telemetry_enabled)


def process_audio_callback(
    ear: EarProtocol,
    utr_gate: SileroUtteranceGate,
    log_state: LogState,
    in_data: bytes,
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
    with ear.lock:
        if not ear.is_recording:
            return (None, pyaudio.paContinue)

        boosted_chunk_bytes = boost_audio_chunk(in_data, ear.gain_multiplier)
        boosted_samples = np.frombuffer(boosted_chunk_bytes, dtype=np.int16)
        ear.last_rms = runtime_get_rms(boosted_chunk_bytes)
        ear.total_frames += 1
        ear.last_frequency_bands = analyze_frequency_bands(
            boosted_samples,
            sample_rate=settings.rate,
        )

        if settings.is_no_streaming_mode:
            cfg = SocketConfig()
            send_message(boosted_chunk_bytes, cfg)

            return (None, pyaudio.paContinue)

        now_seconds = time.time()
        speech_now = utr_gate.push(
            audio_chunk=in_data,
            now=now_seconds,
            analysis_chunk=boosted_chunk_bytes,
        )

        if speech_now and not log_state.chunk_speech_logged:
            log_state.chunk_speech_logged = True
            log_state.silence_pending_logged = False
            log.info("[Ear] 🗣️  VAD speech detected")

        if now_seconds - log_state.vad_state_log_time >= settings.vad_status_log_interval:
            try:
                silence_len = (
                    utr_gate.silence_len(now_seconds) if utr_gate.has_speech_started else 0.0
                )
                log.debug(
                    "[Ear] 🔎 "
                    f"VAD score={utr_gate.last_score:.3f} "
                    f"threshold={settings.vad_score_threshold:.2f} "
                    f"started={utr_gate.has_speech_started} silence={silence_len:.2f}s "
                    f"rms={ear.last_rms:.4f} "
                    f"energy={utr_gate.last_score:.4f} energy_threshold={utr_gate.last_dynamic_threshold:.4f}",
                )
            except (OSError, ValueError, TypeError):
                pass
            log_state.vad_state_log_time = now_seconds

    return (None, pyaudio.paContinue)
