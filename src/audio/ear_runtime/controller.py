"""Ear Runtime Controller
======================

The main audio capture and processing engine for Parakeet Flow.
The Ear is responsible for opening the microphone stream, performing
real-time VAD (Voice Activity Detection), and splitting speech into
logical chunks. It coordinates with the Brain via sockets to send
audio data and with the HUD to provide visual feedback to the user.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import pyaudio
from structlog import get_logger

from src import log
from src.audio.ear_runtime.devices import resolve_input_device_index
from src.audio.ear_runtime.recording import (
    LogState,
    begin_recording_session,
    flush_current_chunk,
    open_mic_stream,
    process_audio_callback,
    reset_chunk_tracking,
    send_session_event_to_telemetry_brain,
    start_recording_state,
)
from src.audio.vad_segmenter import SileroUtteranceGate, SileroVAD
from src.input.hotkeys import is_rcmd
from src.streaming.capture_session import CaptureSession
from src.streaming.session import should_split
from src.ui.hud_client import change_ui_status, ui_wave_input
from src.utils.settings import settings

logger = get_logger()
try:
    from pynput import keyboard
except ImportError:
    logger.critical("pynput is not installed. Install it with: pip install pynput")
    sys.exit()




class Ear:
    """The main audio capture and processing engine for Parakeet Flow.

    The Ear is responsible for opening the microphone stream, performing
    real-time VAD (Voice Activity Detection), and splitting speech into
    logical chunks. It coordinates with the Brain via sockets to send
    audio data and with the HUD to provide visual feedback to the user.

    Attributes:
        pyaudio_inst: PyAudio library instance.
        stream: Audio capture stream from PyAudio.
        is_recording: True if recording is currently active.
        last_rms: The Root Mean Square energy of the last audio chunk.
        gain_multiplier: Multiplier to boost audio volume.
        input_device_index: Index of the resolved microphone device.
        active_mic_name: Name of the active mic.
        current_model: Name of the active transcription model.
    """

    pyaudio_inst: pyaudio.PyAudio
    stream: Optional[pyaudio.Stream]
    is_recording: bool
    last_rms: float
    gain_multiplier: float
    input_device_index: int
    active_mic_name: str
    current_model: str

    lock: threading.Lock
    total_frames: int
    last_frequency_bands: Dict[str, float]
    cmd_press_time: float
    toggle_active: bool
    telemetry_enabled: bool

    def __init__(
        self,
        input_device_index: Optional[int] = None,
        pyaudio_lib: Optional[pyaudio.PyAudio] = None,
    ) -> None:
        """Initialize the Ear audio controller and load processing engines.

        Args:
            input_device_index: Explicit index of the microphone device to use.
                If None, the system default input device will be resolved.
            pyaudio_lib: Pre-initialized PyAudio instance. If None, a new
                PyAudio instance will be created.
        """
        self.pyaudio_inst = pyaudio_lib or pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.lock = threading.Lock()
        self.last_rms = 0.0
        self.gain_multiplier = 1.2  # Increased from 1.1 to fix quiet mic issues
        self.total_frames = 0
        self.last_frequency_bands = {"bass": 0.33, "mid": 0.33, "treble": 0.34}
        self.input_device_index = resolve_input_device_index(
            self.pyaudio_inst,
            input_device_index,
        )

        self.active_mic_name = self.pyaudio_inst.get_device_info_by_index(
            self.input_device_index
        ).get("name")

        self.cmd_press_time = 0.0
        self.toggle_active = False
        self.telemetry_enabled = os.environ.get("STREAMING_TELEMETRY_ENABLED", "0").strip() == "1"
        self.current_model = "parakeet-tdt-0.6b-v3"  # Default model

        log.info(
            "[Ear] VAD config: "
            f"threshold={settings.vad_score_threshold:.2f}, "
            f"silence_timeout={settings.silence_timeout_seconds:.2f}s, "
            f"energy_threshold={settings.vad_energy_threshold:.3f}, "
            f"energy_ratio={settings.vad_energy_ratio:.2f}"
        )

        # ★ ALWAYS LISTENING MODE: Open stream once at startup
        self.stream = None
        log.info(f"[Ear] Mic selected: {self.active_mic_name} ✓")

    def stop_no_streaming(self) -> None:
        """Finalize a non-streaming recording and close its raw audio socket.

        This method switches off the active recording flag, calculates the total
        session duration, and closes the raw socket connection to the Brain.
        It runs the socket cleanup in a background thread to prevent GUI lockups.

        Args:
            ear: The Ear controller instance.
        """
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            total_frames = self.total_frames
            self.total_frames = 0
            self.last_rms = 0.0

        duration_seconds = (total_frames * settings.chunk) / settings.rate
        log.info(
            f"\r\n⏹️  Streamed {duration_seconds:.1f}s ({total_frames} chunks) — Brain transcribing...\n"
        )
        change_ui_status("process")
        self.close_mic_stream

    @property
    def close_mic_stream(self) -> None:
        """Close and release the active PyAudio microphone input stream.

        If a stream is currently active, it stops audio capture, closes the stream
        interface, and sets the stream attribute back to None. It catches and
        silences any OS-level errors during the teardown process.

        Args:
            ear: The Ear controller instance.
        """
        if self.stream is None:
            return

        try:
            self.stream.stop_stream()
            self.stream.close()
        except OSError:
            pass
        self.stream = None

    def stop_and_send(
        self,
        session: CaptureSession,
        utr_gate: SileroUtteranceGate,
        log_state: LogState,
        stop_session: bool = True,
    ) -> None:
        """Unified method to stop recording and transmit the final data."""
        if settings.is_no_streaming_mode:
            self.stop_no_streaming()
        flush_current_chunk(
            self, session, utr_gate, log_state, stop_session, self.telemetry_enabled
        )

    def record_loop(
        self,
        input_trigger: Optional[Any],
        utr_gate: SileroUtteranceGate,
        session: CaptureSession,
        log_state: LogState,
    ) -> None:
        """Run the main background loop that keeps the Ear process alive."""
        while True:
            self.record_loop_tick(input_trigger, utr_gate, session, log_state)

    def record_loop_tick(
        self,
        input_trigger: Optional[Any],
        utr_gate: SileroUtteranceGate,
        session: CaptureSession,
        log_state: LogState,
    ) -> None:
        """Execute a single tick of the background recording controller loop."""
        with self.lock:
            recording = self.is_recording
            rms = self.last_rms

        if input_trigger is not None:
            input_trigger.check_mouse_hold()

        if not recording:
            return

        now_seconds = time.time()
        if now_seconds - log_state.level_log_time >= settings.recording_level_log_interval:
            meter_width = 30
            level = min(int(rms * 300), meter_width)
            meter = "█" * level + "░" * (meter_width - level)
            log_state.level_log_time = now_seconds
            print(f"\r  Voice Level: [{meter}] ", end="", flush=True)

        if not settings.is_silence_streaming_mode:
            return

        if "nemotron" in self.current_model.lower():
            if session.chunk_age >= 1.12:
                self.stop_and_send(session, utr_gate, log_state, stop_session=False)
            return

        if utr_gate.has_speech_started and not log_state.silence_pending_logged:
            if utr_gate.silence_len(now_seconds) > 0.0:
                log_state.silence_pending_logged = True

        silence_len = (
            utr_gate.silence_len(now_seconds)
            if utr_gate.has_speech_started
            else utr_gate.finalize_elapsed(now_seconds)
        )
        split_decision = should_split(
            start_time=session.chunk_start,
            now=now_seconds,
            min_age=(settings.minimum_chunk_age_before_silence_split_seconds),
            gate_finalize=utr_gate.should_finalize(now_seconds),
            silence_len=silence_len,
        )
        if split_decision.should_split:
            self.stop_and_send(session, utr_gate, log_state, stop_session=False)

    def cleanup(self) -> None:
        """Perform a clean shutdown of the active Ear controller resources.

        This method closes the active microphone stream, shuts down the raw audio
        socket connection to the Brain, and terminates the underlying PyAudio
        library instance to release system audio handles.
        """
        self.close_mic_stream
        self.pyaudio_inst.terminate()
