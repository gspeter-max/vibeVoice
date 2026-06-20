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
from src.audio.ear_runtime.menu import TerminalMenu
from src.audio.ear_runtime.recording import (
    close_mic_stream,
    flush_current_chunk,
    process_audio_callback,
    start_recording_state,
    stop_no_streaming,
)
from src.audio.ear_runtime.system_audio import load_start_sound
from src.audio.vad_segmenter import SileroUtteranceGate, SileroVAD
from src.ipc.client import (
    close_raw_audio_stream_to_brain,
    open_checked_raw_audio_stream_to_brain,
)
from src.streaming.capture_session import CaptureSession
from src.streaming.session import should_split
from src.ui.hud_client import start_hud_command_thread, start_volume_sender_thread
from src.utils.settings import settings
from src.input.hotkeys import _is_rcmd

logger = get_logger()
try:
    from pynput import keyboard
except ImportError:
    logger.critical("pynput is not installed. Install it with: pip install pynput")
    sys.exit()

load_start_sound()


class Ear:
    """The main audio capture and processing engine for Parakeet Flow.

    The Ear is responsible for opening the microphone stream, performing
    real-time VAD (Voice Activity Detection), and splitting speech into
    logical chunks. It coordinates with the Brain via sockets to send
    audio data and with the HUD to provide visual feedback to the user.

    Attributes:
        pyaudio_library_for_capturing_audio: PyAudio library instance.
        stream: Audio capture stream from PyAudio.
        is_recording: True if recording is currently active.
        last_rms: The Root Mean Square energy of the last audio chunk.
        gain_multiplier: Multiplier to boost audio volume.
        input_device_index: Index of the resolved microphone device.
        active_mic_name: Name of the active mic.
        current_model: Name of the active transcription model.
    """

    pyaudio_library_for_capturing_audio: pyaudio.PyAudio
    stream: Optional[pyaudio.Stream]
    is_recording: bool
    last_rms: float
    gain_multiplier: float
    input_device_index: int
    active_mic_name: str
    current_model: str

    _lock: threading.Lock
    _total_frames: int
    last_frequency_bands: Dict[str, float]
    _cmd_press_time: float
    _toggle_active: bool
    _brain_sock: Optional[socket.socket]
    _brain_sock_lock: threading.Lock
    _telemetry_enabled: bool
    _chunk_speech_logged: bool
    _silence_pending_logged: bool
    _vad_state_log_time: float
    _recording_level_log_time: float
    _vad_no_speech_warned: bool
    _capture_session: CaptureSession
    _vad_engine: Optional[SileroVAD]
    _utterance_gate: SileroUtteranceGate

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
        self.pyaudio_library_for_capturing_audio = pyaudio_lib or pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self._lock = threading.Lock()
        self.last_rms = 0.0
        self.gain_multiplier = 1.2  # Increased from 1.1 to fix quiet mic issues
        self._total_frames = 0
        self.last_frequency_bands = {"bass": 0.33, "mid": 0.33, "treble": 0.34}
        self.input_device_index = resolve_input_device_index(
            self.pyaudio_library_for_capturing_audio,
            input_device_index,
        )

        self.active_mic_name = (
            self.pyaudio_library_for_capturing_audio.get_device_info_by_index(
                self.input_device_index
            ).get("name")
        )

        self._cmd_press_time = 0.0
        self._toggle_active = False
        self._brain_sock = None
        self._brain_sock_lock = threading.Lock()
        self._telemetry_enabled = (
            os.environ.get("STREAMING_TELEMETRY_ENABLED", "0").strip() == "1"
        )

        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_state_log_time = 0.0
        self._recording_level_log_time = 0.0
        self._vad_no_speech_warned = False
        # Session ID is generated ONCE when the app launches — it never changes.
        # _recording_index tracks how many times the user has pressed the record button.
        self._capture_session = CaptureSession(
            sample_rate=settings.rate,
            overlap_seconds=settings.overlap_seconds,
        )
        self.current_model = "parakeet-tdt-0.6b-v3"  # Default model

        # ★ VAD: buffer full utterances locally before sending to Brain
        try:
            self._vad_engine = SileroVAD(settings.vad_model_path)
            log.info("[Ear] Silero VAD loaded ✓")
        except Exception as e:
            self._vad_engine = None
            log.info("[Ear] Silero VAD load failed: %s — using buffer-only fallback", e)

        self._utterance_gate = SileroUtteranceGate(
            self._vad_engine,
            voice_threshold=settings.vad_score_threshold,
            silence_timeout_s=settings.silence_timeout_seconds,
            energy_threshold=settings.vad_energy_threshold,
            energy_ratio=settings.vad_energy_ratio,
        )
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

    def _audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Dict[str, float],
        status: int,
    ) -> Tuple[Optional[bytes], int]:
        """Delegate callback processing to `recording.py`.

        Args:
            in_data: Raw audio byte buffer from PyAudio.
            frame_count: Number of frames in the chunk.
            time_info: PyAudio timing metadata dictionary.
            status: PyAudio callback status flags.

        Returns:
            Tuple[Optional[bytes], int]: Output buffer and PyAudio control code.
        """
        return process_audio_callback(self, in_data, frame_count, time_info, status)

    def on_press(self, key: Any) -> None:
        """Keyboard event handler for when a key is pressed down.

        It specifically listens for the Right Command key to trigger the
        recording state. Depending on whether the app is in streaming or
        non-streaming mode, it will either start buffering chunks locally
        or open a persistent socket to the Brain. It also notifies the HUD
        to display the 'listening' state to provide visual confirmation.

        Args:
            key: Keyboard key object from pynput.
        """
        if not _is_rcmd(key):
            return

        log.debug("[Ear] Right CMD pressed")

        if self._toggle_active:
            self._toggle_active = False
            self._stop_and_send(stop_session=True)
            return

        with self._lock:
            if self.is_recording:
                log.info("[Ear] ⚠️ Already recording, ignoring press")
                return

        if settings.is_no_streaming_mode:
            log.debug("[Ear] About to open brain stream")
            raw_stream_socket = open_checked_raw_audio_stream_to_brain(
                timeout_seconds=5.0,
                socket_factory=socket.socket,
            )
            if raw_stream_socket is None:
                log.info("[Ear] ❌ Failed to open brain stream, aborting")
                return
            with self._brain_sock_lock:
                self._brain_sock = raw_stream_socket

        start_recording_state(self, from_hold=False)

        self._cmd_press_time = time.time()
        log.info("\r\n" + "─" * 50)
        log.info(f"\r🎙️  RECORDING ({self.active_mic_name})")

        start_hud_command_thread("listen", socket_factory=socket.socket)
        start_volume_sender_thread(
            self,
            volume_port=settings.vol_port,
            socket_factory=socket.socket,
        )

    def on_release(self, key: Any) -> None:
        """Keyboard event handler for when a key is released.

        If the Right Command key is released after being held for more
        than a short threshold, it stops the recording and signals the
        Brain to finalize. If the release happens very quickly, the app
        enters 'toggle mode', where recording continues until the key
        is pressed again, allowing for hands-free dictation when needed.

        Args:
            key: Keyboard key object from pynput.
        """
        if not _is_rcmd(key):
            return

        if self._toggle_active:
            return

        with self._lock:
            if not self.is_recording:
                return

        press_duration = time.time() - self._cmd_press_time
        if press_duration >= settings.recording_button_hold_threshold:
            log.info("\r[Ear] ⏹️  Right CMD released - finalizing now")
            self._stop_and_send(stop_session=True)
        else:
            self._toggle_active = True
            log.info("\r\n⏸️  Toggle mode — tap Right CMD again to stop")

    def _stop_and_send(self, *, stop_session: bool = True) -> None:
        """Unified method to stop recording and transmit the final data.

        It routes the stop command to either the streaming or non-streaming
        handler depending on the current configuration. This ensures that
        all recording logic follows the same cleanup path, whether it was
        triggered by a keyboard release, a mouse release, or a toggle click.

        Args:
            stop_session: If True, fully commits and stops the recording.
        """
        if settings.is_no_streaming_mode:
            stop_no_streaming(self)
            return
        flush_current_chunk(self, stop_session=stop_session)

    def record_loop(self, input_trigger: Optional[Any] = None) -> None:
        """Main background loop that manages the active recording state.

        Args:
            input_trigger: The InputTrigger instance from hotkeys.py. When provided,
                each tick calls input_trigger.check_mouse_hold_threshold()
                to handle right-mouse-button hold-to-record activation.
        """
        while True:
            self._record_loop_tick(input_trigger=input_trigger)

    def _record_loop_tick(ear: Ear, input_trigger: Optional[Any] = None) -> None:
        """Run one controller-managed recording loop tick.

        Args:
            ear: The Ear controller instance.
            input_trigger: Optional mouse-trigger hook to check hold status.
        """
        with ear._lock:
            recording = ear.is_recording
            rms = ear.last_rms

        if input_trigger is not None:
            input_trigger.check_mouse_hold()

        if not recording:
            return

        now_seconds = time.time()
        if (
            now_seconds - ear._recording_level_log_time
            >= settings.recording_level_log_interval
        ):
            meter_width = 30
            level = min(int(rms * 300), meter_width)
            meter = "█" * level + "░" * (meter_width - level)
            ear._recording_level_log_time = now_seconds
            print(f"\r  Voice Level: [{meter}] ", end="", flush=True)

        if not settings.is_silence_streaming_mode:
            return

        if "nemotron" in ear.current_model.lower():
            time_since_last_chunk = ear._capture_session.current_chunk_age_seconds(
                now_seconds
            )
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
        split_decision = should_split(
            start_time=ear._capture_session.chunk_started_at_seconds,
            now=now_seconds,
            min_age=(settings.minimum_chunk_age_before_silence_split_seconds),
            gate_finalize=ear._utterance_gate.should_finalize(now_seconds),
            silence_len=silence_elapsed_seconds,
        )
        if split_decision.should_split:
            ear._stop_and_send(stop_session=False)

    def cleanup(self) -> None:
        """Performs a clean shutdown of all Ear resources."""
        with self._brain_sock_lock:
            raw_stream_socket = self._brain_sock
            self._brain_sock = None
        close_raw_audio_stream_to_brain(raw_stream_socket)
        close_mic_stream(self)
        self.pyaudio_library_for_capturing_audio.terminate()
