"""
Ear Runtime Controller
======================

The main audio capture and processing engine for Parakeet Flow.
The Ear is responsible for opening the microphone stream, performing
real-time VAD (Voice Activity Detection), and splitting speech into
logical chunks. It coordinates with the Brain via sockets to send
audio data and with the HUD to provide visual feedback to the user.
"""

import os
import socket
import threading
import time
import pyaudio

from src import log
from src.audio.vad_segmenter import SileroVAD, SileroUtteranceGate
from src.audio.ear_runtime.menu import (
    TerminalMenu as RuntimeTerminalMenu,
)
from src.audio.ear_runtime.devices import resolve_input_device_index
from src.audio.ear_runtime.system_audio import (
    load_start_sound,
)
from src.audio.ear_runtime.recording import (
    close_mic_stream,
    flush_current_chunk,
    open_mic_stream,
    process_audio_callback,
    record_loop_tick,
    start_recording_state,
    stop_no_streaming,
)
from src.ipc.client import close_raw_audio_stream_and_forget, open_checked_raw_audio_stream_to_brain
from src.streaming.capture_session import CaptureSession
from src.ui.hud_client import start_hud_command_thread, start_volume_sender_thread
from src.utils.settings import settings

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover
    class _FallbackKey:
        cmd_r = "cmd_r"
        esc = "esc"

    class _FallbackKeyboardListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            """Match pynput's listener API without doing any work."""
            return None

    class _FallbackKeyboardModule:
        Key = _FallbackKey()
        Listener = _FallbackKeyboardListener

    class _FallbackMouseButton:
        left = "left"
        right = "right"

    class _FallbackMouseListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            """Match pynput's listener API without doing any work."""
            return None

    class _FallbackMouseModule:
        Button = _FallbackMouseButton()
        Listener = _FallbackMouseListener

    keyboard = _FallbackKeyboardModule()
    mouse = _FallbackMouseModule()

# Initialize sounds
load_start_sound()


def _is_right_cmd(key) -> bool:
    """Return whether a keyboard event matches the Right Command hotkey."""
    return (
        key == keyboard.Key.cmd_r
        or getattr(key, "name", None) == "cmd_r"
        or getattr(key, "vk", None) == settings.right_cmd_vk
    )

TerminalMenu = RuntimeTerminalMenu


# ── Main ear class (STREAMING) ─────────────────────────────────────────────────
class Ear:
    """
    The main audio capture and processing engine for Parakeet Flow.
    The Ear is responsible for opening the microphone stream, performing
    real-time VAD (Voice Activity Detection), and splitting speech into
    logical chunks. It coordinates with the Brain via sockets to send
    audio data and with the HUD to provide visual feedback to the user.
    """
    def __init__(self, input_device_index=None, pyaudio_lib=None):
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
            self.pyaudio_library_for_capturing_audio
            .get_device_info_by_index(self.input_device_index)
            .get("name")
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
        open_mic_stream(self)
        log.info(f"[Ear] Mic selected: {self.active_mic_name} ✓")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Delegate callback processing to `recording.py`."""
        return process_audio_callback(self, in_data, frame_count, time_info, status)

    def on_press(self, key):
        """
        Keyboard event handler for when a key is pressed down.
        It specifically listens for the Right Command key to trigger the
        recording state. Depending on whether the app is in streaming or
        non-streaming mode, it will either start buffering chunks locally
        or open a persistent socket to the Brain. It also notifies the HUD
        to display the 'listening' state to provide visual confirmation.
        """
        if not _is_right_cmd(key):
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
                socket_path=settings.socket_path,
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

    def on_release(self, key):
        """
        Keyboard event handler for when a key is released.
        If the Right Command key is released after being held for more
        than a short threshold, it stops the recording and signals the
        Brain to finalize. If the release happens very quickly, the app
        enters 'toggle mode', where recording continues until the key
        is pressed again, allowing for hands-free dictation when needed.
        """
        if not _is_right_cmd(key):
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

    def _stop_and_send(self, *, stop_session: bool = True):
        """
        Unified method to stop recording and transmit the final data.
        It routes the stop command to either the streaming or non-streaming
        handler depending on the current configuration. This ensures that
        all recording logic follows the same cleanup path, whether it was
        triggered by a keyboard release, a mouse release, or a toggle click.
        """
        if settings.is_no_streaming_mode:
            stop_no_streaming(self)
            return
        flush_current_chunk(self, stop_session=stop_session)

    def record_loop(self, input_trigger=None):
        """
        Main background loop that manages the active recording state.

        Args:
            input_trigger: The InputTrigger instance from hotkeys.py. When provided,
                           each tick calls input_trigger.check_mouse_hold_threshold()
                           to handle right-mouse-button hold-to-record activation.
        """
        while True:
            time.sleep(0.05)
            self._record_loop_tick(input_trigger=input_trigger)

    def _record_loop_tick(self, input_trigger=None):
        """Delegate tick mechanics to `recording.py`."""
        record_loop_tick(self, input_trigger=input_trigger)

    def cleanup(self):
        """
        Performs a clean shutdown of all Ear resources.
        """
        with self._brain_sock_lock:
            raw_stream_socket = self._brain_sock
            self._brain_sock = None
        close_raw_audio_stream_and_forget(raw_stream_socket)
        close_mic_stream(self)
        self.pyaudio_library_for_capturing_audio.terminate()
