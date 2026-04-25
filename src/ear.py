"""
ear.py — Parakeet Flow v2 (Streaming Edition)
===============================================
Captures microphone audio and splits speech into chunks on in-recording silence.
Each chunk is sent to brain.py immediately while recording continues.

Hold RIGHT CMD → speak → silence splits chunk → release → final chunk + commit.
"""

import os
import sys
import socket
import json
import threading
import time
import uuid
import pyaudio
import math
import select
import termios
import tty
import numpy as np

from src.env_utils import get_float_from_environment
from streaming_shared_logic import (
    DEFAULT_ENERGY_RATIO,
    DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_SILENCE_TIMEOUT_SECONDS,
    DEFAULT_VAD_ENERGY_THRESHOLD,
    DEFAULT_VAD_SCORE_THRESHOLD,
    apply_previous_chunk_overlap,
    should_split_chunk_after_silence,
)
from vad_segmenter import SileroVAD, SileroUtteranceGate
from src import log
try:
    from pynput import keyboard, mouse
except Exception:  # pragma: no cover - test environments may not support pynput backends
    class _FallbackKey:
        cmd_r = "cmd_r"
        esc = "esc"

    class _FallbackKeyboardListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
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
            return None

    class _FallbackMouseModule:
        Button = _FallbackMouseButton()
        Listener = _FallbackMouseListener

    keyboard = _FallbackKeyboardModule()
    mouse = _FallbackMouseModule()

# ── macOS Voice Isolation ──────────────────────────────────────────────────────
_VOICE_ISOLATION_ACTIVE = False
try:
    import AVFoundation 
    def _enable_macos_voice_isolation():
        global _VOICE_ISOLATION_ACTIVE
        try:
            engine = AVFoundation.AVAudioEngine.alloc().init()
            input_node = engine.inputNode()
            if hasattr(input_node, 'setVoiceProcessingEnabled_error_'):
                success, error = input_node.setVoiceProcessingEnabled_error_(True, None)
                if success:
                    _VOICE_ISOLATION_ACTIVE = True
                else:
                    log.info(f"[Ear] Voice processing not enabled: {error}")
            else:
                log.info("[Ear] inputNode does not support setVoiceProcessingEnabled_error_")
        except Exception as e:
            log.info(f"[Ear] Voice isolation init failed: {e}")
except ImportError:
    def _enable_macos_voice_isolation():
        log.info("[Ear] AVFoundation not available")

# ── Config ─────────────────────────────────────────────────────────────────────
SOCKET_PATH     = "/tmp/parakeet.sock"
BACKEND         = os.environ.get("BACKEND", "parakeet")
# This chooses the send style:
# - no_streaming: send raw audio and let Brain wait for socket close
# - silence_streaming: split speech on silence and send chunks
RECORDING_MODE  = os.environ.get("RECORDING_MODE", "silence_streaming").strip().lower()
NO_STREAMING_MODE = "no_streaming"
SILENCE_STREAMING_MODE = "silence_streaming"
VOL_PORT        = 57235
RECORDING_BUTTON_HOLD_THRESHOLD  = 0.4
VAD_MODEL_PATH  = os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx")
VAD_THRESHOLD   = get_float_from_environment("VAD_THRESHOLD", DEFAULT_VAD_SCORE_THRESHOLD)
VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT = get_float_from_environment(
    "VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT",
    DEFAULT_SILENCE_TIMEOUT_SECONDS,
)
VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION = get_float_from_environment(
    "VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION", 1.0
)
VAD_ENERGY_THRESHOLD = get_float_from_environment("VAD_ENERGY_THRESHOLD", DEFAULT_VAD_ENERGY_THRESHOLD)
VAD_ENERGY_RATIO = get_float_from_environment("VAD_ENERGY_RATIO", DEFAULT_ENERGY_RATIO)
VAD_STATUS_LOG_INTERVAL = 5.0
RECORDING_LEVEL_LOG_INTERVAL = 0.4
OVERLAP_SECONDS = get_float_from_environment("OVERLAP_SECONDS", DEFAULT_OVERLAP_SECONDS)
MIN_CHUNK_SECONDS_REQ_FOR_SPLITING_DUE_TO_SILENCE_STREAMING = get_float_from_environment(
    "MIN_CHUNK_SECONDS",
    DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
)

_RCMD_VK = 54

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000
CHUNK    = 1024

def get_active_models() -> list[str]:
    """
    Dynamically returns the list of available transcription models.
    Nemotron is specifically designed for streaming audio, so we only
    expose it if the user has explicitly selected the 'silence_streaming' mode.
    This prevents users from selecting an incompatible model in standard mode.
    """
    models = [
        "fast-conformer-ctc-en-24500", 
        "moonshine-base",
        "parakeet-tdt-0.6b-v2", 
        "parakeet-tdt-0.6b-v3",
    ]
    
    # Only append Nemotron if we are in streaming mode
    recording_mode = os.environ.get("RECORDING_MODE", "silence_streaming").strip().lower()
    if recording_mode == "silence_streaming":
        models.append("nemotron-streaming-0.6b")
        
    return models

def _is_right_cmd(key) -> bool:
    """
    Checks if a keyboard event corresponds to the Right Command key.
    Since different operating systems and keyboard libraries represent keys
    in various ways (using names, virtual key codes, or special objects),
    this function provides a unified check. It ensures that the 'Ear' correctly
    identifies the hotkey regardless of how the underlying system reports it.
    """
    if key == keyboard.Key.cmd_r:
        return True
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == _RCMD_VK:
        return True
    return False


def get_rms(block: bytes) -> float:
    """
    Calculates the Root Mean Square (RMS) volume level of an audio block.
    This provides a numerical value representing the 'loudness' of the audio.
    By converting raw bytes into normalized shorts and then averaging their
    squared values, we get a consistent metric that can be used for VAD
    (Voice Activity Detection) and for driving the visual volume meter.
    """
    import struct
    count = len(block) // 2
    shorts = struct.unpack(f"{count}h", block[:count * 2])
    if not shorts:
        return 0.0
    sum_sq = sum((s / 32768.0) ** 2 for s in shorts)
    return math.sqrt(sum_sq / len(shorts))


def send_switch_command(model_name, ear_instance=None):
    """
    Sends a request to the Brain to switch the active transcription model.
    If an Ear instance is provided, its current_model state is updated to
    allow for model-specific behavior changes (like heartbeat intervals).
    """
    log.info(f"\n🔄 Switching Brain to use: {model_name}...\n")
    if ear_instance:
        ear_instance.current_model = model_name
        
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        command = f"CMD_SWITCH_MODEL:{model_name}"
        client.sendall(command.encode('utf-8'))
        client.shutdown(socket.SHUT_WR)
        client.close()
    except Exception as e:
        log.info(f"\n❌ Failed to send switch command: {e}\n")


def run_self_test():
    """
    Executes a diagnostic test by sending synthetic audio to the Brain.
    It generates a 1-second sine wave at 440Hz and attempts to transmit it
    over the Unix socket. This helps verify that the communication path
    between the Ear and the Brain is working correctly and that the Brain
    is ready to accept and process audio data without needing a microphone.
    """
    log.info("\n🧪 Running SELF-TEST (synthetic audio)...\n")
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(RATE * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16).tobytes()

    # Retry logic for more robust connection
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(5)  # Shorter timeout for retries

            # Check if socket exists first
            if not os.path.exists(SOCKET_PATH):
                if attempt < max_retries - 1:
                    log.info(f"\r⏳ Socket not ready, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})\n")
                    time.sleep(retry_delay)
                    continue
                else:
                    log.info(f"\r❌ Self-test failed: Socket not found at {SOCKET_PATH}\n")
                    log.info("   Is Brain running? Check this terminal for Brain output.\n")
                    return

            client.connect(SOCKET_PATH)
            client.sendall(audio_data)
            client.shutdown(socket.SHUT_WR)
            client.close()

            log.info("\r✅ Self-test audio sent to Brain\n")
            return

        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                log.info(f"\r⏳ Brain busy, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})\n")
                time.sleep(retry_delay)
            else:
                log.info(f"\r❌ Self-test failed: Brain not accepting connections\n")
                log.info("   Brain might be loading model. Check this terminal for Brain output.\n")

        except Exception as e:
            log.info(f"\r❌ Self-test failed: {e}\n")
            break


class TerminalMenu(threading.Thread):
    """
    Background thread that listens for keyboard input in the terminal.
    Allows switching models and running self-tests without blocking recording.
    """
    def __init__(self, ear_instance=None):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()
        self.ear = ear_instance

    def run(self):
        if not sys.stdin.isatty():
            return
        old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1)
                    if c in '12345':
                        idx = int(c) - 1
                        active_models = get_active_models()
                        if idx < len(active_models):
                            send_switch_command(active_models[idx], self.ear)
                    elif c.lower() == 't':
                        threading.Thread(target=run_self_test, daemon=True).start()
                    elif c == '\x03':
                        os.kill(os.getpid(), 2)
                        break
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old_settings)

    def stop(self):
        self._stop.set()


def select_mic(p):
    """
    Interactive utility for selecting an input microphone from a list.
    It scans the system for all available audio input devices and presents
    them to the user in a numbered list. The user can then enter the
    index of their preferred microphone, or just press Enter to use
    the system default. This ensures the app records from the right source.
    """
    log.info("\n🎤  SELECT YOUR MICROPHONE:")
    log.info("─" * 30)
    devices = []
    default_device = p.get_default_input_device_info()
    default_device_index = default_device.get("index")
    default_choice_index = 0
    for device_index in range(p.get_device_count()):
        info = p.get_device_info_by_index(device_index)
        if info.get("maxInputChannels") > 0:
            name = info.get("name")
            choice_index = len(devices)
            is_default = " (DEFAULT)" if device_index == default_device_index else ""
            log.info(f" [{choice_index}] {name}{is_default}")
            devices.append(device_index)
            if device_index == default_device_index:
                default_choice_index = choice_index
    log.info("─" * 30)
    while True:
        try:
            choice = input(f"Select Mic Index [default {default_choice_index}]: ").strip()
            if not choice:
                return default_device_index
            choice_index = int(choice)
            if 0 <= choice_index < len(devices):
                return devices[choice_index]
            else:
                log.info("❌ Invalid index.")
        except ValueError:
            log.info("❌ Please enter a valid number.")


# ── Main ear class (STREAMING) ─────────────────────────────────────────────────
class Ear:
    """
    The main audio capture and processing engine for Parakeet Flow.
    The Ear is responsible for opening the microphone stream, performing
    real-time VAD (Voice Activity Detection), and splitting speech into
    logical chunks. It coordinates with the Brain via sockets to send
    audio data and with the HUD to provide visual feedback to the user.
    """
    def __init__(self, input_device_index=None):
        self.pyaudio_libaray_for_capturing_audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self._lock = threading.Lock()
        self.last_rms = 0.0
        self.gain_multiplier = 1.2 # Increased from 1.1 to fix quiet mic issues
        self.vad_sensitivity_boost = VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION
        self._total_frames = 0
        self.last_frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.34}
        self._last_raw_rms = 0.0
        self._last_vad_rms = 0.0

        # Enable macOS Voice Isolation if requested
        if os.environ.get("VOICE_ISOLATION", "0") == "1":
            _enable_macos_voice_isolation()
        else:
            log.info("[Ear] Voice Isolation disabled by default (set VOICE_ISOLATION=1 to enable)")

        if input_device_index is None:
            self.input_device_index = self.pyaudio_libaray_for_capturing_audio.get_default_input_device_info().get("index")
        else:
            self.input_device_index = input_device_index

        self.active_mic_name = self.pyaudio_libaray_for_capturing_audio.get_device_info_by_index(self.input_device_index).get("name")

        self.hud_proc = None
        self._cmd_press_time = 0.0
        self._toggle_active = False
        self._brain_sock = None
        self._brain_sock_lock = threading.Lock()
        self._telemetry_enabled = os.environ.get("STREAMING_TELEMETRY_ENABLED", "0").strip() == "1"

        # ★ MOUSE CONTROL: Hold-to-record for voice activation
        self._mouse_press_start_time = 0.0  # When mouse button was pressed
        self._is_holding = False              # Currently holding mouse button
        self._recording_from_hold = False     # Recording started from mouse hold

        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_state_log_time = 0.0
        self._recording_level_log_time = 0.0
        self._vad_no_speech_warned = False
        # Session ID is generated ONCE when the app launches — it never changes.
        # _recording_index tracks how many times the user has pressed the record button.
        self._current_session_id = uuid.uuid4().hex
        self._recording_index = 0
        self._chunk_seq = 0
        self._chunk_started_at = 0.0
        self.current_model = "parakeet-tdt-0.6b-v3" # Default model
        self._chunk_overlap_audio_bytes = int(RATE * 2 * OVERLAP_SECONDS)
        self._pending_chunk_overlap_audio = b""

        # ★ VAD: buffer full utterances locally before sending to Brain
        try:
            self._vad_engine = SileroVAD(VAD_MODEL_PATH)
            log.info("[Ear] Silero VAD loaded ✓")
        except Exception as e:
            self._vad_engine = None
            log.info(f"[Ear] ⚠️ Silero VAD load failed: {e} — using buffer-only fallback")

        self._utterance_gate = SileroUtteranceGate(
            self._vad_engine,
            voice_threshold=VAD_THRESHOLD,
            silence_timeout_s=VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT,
            energy_threshold=VAD_ENERGY_THRESHOLD,
            energy_ratio=VAD_ENERGY_RATIO,
        )
        log.info(
            f"[Ear] VAD config: threshold={VAD_THRESHOLD:.2f}, silence_timeout={VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT:.2f}s, "
            f"vad_gain={self.vad_sensitivity_boost:.1f}x, energy_threshold={VAD_ENERGY_THRESHOLD:.3f}, "
            f"energy_ratio={VAD_ENERGY_RATIO:.2f}"
        )

        # ★ ALWAYS LISTENING MODE: Open stream once at startup
        self.stream = None
        self._open_mic_stream()
        log.info(f"[Ear] Mic selected: {self.active_mic_name} ✓")

    def _send_hud(self, cmd):
        """
        Sends state commands to the HUD over a TCP socket.
        It updates the user interface by signaling whether the app is
        currently listening, processing, or done. This keeps the user
        informed about the AI's internal state without needing to look
        at the terminal window, which is especially useful for a background tool.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect(('127.0.0.1', 57234))
            s.sendall(cmd.encode())
            s.close()
        except Exception as e:
            log.info(f"[Ear] ❌ HUD command '{cmd}' failed: {e}")

    def _start_volume_sender(self):
        """
        Launches a background thread to stream volume data to the HUD.
        This function uses UDP to send real-time RMS levels and frequency
        band data (bass, mid, treble) approximately 25 times per second.
        The HUD uses this data to drive its visual volume meter and
        animations, providing a smooth and responsive user experience.
        """
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def _sender():
            packets_sent = 0
            while True:
                with self._lock:
                    if not self.is_recording:
                        log.info(f"[Ear] Volume sender stopped (sent {packets_sent} packets)")
                        break
                    rms = self.last_rms
                    freq_bands = self.last_frequency_bands
                try:
                    # Send volume + frequency bands in format: "vol:RMS,bass:BASS,mid:MID,treble:TREBLE"
                    message = f"vol:{rms:.4f},bass:{freq_bands['bass']:.3f},mid:{freq_bands['mid']:.3f},treble:{freq_bands['treble']:.3f}"
                    udp.sendto(message.encode(), ('127.0.0.1', VOL_PORT))
                    packets_sent += 1
                except Exception as e:
                    log.info(f"[Ear] ❌ Failed to send volume: {e}")
                time.sleep(0.04)
            udp.close()
        threading.Thread(target=_sender, daemon=True).start()
        log.debug("[Ear] Volume sender thread started")

    def _is_no_streaming_mode(self) -> bool:
        """Return True when Ear should send raw audio to one open Brain socket."""
        return RECORDING_MODE == NO_STREAMING_MODE

    def _is_silence_streaming_mode(self) -> bool:
        """Return True when Ear should split speech on silence."""
        return RECORDING_MODE == SILENCE_STREAMING_MODE

    def _begin_recording_session(self):
        """
        Initializes a new streaming session with a unique ID.
        This function resets the chunk counters and records the start time,
        then sends a 'session_started' event to the Brain. This ensures
        that both the Ear and the Brain are synchronized and ready to
        process a series of related audio chunks as a single conversation.
        """
        # session_id was already created in __init__; just reset the per-recording counters.
        self._chunk_seq = 0
        self._chunk_started_at = time.time()
        self._send_session_event_to_brain(
            "session_started",
            {"recording_mode": RECORDING_MODE},
        )

    def _send_session_event_to_brain(self, event_type: str, fields: dict | None = None) -> bool:
        """
        Transmits a telemetry event from the Ear to the Brain's server.
        It packs the event type and optional data fields into a JSON
        message and sends it over the Unix domain socket. This allows
        the Brain to keep track of microphone levels and VAD state
        changes that happen on the Ear's side of the application.
        """
        if not self._telemetry_enabled or not self._current_session_id:
            return False

        payload = {"type": event_type}
        if fields:
            payload.update(fields)

        # Header format: CMD_SESSION_EVENT:SESSION_ID:RECORDING_INDEX
        header = f"CMD_SESSION_EVENT:{self._current_session_id}:{self._recording_index}\n\n".encode("utf-8")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(SOCKET_PATH)
                client.sendall(header + body)
                client.shutdown(socket.SHUT_WR)
            return True
        except Exception as e:
            log.info(f"[Ear] ❌ Failed to send telemetry event '{event_type}': {e}")
            return False

    def _send_audio_chunk_to_brain(self, utterance_bytes: bytes) -> bool:
        """
        Sends a single processed audio chunk to the Brain for transcription.
        The chunk is prefixed with a session header and a sequence number
        so the Brain can correctly stitch it back together with other chunks.
        It also records a 'chunk_sent_to_brain' event to the telemetry log
        to help debug any network latency or data loss between processes.
        """
        if not utterance_bytes or not self._current_session_id:
            return False

        session_id = self._current_session_id
        seq = self._chunk_seq
        self._chunk_seq += 1
        # Header format: CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ
        header = f"CMD_AUDIO_CHUNK:{session_id}:{self._recording_index}:{seq}\n\n".encode("utf-8")

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(SOCKET_PATH)
                client.sendall(header + utterance_bytes)
                client.shutdown(socket.SHUT_WR)
            log.info("📤 Audio chunk sent", seq=seq, size=len(utterance_bytes), session=session_id[:8])
            self._send_session_event_to_brain(
                "chunk_sent_to_brain",
                {
                    "chunk_index": seq,
                    "audio_bytes": len(utterance_bytes),
                },
            )
            return True
        except Exception as e:
            log.error("❌ Failed to send chunk", error=str(e))
            return False

    def _commit_recording_session(self) -> bool:
        """
        Signals to the Brain that the current recording session is finished.
        This command is sent when the user releases the recording hotkey.
        It tells the Brain that no more audio chunks will be arriving for
        this session ID, allowing the Brain to finalize the transcription
        and paste the text as soon as all pending chunks are processed.
        """
        if not self._current_session_id:
            return False

        session_id = self._current_session_id
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(SOCKET_PATH)
                # Header format: CMD_SESSION_COMMIT:SESSION_ID:RECORDING_INDEX
                client.sendall(f"CMD_SESSION_COMMIT:{session_id}:{self._recording_index}".encode("utf-8"))
                client.shutdown(socket.SHUT_WR)
            log.info("✅ Session committed", session=session_id[:8], recording=self._recording_index)
            self._recording_index += 1  # button released → next press gets the next recording slot
            return True
        except Exception as e:
            log.error("❌ Failed to commit session", error=str(e))
            return False

    def _boost_pcm16_bytes(self, pcm16_bytes: bytes) -> bytes:
        """
        Applies a digital gain multiplier to raw 16-bit PCM audio data.
        This is used to artificially increase the volume of the microphone
        input before it is sent to the AI for transcription. By boosting
        the signal, we can improve the accuracy of the transcription for
        users with quiet microphones while ensuring the audio doesn't clip.
        """
        if not pcm16_bytes:
            return pcm16_bytes
        if len(pcm16_bytes) % 2:
            pcm16_bytes = pcm16_bytes[:-1]
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        boosted = (audio * self.gain_multiplier).clip(-32768, 32767).astype(np.int16)
        return boosted.tobytes()

    def _prepare_vad_chunk(self, pcm16_bytes: bytes) -> bytes:
        """
        Conditions raw audio data specifically for the VAD engine.
        It applies a separate sensitivity boost to the audio before
        passing it to the Voice Activity Detector. This allows us to
        fine-tune how easily the system 'wakes up' to speech without
        affecting the actual audio quality that is sent to the Brain
        for transcription, providing a more responsive user experience.
        """
        if not pcm16_bytes:
            return pcm16_bytes
        if len(pcm16_bytes) % 2:
            pcm16_bytes = pcm16_bytes[:-1]

        self._last_raw_rms = get_rms(pcm16_bytes)
        if self.vad_sensitivity_boost == 1.0:
            self._last_vad_rms = self._last_raw_rms
            return pcm16_bytes

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        conditioned = (audio * self.vad_sensitivity_boost).clip(-32768, 32767).astype(np.int16)
        conditioned_bytes = conditioned.tobytes()
        self._last_vad_rms = get_rms(conditioned_bytes)
        return conditioned_bytes

    def _reset_chunk_tracking(self):
        """
        Resets internal flags and counters for a new audio chunk.
        When a chunk is finalized or split, we need to clear the state
        that tracks whether we've already logged speech or warned about
        silence for that specific segment. This ensures that log messages
        and telemetry events are correctly associated with each new
        part of the user's ongoing speech.
        """
        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_no_speech_warned = False

    def _prepend_pending_chunk_overlap(
        self, 
        audio_chunk_for_brain: bytes, 
        *, 
        stop_session: bool,
        silence_seconds: float = 0.0
    ) -> bytes:
        """
        Adds a small overlap from the previous chunk to the current one.
        This technique ensures that speech isn't 'cut' exactly at a word
        boundary, which can confuse the AI transcription model. By including
        a tiny bit of context from the end of the last chunk, we provide
        the Brain with enough surrounding information to maintain a smooth
        and accurate flow of text across all chunks in a session.
        """
        silence_audio_byte_count = int(silence_seconds * RATE * 2)
        overlap_application_result = apply_previous_chunk_overlap(
            current_chunk_audio_bytes=audio_chunk_for_brain,
            previous_pending_overlap_audio_bytes=self._pending_chunk_overlap_audio,
            overlap_audio_byte_count=self._chunk_overlap_audio_bytes,
            silence_audio_byte_count=silence_audio_byte_count,
            sample_rate=RATE,
            stop_session=stop_session,
        )
        self._pending_chunk_overlap_audio = overlap_application_result.next_pending_overlap_audio_bytes
        return overlap_application_result.overlapped_audio_bytes

    def _flush_current_chunk(self, *, stop_session: bool) -> bool:
        """
        Finalizes the current utterance and sends it to the Brain.
        This function is called either when silence is detected or when
        the user stops recording. It collects all buffered audio from the
        VAD, applies gain and overlap, and then transmits the final result.
        It also handles the session commit command if this was the last chunk
        of the recording, effectively closing the loop on that session.
        """
        now = time.time()
        silence_elapsed = self._utterance_gate.silence_elapsed(now)
        had_session = bool(self._current_session_id)

        with self._lock:
            if not self.is_recording:
                return False
            total = self._total_frames
            if stop_session:
                self.is_recording = False
                self._recording_from_hold = False
                self._chunk_started_at = 0.0
            self._total_frames = 0
            self.last_rms = 0.0
            self._reset_chunk_tracking()

        utterance = self._utterance_gate.flush()
        if not utterance:
            if stop_session:
                self._pending_chunk_overlap_audio = b""
                log.info("[Ear] 🔇 No speech captured; stopping recording")
                # Always commit so the Brain closes out this recording slot
                self._commit_recording_session()
                # Reset per-recording seq counter; session ID stays alive
                self._chunk_seq = 0
            return False

        utterance_for_brain = self._boost_pcm16_bytes(utterance)
        previous_pending_overlap_audio = self._pending_chunk_overlap_audio
        utterance_for_brain = self._prepend_pending_chunk_overlap(
            utterance_for_brain,
            stop_session=stop_session,
            silence_seconds=silence_elapsed if not stop_session else 0.0
        )
        overlap_seconds_added = len(previous_pending_overlap_audio) / 2.0 / RATE
        chunk_age_seconds = max(0.0, now - self._chunk_started_at)

        duration = (total * CHUNK) / RATE
        if stop_session:
            log.info(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n")
            threading.Thread(target=self._send_hud, args=("process",), daemon=True).start()
        else:
            log.info(
                f"\r[Ear] ✂️  Silence boundary hit ({silence_elapsed:.2f}s) — sending chunk "
                f"{duration:.1f}s ({total} chunks)"
            )

        sent = self._send_audio_chunk_to_brain(utterance_for_brain)
        if sent:
            self._send_session_event_to_brain(
                "silence_threshold_hit" if not stop_session else "session_stopped",
                {
                    "chunk_index": self._chunk_seq - 1,
                    "chunk_age_seconds": round(chunk_age_seconds, 2),
                    "silence_elapsed_seconds": round(silence_elapsed, 2),
                    "split_reason": "silence_threshold_hit" if not stop_session else "session_stop",
                    "overlap_seconds_added": round(overlap_seconds_added, 4),
                    "audio_bytes": len(utterance_for_brain),
                },
            )
        if stop_session:
            self._commit_recording_session()
            # Reset per-recording seq counter; session ID stays alive for the app lifetime
            self._chunk_seq = 0
        else:
            self._chunk_started_at = time.time()
        return sent

    def _stop_no_streaming(self):
        """
        Stops a non-streaming recording session and closes the Brain socket.
        In legacy non-streaming mode, we don't split audio into chunks.
        Instead, we wait until the user finishes and then signal the Brain
        to process the entire buffer at once. This function manages that
        transition by updating the recording state, notifying the HUD,
        and cleanly shutting down the background socket connection.
        """
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._recording_from_hold = False
            total = self._total_frames
            self._total_frames = 0
            self.last_rms = 0.0

        duration = (total * CHUNK) / RATE
        log.info(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n")
        threading.Thread(target=self._send_hud, args=("process",), daemon=True).start()
        threading.Thread(target=self._close_brain_stream, daemon=True).start()

    def _open_brain_stream(self) -> bool:
        """
        Opens a dedicated socket for non-streaming audio data.
        If a connection to the Brain's Unix socket isn't already open,
        this function attempts to establish one. It includes error handling
        to inform the user if the Brain process isn't running or isn't
        responding, ensuring the application fails gracefully instead
        of crashing during a recording attempt.
        """
        with self._brain_sock_lock:
            if self._brain_sock is not None:
                return True
            if not os.path.exists(SOCKET_PATH):
                log.info(f"\r❌ Brain socket not found\n")
                return False
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(SOCKET_PATH)
                self._brain_sock = sock
                return True
            except Exception as e:
                log.info(f"\r❌ Brain connect failed: {e}\n")
                return False

    def _stream_chunk_to_brain(self, chunk_bytes: bytes):
        """Send raw audio bytes over the open no_streaming socket."""
        with self._brain_sock_lock:
            if self._brain_sock is None:
                return
            try:
                self._brain_sock.sendall(chunk_bytes)
            except (BrokenPipeError, ConnectionResetError):
                log.info(f"\r⚠️  Brain disconnected — will transcribe on release\n")
                try:
                    self._brain_sock.close()
                except Exception:
                    pass
                self._brain_sock = None
            except Exception as e:
                log.info(f"\r❌ Stream send error: {e}\n")
                try:
                    self._brain_sock.close()
                except Exception:
                    pass
                self._brain_sock = None

    def _close_brain_stream(self):
        """Close the raw-audio socket so Brain can process the full buffer."""
        with self._brain_sock_lock:
            if self._brain_sock is None:
                return
            try:
                self._brain_sock.shutdown(socket.SHUT_WR)
                self._brain_sock.close()
            except Exception:
                pass
            self._brain_sock = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle one microphone callback and route it to the active mode."""
        if status:
            pass # Suppress status printing in always-listening mode to avoid spam

        with self._lock:
            if not self.is_recording:
                return (None, pyaudio.paContinue)

            audio_data = np.frombuffer(in_data, dtype=np.int16)
            boosted = (audio_data.astype(np.float32) * self.gain_multiplier).clip(-32768, 32767).astype(np.int16)
            chunk_bytes = boosted.tobytes()
            self.last_rms = get_rms(chunk_bytes)
            self._total_frames += 1

            # ★ FREQUENCY ANALYSIS: Extract bass/mid/treble bands using FFT
            self.last_frequency_bands = self._analyze_frequency_bands(boosted)

            if self._is_no_streaming_mode():
                # Raw mode: send bytes straight to Brain and wait for socket close.
                self._stream_chunk_to_brain(chunk_bytes)
            else:
                # Silence mode: keep the utterance locally until silence splits it.
                now = time.time()
                vad_bytes = self._prepare_vad_chunk(in_data)
                
                speech_now = self._utterance_gate.push(
                    pcm16_bytes=in_data,
                    now=now ,
                    analysis_pcm16_bytes= vad_bytes 
                )

                if speech_now and not self._chunk_speech_logged:
                    self._chunk_speech_logged = True
                    self._silence_pending_logged = False
                    log.info("[Ear] 🗣️  VAD speech detected")
                
                if now - self._vad_state_log_time >= VAD_STATUS_LOG_INTERVAL:
                    try:
                        score = self._utterance_gate.last_score()
                        energy = self._utterance_gate.last_energy()
                        dynamic_threshold = self._utterance_gate.last_dynamic_threshold()
                        started = self._utterance_gate.has_speech_started()
                        silence_elapsed = self._utterance_gate.silence_elapsed(now) if started else 0.0
                        log.debug(
                            f"[Ear] 🔎 VAD score={score:.3f} threshold={VAD_THRESHOLD:.2f} "
                            f"started={started} silence={silence_elapsed:.2f}s "
                            f"raw_rms={self._last_raw_rms:.4f} vad_rms={self._last_vad_rms:.4f} "
                            f"energy={energy:.4f} energy_threshold={dynamic_threshold:.4f}",
                        )
                    except Exception:
                        pass
                    self._vad_state_log_time = now

        return (None, pyaudio.paContinue)

    def _analyze_frequency_bands(self, audio_samples: np.ndarray) -> dict:
        """
        Analyze audio samples to extract bass, mid, and treble frequency bands.

        Args:
            audio_samples: NumPy array of audio samples (int16)

        Returns:
            Dict with 'bass', 'mid', 'treble' values (0.0 to 1.0)
        """
        try:
            # Convert to float for FFT
            samples_float = audio_samples.astype(np.float32) / 32768.0

            # Apply windowing function to reduce spectral leakage
            window = np.hanning(len(samples_float))
            windowed = samples_float * window

            # Compute FFT
            fft_result = np.fft.fft(windowed)
            fft_magnitude = np.abs(fft_result[:len(fft_result)//2])

            # Frequency bins (sample rate is 16000 Hz)
            # Nyquist frequency = 8000 Hz
            freq_bins = np.fft.fftfreq(len(windowed), 1.0/RATE)[:len(fft_magnitude)]

            # Define frequency bands for speech
            # Bass: 20-250 Hz (low frequency hum, vowels)
            bass_mask = (freq_bins >= 20) & (freq_bins < 250)
            bass_energy = np.sum(fft_magnitude[bass_mask])

            # Mid: 250-4000 Hz (speech intelligibility range)
            mid_mask = (freq_bins >= 250) & (freq_bins < 4000)
            mid_energy = np.sum(fft_magnitude[mid_mask])

            # Treble: 4000-8000 Hz (consonants, high frequency sounds)
            treble_mask = (freq_bins >= 4000) & (freq_bins < 8000)
            treble_energy = np.sum(fft_magnitude[treble_mask])

            # Normalize to 0.0-1.0 range
            total_energy = bass_energy + mid_energy + treble_energy

            if total_energy > 0:
                bass_norm = bass_energy / total_energy
                mid_norm = mid_energy / total_energy
                treble_norm = treble_energy / total_energy
            else:
                # Equal distribution when no signal
                bass_norm = 0.33
                mid_norm = 0.33
                treble_norm = 0.34

            return {
                'bass': bass_norm,
                'mid': mid_norm,
                'treble': treble_norm
            }

        except Exception as e:
            # Fallback to equal distribution if FFT fails
            log.info(f"[Ear] ⚠️ Frequency analysis failed: {e}")
            return {'bass': 0.33, 'mid': 0.33, 'treble': 0.34}

    def _open_mic_stream(self):
        """Open a FRESH mic stream each recording session."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass

        self.stream = self.pyaudio_libaray_for_capturing_audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        log.info("[Ear] 🎤 Mic stream opened")

    def _close_mic_stream(self):
        """
        Shuts down the active PyAudio microphone stream.
        This function stops the audio input and releases the hardware
        resources. It is called when the application is closing to ensure
        that the microphone isn't left in an 'active' state by the OS,
        which could prevent other applications from using it or cause
        unnecessary battery drain on mobile devices like MacBooks.
        """
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _start_recording_state(self, *, from_hold: bool) -> None:
        """
        Resets the internal state and prepares the Ear for a new recording.
        It clears all volume and frame counters, sets the recording flag,
        and determines if this recording was triggered by a sustained hold
        or a single keypress. In streaming mode, it also resets the VAD
        engine and starts a fresh telemetry session to track the new recording.
        """
        with self._lock:
            self.is_recording = True
            self.last_rms = 0.0
            self._total_frames = 0
            self._recording_from_hold = from_hold
            self._chunk_speech_logged = False
            self._silence_pending_logged = False
            self._vad_no_speech_warned = False
            self._recording_level_log_time = 0.0

        self._pending_chunk_overlap_audio = b""
        if self._is_silence_streaming_mode():
            self._utterance_gate.reset()
            self._begin_recording_session()

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

        if self._is_no_streaming_mode():
            log.debug("[Ear] About to open brain stream")
            if not self._open_brain_stream():
                log.info("[Ear] ❌ Failed to open brain stream, aborting")
                return

        self._start_recording_state(from_hold=False)

        self._cmd_press_time = time.time()
        log.info("\r\n" + "─" * 50)
        log.info(f"\r🎙️  RECORDING ({self.active_mic_name})")

        threading.Thread(target=self._send_hud, args=("listen",), daemon=True).start()
        self._start_volume_sender()

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

        from_first_cmd_press_to_release_time_diff = time.time() - self._cmd_press_time
        if from_first_cmd_press_to_release_time_diff >= RECORDING_BUTTON_HOLD_THRESHOLD:
            log.info("\r[Ear] ⏹️  Right CMD released - finalizing now")
            self._stop_and_send(stop_session=True)
        else:
            self._toggle_active = True
            log.info("\r\n⏸️  Toggle mode — tap Right CMD again to stop")

    def on_mouse_click(self, x, y, button, pressed):
        """
        Handle mouse button press/release for hold-to-record recording control.
        This allows the user to trigger recording using the Right Mouse Button.
        Like the keyboard hotkey, it supports hold-to-record behavior, where
        releasing the mouse button after a sustained hold will stop the
        recording. This provides an alternative input method for users who
        prefer using their mouse over keyboard shortcuts for voice activation.
        """
        # Only right button triggers hold logic (avoids text selection from left button)
        if button != mouse.Button.right:
            return

        if pressed:
            # MOUSE BUTTON PRESSED: Start hold timer
            self._mouse_press_start_time = time.time()
            self._is_holding = True
            # No visual feedback during hold delay (silent wait)

        else:
            # MOUSE BUTTON RELEASED: Stop recording or cancel hold
            self._is_holding = False

            # Only stop if we started recording from this specific hold
            if self._recording_from_hold:
                # Check if actually recording before stopping
                with self._lock:
                    if not self.is_recording:
                        self._recording_from_hold = False
                        return
                log.info("\r[Ear] ⏹️  Mouse released - finalizing now")
                self._stop_and_send(stop_session=True)
                self._recording_from_hold = False

    def _stop_and_send(self, *, stop_session: bool = True):
        """
        Unified method to stop recording and transmit the final data.
        It routes the stop command to either the streaming or non-streaming
        handler depending on the current configuration. This ensures that
        all recording logic follows the same cleanup path, whether it was
        triggered by a keyboard release, a mouse release, or a toggle click.
        """
        if self._is_no_streaming_mode():
            self._stop_no_streaming()
            return
        self._flush_current_chunk(stop_session=stop_session)

    def record_loop(self):
        """
        Main background loop that manages the active recording state.
        This loop runs continuously while the Ear is active, checking for
        mouse hold durations and updating the visual volume meter in the
        terminal. It also monitors the VAD engine for silence timeouts to
        automatically split speech into chunks during long dictation sessions.
        """
        while True:
            time.sleep(0.05)
            self._record_loop_tick()

    def _record_loop_tick(self):
        """
        Internal tick function called by the record loop to update state.
        It checks if a mouse hold has reached the 1-second threshold to start
        recording and handles the periodic logging of microphone levels.
        In streaming mode, it also queries the VAD engine to see if a
        silence threshold has been hit, triggering an automatic chunk split
        to keep the transcription feedback fast and responsive.
        """
        with self._lock:
            recording = self.is_recording
            rms = self.last_rms

        # CHECK: Hold duration >= 1.0s → Start recording
        # Read hold state with lock protection
        with self._lock:
            is_holding = self._is_holding
            press_start_time = self._mouse_press_start_time

        if is_holding and not recording:
            hold_duration = time.time() - press_start_time

            if hold_duration >= 1.0:
                # Hold duration exceeded threshold - start recording
                if self._is_no_streaming_mode():
                    if not self._open_brain_stream():
                        log.info("\r[Ear] ❌ Failed to open brain stream")
                        with self._lock:
                            self._is_holding = False
                        return

                self._start_recording_state(from_hold=True)

                log.info("\r\n" + "─" * 50)
                log.info(f"\r🎙️  RECORDING via MOUSE HOLD ({self.active_mic_name})")

                threading.Thread(target=self._send_hud, args=("listen",), daemon=True).start()
                self._start_volume_sender()

        # Display volume meter when recording
        if recording:
            now = time.time()
            if now - self._recording_level_log_time >= RECORDING_LEVEL_LOG_INTERVAL:
                meter = "█" * min(int(rms * 500), 50)
                log.info(f"\r  Level: [{meter:<50}]")
                self._recording_level_log_time = now

            if self._is_silence_streaming_mode():
                # --- Nemotron Fixed Time Gap Logic ---
                # Nemotron works best when we send sound every 1.12 seconds.
                # If we are using Nemotron, we ignore silence and just use this time gap.
                if "nemotron" in self.current_model.lower():
                    time_since_last_chunk = now - self._chunk_started_at
                    if time_since_last_chunk >= 1.12:
                        log.info(f"\r[Ear] 💓 Nemotron time gap reached (1.12s); sending sound")
                        self._stop_and_send(stop_session=False)
                    return
                # --- End Nemotron Logic ---

                # Silence mode watches for speech ending so it can send a chunk.
                if self._utterance_gate.has_speech_started() and not self._silence_pending_logged:
                    silence_elapsed = self._utterance_gate.silence_elapsed(now)
                    if silence_elapsed > 0.0:
                        self._silence_pending_logged = True
                        log.info(
                            f"[Ear] 🤫 Silence pending ({silence_elapsed:.2f}s / {VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT:.2f}s)",
                        )

                if (
                    not self._utterance_gate.has_speech_started()
                    and not self._vad_no_speech_warned
                    and self._chunk_started_at > 0.0
                    and (now - self._chunk_started_at) >= 1.0
                ):
                    try:
                        max_score = self._utterance_gate.max_score()
                    except Exception:
                        max_score = 0.0
                    log.info(
                        f"[Ear] ⚠️  VAD has not entered speech state yet "
                        f"(max_score={max_score:.3f}, threshold={VAD_THRESHOLD:.2f}, "
                        f"last_energy={self._utterance_gate.last_energy():.4f}, "
                        f"energy_threshold={self._utterance_gate.last_dynamic_threshold():.4f})",
                    )
                    self._send_session_event_to_brain(
                        "vad_no_speech_warning",
                        {
                            "chunk_index": self._chunk_seq,
                            "max_score": round(max_score, 3),
                            "threshold": round(VAD_THRESHOLD, 2),
                            "last_energy": round(self._utterance_gate.last_energy(), 4),
                            "energy_threshold": round(self._utterance_gate.last_dynamic_threshold(), 4),
                        },
                    )
                    self._vad_no_speech_warned = True
                
                silence_elapsed = (
                    self._utterance_gate.silence_elapsed(now)
                    if self._utterance_gate.has_speech_started()
                    else self._utterance_gate.finalize_elapsed(now)
                )
                split_decision = should_split_chunk_after_silence(
                    chunk_started_at_seconds=self._chunk_started_at,
                    now_seconds=now,
                    minimum_chunk_age_before_silence_split_seconds=MIN_CHUNK_SECONDS_REQ_FOR_SPLITING_DUE_TO_SILENCE_STREAMING,
                    utterance_gate_should_finalize_now=self._utterance_gate.should_finalize(now),
                    silence_duration_seconds=silence_elapsed,
                )
                if split_decision.should_split_now:
                    log.info(
                        f"\r[Ear] ✂️  Silence threshold hit ({silence_elapsed:.2f}s >= {VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT:.2f}s); sending chunk",
                    )
                    self._stop_and_send(stop_session=False)

    def cleanup(self):
        """
        Performs a clean shutdown of all Ear resources.
        It closes the microphone stream, terminates the PyAudio instance,
        and ensures that any open network sockets to the Brain are
        disconnected. This function is vital for preventing 'zombie'
        processes or locked audio hardware when the user exits the
        application using Ctrl+C or by closing the terminal window.
        """
        self._close_brain_stream()
        self._close_mic_stream()
        self.pyaudio_libaray_for_capturing_audio.terminate()


def start_ear():
    """
    The main entry point for the Ear process.
    It handles initial microphone selection, starts the background
    terminal menu, and initializes the Ear engine. It also sets up
    keyboard and mouse listeners to capture user input and enters
    the main recording loop. This function coordinates all the
    moving parts required to turn your voice into digital text.
    """
    p_temp = pyaudio.PyAudio()
    selected_mic_index = select_mic(p_temp)
    p_temp.terminate()

    ear = Ear(input_device_index=selected_mic_index)

    menu = TerminalMenu(ear_instance=ear)
    menu.start()

    # Keyboard listener for Right CMD shortcut
    listener = keyboard.Listener(on_press=ear.on_press, on_release=ear.on_release)
    listener.start()

    # Mouse listener for hold-to-record
    mouse_listener = mouse.Listener(on_click=ear.on_mouse_click)
    mouse_listener.start()
    log.info("[Ear] 🖱️  Mouse listener started - Hold RIGHT button for 1s to record")

    backend_label = {
        "parakeet": "sherpa-onnx + parakeet-tdt-v3 (INT8)",
    }.get(BACKEND, BACKEND)

    mic_mode = "Voice Isolation (macOS)" if os.environ.get("VOICE_ISOLATION", "0") == "1" else "Standard (Raw Audio)"

    
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║      🎙️  PARAKEET FLOW v2 — STREAMING MODE       ║")
    log.info(f"║  Backend : {backend_label:<38}║")
    log.info(f"║  Mic Mode: {mic_mode:<38}║")
    log.info(f"║  Hotkey  : RIGHT CMD (hold to record)            ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info(" Press [1] Conformer    [2] Moonshine")
    log.info(" Press [3] Parakeet v2  [4] Parakeet v3")
    
    active_models = get_active_models()
    if "nemotron-streaming-0.6b" in active_models:
        log.info(" Press [5] Nemotron     [t] Self-test")
    else:
        log.info(" Press [t] Self-test")
    log.info("─" * 52)
    log.info(" Brain output prints directly in this terminal.")
    log.info("─" * 52)

    try:
        ear.record_loop()
    except KeyboardInterrupt:
        log.info("\r\n\nShutting down Ear...")
    finally:
        menu.stop()
        ear.cleanup()
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, termios.tcgetattr(sys.stdin.fileno()))


if __name__ == "__main__":
    start_ear()
