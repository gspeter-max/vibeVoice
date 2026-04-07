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
import threading
import time
import uuid
import pyaudio
import math
import select
import termios
import tty
import numpy as np

from streaming_shared_logic import apply_previous_chunk_overlap, should_split_chunk_after_silence
from vad_segmenter import SileroVAD, SileroUtteranceGate

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
                    print(f"[Ear] Voice processing not enabled: {error}", flush=True)
            else:
                print("[Ear] inputNode does not support setVoiceProcessingEnabled_error_", flush=True)
        except Exception as e:
            print(f"[Ear] Voice isolation init failed: {e}", flush=True)
except ImportError:
    def _enable_macos_voice_isolation():
        print("[Ear] AVFoundation not available", flush=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SOCKET_PATH     = "/tmp/parakeet.sock"
BACKEND         = os.environ.get("BACKEND", "faster_whisper")
RECORDING_MODE  = os.environ.get("RECORDING_MODE", "silence_streaming").strip().lower()
NO_STREAMING_MODE = "no_streaming"
SILENCE_STREAMING_MODE = "silence_streaming"
VOL_PORT        = 57235
RECORDING_BUTTON_HOLD_THRESHOLD  = 0.4
VAD_MODEL_PATH  = os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx")
VAD_THRESHOLD   = float(os.environ.get("VAD_THRESHOLD", "0.50"))
VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT = float(os.environ.get("VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT", "0.65"))
VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION = float(os.environ.get("VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION", "6.0"))
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.05"))
VAD_ENERGY_RATIO = float(os.environ.get("VAD_ENERGY_RATIO", "2.5"))
VAD_STATUS_LOG_INTERVAL = 0.5
OVERLAP_SECONDS = float(os.environ.get("OVERLAP_SECONDS", "0.50"))
MIN_CHUNK_SECONDS_REQ_FOR_SPLITING_DUE_TO_SILENCE_STREAMING = float(os.environ.get("MIN_CHUNK_SECONDS", "12.0"))

_RCMD_VK = 54

def _is_right_cmd(key) -> bool:
    if key == keyboard.Key.cmd_r:
        return True
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == _RCMD_VK:
        return True
    return False

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000
CHUNK    = 1024

MODELS = [
    "tiny.en", "base.en", "small.en",
    "Systran/faster-distil-whisper-large-v3",
    "medium.en", "large-v2", "large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3"
]

def get_rms(block: bytes) -> float:
    import struct
    count = len(block) // 2
    shorts = struct.unpack(f"{count}h", block[:count * 2])
    if not shorts:
        return 0.0
    sum_sq = sum((s / 32768.0) ** 2 for s in shorts)
    return math.sqrt(sum_sq / len(shorts))

def send_switch_command(model_name):
    print(f"\n🔄 Switching Brain to use: {model_name}...\n", end="", flush=True)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        command = f"CMD_SWITCH_MODEL:{model_name}"
        client.sendall(command.encode('utf-8'))
        client.shutdown(socket.SHUT_WR)
        client.close()
    except Exception as e:
        print(f"\n❌ Failed to send switch command: {e}\n", end="", flush=True)

def run_self_test():
    print("\n🧪 Running SELF-TEST (synthetic audio)...\n", end="", flush=True)
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
                    print(f"\r⏳ Socket not ready, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})\n", end="", flush=True)
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"\r❌ Self-test failed: Socket not found at {SOCKET_PATH}\n", end="", flush=True)
                    print("   Is Brain running? Check this terminal for Brain output.\n", end="", flush=True)
                    return

            client.connect(SOCKET_PATH)
            client.sendall(audio_data)
            client.shutdown(socket.SHUT_WR)
            client.close()

            print("\r✅ Self-test audio sent to Brain\n", end="", flush=True)
            return

        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"\r⏳ Brain busy, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})\n", end="", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"\r❌ Self-test failed: Brain not accepting connections\n", end="", flush=True)
                print("   Brain might be loading model. Check this terminal for Brain output.\n", end="", flush=True)

        except Exception as e:
            print(f"\r❌ Self-test failed: {e}\n", end="", flush=True)
            break


class TerminalMenu(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()

    def run(self):
        if not sys.stdin.isatty():
            return
        old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1)
                    if c in '1234567890':
                        idx = int(c) - 1 if c != '0' else 9
                        send_switch_command(MODELS[idx])
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
    print("\n🎤  SELECT YOUR MICROPHONE:")
    print("─" * 30)
    devices = []
    default_device = p.get_default_input_device_info()
    default_index = default_device.get("index")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels") > 0:
            name = info.get("name")
            is_default = " (DEFAULT)" if i == default_index else ""
            print(f" [{i}] {name}{is_default}")
            devices.append(i)
    print("─" * 30)
    while True:
        try:
            choice = input(f"Select Mic Index [default {default_index}]: ").strip()
            if not choice:
                return default_index
            idx = int(choice)
            if idx in devices:
                return idx
            else:
                print("❌ Invalid index.")
        except ValueError:
            print("❌ Please enter a valid number.")


# ── Main ear class (STREAMING) ─────────────────────────────────────────────────
class Ear:
    def __init__(self, input_device_index=None):
        self.pyaudio_libaray_for_capturing_audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self._lock = threading.Lock()
        self.last_rms = 0.0
        self.gain_multiplier = 2.5 # Increased from 1.1 to fix quiet mic issues
        self.vad_sensitivity_boost = VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION
        self._total_frames = 0
        self.last_frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.34}
        self._last_raw_rms = 0.0
        self._last_vad_rms = 0.0

        # Enable macOS Voice Isolation if requested
        if os.environ.get("VOICE_ISOLATION", "0") == "1":
            _enable_macos_voice_isolation()
        else:
            print("[Ear] Voice Isolation disabled by default (set VOICE_ISOLATION=1 to enable)", flush=True)

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

        # ★ MOUSE CONTROL: Hold-to-record for voice activation
        self._mouse_press_start_time = 0.0  # When mouse button was pressed
        self._is_holding = False              # Currently holding mouse button
        self._recording_from_hold = False     # Recording started from mouse hold

        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_state_log_time = 0.0
        self._vad_no_speech_warned = False
        self._current_session_id = None
        self._chunk_seq = 0
        self._chunk_started_at = 0.0
        self._chunk_overlap_audio_bytes = int(RATE * 2 * OVERLAP_SECONDS)
        self._pending_chunk_overlap_audio = b""

        # ★ VAD: buffer full utterances locally before sending to Brain
        try:
            self._vad_engine = SileroVAD(VAD_MODEL_PATH)
            print("[Ear] Silero VAD loaded ✓", flush=True)
        except Exception as e:
            self._vad_engine = None
            print(f"[Ear] ⚠️ Silero VAD load failed: {e} — using buffer-only fallback", flush=True)

        self._utterance_gate = SileroUtteranceGate(
            self._vad_engine,
            voice_threshold=VAD_THRESHOLD,
            silence_timeout_s=VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT,
            energy_threshold=VAD_ENERGY_THRESHOLD,
            energy_ratio=VAD_ENERGY_RATIO,
        )
        print(
            f"[Ear] VAD config: threshold={VAD_THRESHOLD:.2f}, silence_timeout={VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT:.2f}s, "
            f"vad_gain={self.vad_sensitivity_boost:.1f}x, energy_threshold={VAD_ENERGY_THRESHOLD:.3f}, "
            f"energy_ratio={VAD_ENERGY_RATIO:.2f}",
            flush=True,
        )

        # ★ ALWAYS LISTENING MODE: Open stream once at startup
        self.stream = None
        self._open_mic_stream()
        print(f"[Ear] Mic selected: {self.active_mic_name} ✓", flush=True)

    def _send_hud(self, cmd):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect(('127.0.0.1', 57234))
            s.sendall(cmd.encode())
            s.close()
        except Exception as e:
            print(f"[Ear] ❌ HUD command '{cmd}' failed: {e}", flush=True)

    def _start_volume_sender(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def _sender():
            packets_sent = 0
            while True:
                with self._lock:
                    if not self.is_recording:
                        print(f"[Ear] Volume sender stopped (sent {packets_sent} packets)", flush=True)
                        break
                    rms = self.last_rms
                    freq_bands = self.last_frequency_bands
                try:
                    # Send volume + frequency bands in format: "vol:RMS,bass:BASS,mid:MID,treble:TREBLE"
                    message = f"vol:{rms:.4f},bass:{freq_bands['bass']:.3f},mid:{freq_bands['mid']:.3f},treble:{freq_bands['treble']:.3f}"
                    udp.sendto(message.encode(), ('127.0.0.1', VOL_PORT))
                    packets_sent += 1
                except Exception as e:
                    print(f"[Ear] ❌ Failed to send volume: {e}", flush=True)
                time.sleep(0.04)
            udp.close()
        threading.Thread(target=_sender, daemon=True).start()
        print(f"[Ear] Volume sender thread started", flush=True)

    def _is_no_streaming_mode(self) -> bool:
        return RECORDING_MODE == NO_STREAMING_MODE

    def _is_silence_streaming_mode(self) -> bool:
        return RECORDING_MODE == SILENCE_STREAMING_MODE

    def _begin_recording_session(self):
        self._current_session_id = uuid.uuid4().hex
        self._chunk_seq = 0
        self._chunk_started_at = time.time()

    def _send_audio_chunk_to_brain(self, utterance_bytes: bytes) -> bool:
        if not utterance_bytes or not self._current_session_id:
            return False

        session_id = self._current_session_id
        seq = self._chunk_seq
        self._chunk_seq += 1
        header = f"CMD_AUDIO_CHUNK:{session_id}:{seq}\n\n".encode("utf-8")

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(SOCKET_PATH)
                client.sendall(header + utterance_bytes)
                client.shutdown(socket.SHUT_WR)
            print(f"[Ear] 📤 Chunk {seq} sent to Brain ({len(utterance_bytes)} bytes)", flush=True)
            return True
        except Exception as e:
            print(f"\r❌ Failed to send chunk to Brain: {e}\n", flush=True)
            return False

    def _commit_recording_session(self) -> bool:
        if not self._current_session_id:
            return False

        session_id = self._current_session_id
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(SOCKET_PATH)
                client.sendall(f"CMD_SESSION_COMMIT:{session_id}".encode("utf-8"))
                client.shutdown(socket.SHUT_WR)
            print(f"[Ear] ✅ Commit sent for session {session_id}", flush=True)
            return True
        except Exception as e:
            print(f"\r❌ Failed to commit session: {e}\n", flush=True)
            return False

    def _boost_pcm16_bytes(self, pcm16_bytes: bytes) -> bytes:
        if not pcm16_bytes:
            return pcm16_bytes
        if len(pcm16_bytes) % 2:
            pcm16_bytes = pcm16_bytes[:-1]
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        boosted = (audio * self.gain_multiplier).clip(-32768, 32767).astype(np.int16)
        return boosted.tobytes()

    def _prepare_vad_chunk(self, pcm16_bytes: bytes) -> bytes:
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
        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_no_speech_warned = False

    def _prepend_pending_chunk_overlap(self, audio_chunk_for_brain: bytes, *, stop_session: bool) -> bytes:
        overlap_application_result = apply_previous_chunk_overlap(
            current_chunk_audio_bytes=audio_chunk_for_brain,
            previous_pending_overlap_audio_bytes=self._pending_chunk_overlap_audio,
            overlap_audio_byte_count=self._chunk_overlap_audio_bytes,
            sample_rate=RATE,
            stop_session=stop_session,
        )
        self._pending_chunk_overlap_audio = overlap_application_result.next_pending_overlap_audio_bytes
        return overlap_application_result.overlapped_audio_bytes

    def _flush_current_chunk(self, *, stop_session: bool) -> bool:
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
                print("[Ear] 🔇 No speech captured; stopping recording", flush=True)
                if had_session:
                    self._commit_recording_session()
                self._current_session_id = None
                self._chunk_seq = 0
            return False

        utterance_for_brain = self._boost_pcm16_bytes(utterance)
        utterance_for_brain = self._prepend_pending_chunk_overlap(
            utterance_for_brain,
            stop_session=stop_session,
        )

        duration = (total * CHUNK) / RATE
        if stop_session:
            print(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n", end="", flush=True)
            threading.Thread(target=self._send_hud, args=("process",), daemon=True).start()
        else:
            print(
                f"\r[Ear] ✂️  Silence boundary hit ({silence_elapsed:.2f}s) — sending chunk "
                f"{duration:.1f}s ({total} chunks)",
                flush=True,
            )

        sent = self._send_audio_chunk_to_brain(utterance_for_brain)
        if stop_session:
            self._commit_recording_session()
            self._current_session_id = None
            self._chunk_seq = 0
        else:
            self._chunk_started_at = time.time()
        return sent

    def _stop_no_streaming(self):
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._recording_from_hold = False
            total = self._total_frames
            self._total_frames = 0
            self.last_rms = 0.0

        duration = (total * CHUNK) / RATE
        print(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n", end="", flush=True)
        threading.Thread(target=self._send_hud, args=("process",), daemon=True).start()
        threading.Thread(target=self._close_brain_stream, daemon=True).start()

    def _open_brain_stream(self) -> bool:
        """Open a persistent socket to brain for no-streaming mode buffering."""
        with self._brain_sock_lock:
            if self._brain_sock is not None:
                return True
            if not os.path.exists(SOCKET_PATH):
                print(f"\r❌ Brain socket not found\n", flush=True)
                return False
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(SOCKET_PATH)
                self._brain_sock = sock
                return True
            except Exception as e:
                print(f"\r❌ Brain connect failed: {e}\n", flush=True)
                return False

    def _stream_chunk_to_brain(self, chunk_bytes: bytes):
        with self._brain_sock_lock:
            if self._brain_sock is None:
                return
            try:
                self._brain_sock.sendall(chunk_bytes)
            except (BrokenPipeError, ConnectionResetError):
                print(f"\r⚠️  Brain disconnected — will transcribe on release\n", flush=True)
                try:
                    self._brain_sock.close()
                except Exception:
                    pass
                self._brain_sock = None
            except Exception as e:
                print(f"\r❌ Stream send error: {e}\n", flush=True)
                try:
                    self._brain_sock.close()
                except Exception:
                    pass
                self._brain_sock = None

    def _close_brain_stream(self):
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
                self._stream_chunk_to_brain(chunk_bytes)
            else:
                # ★ VAD BUFFERING: keep the full utterance locally until silence closes it
                now = time.time()
                vad_bytes = self._prepare_vad_chunk(in_data)
                speech_now = self._utterance_gate.push(vad_bytes, now=now)
                if speech_now and not self._chunk_speech_logged:
                    self._chunk_speech_logged = True
                    self._silence_pending_logged = False
                    print("[Ear] 🗣️  VAD speech detected", flush=True)
                if now - self._vad_state_log_time >= VAD_STATUS_LOG_INTERVAL:
                    try:
                        score = self._utterance_gate.last_score()
                        energy = self._utterance_gate.last_energy()
                        dynamic_threshold = self._utterance_gate.last_dynamic_threshold()
                        started = self._utterance_gate.has_speech_started()
                        silence_elapsed = self._utterance_gate.silence_elapsed(now) if started else 0.0
                        print(
                            f"[Ear] 🔎 VAD score={score:.3f} threshold={VAD_THRESHOLD:.2f} "
                            f"started={started} silence={silence_elapsed:.2f}s "
                            f"raw_rms={self._last_raw_rms:.4f} vad_rms={self._last_vad_rms:.4f} "
                            f"energy={energy:.4f} energy_threshold={dynamic_threshold:.4f}",
                            flush=True,
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
            print(f"[Ear] ⚠️ Frequency analysis failed: {e}", flush=True)
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
        print(f"[Ear] 🎤 Mic stream opened", flush=True)

    def _close_mic_stream(self):
        """Close mic stream after recording."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _start_recording_state(self, *, from_hold: bool) -> None:
        with self._lock:
            self.is_recording = True
            self.last_rms = 0.0
            self._total_frames = 0
            self._recording_from_hold = from_hold
            self._chunk_speech_logged = False
            self._silence_pending_logged = False
            self._vad_no_speech_warned = False

        self._pending_chunk_overlap_audio = b""
        if self._is_silence_streaming_mode():
            self._utterance_gate.reset()
            self._begin_recording_session()

    def on_press(self, key):
        if not _is_right_cmd(key):
            return

        print(f"[Ear] 🔵 Right CMD pressed - on_press() called", flush=True)

        if self._toggle_active:
            self._toggle_active = False
            self._stop_and_send(stop_session=True)
            return

        with self._lock:
            if self.is_recording:
                print(f"[Ear] ⚠️ Already recording, ignoring press", flush=True)
                return

        if self._is_no_streaming_mode():
            print(f"[Ear] 🔵 About to open brain stream", flush=True)
            if not self._open_brain_stream():
                print(f"[Ear] ❌ Failed to open brain stream, aborting", flush=True)
                return

        print(f"[Ear] 🔵 About to start recording", flush=True)
        self._start_recording_state(from_hold=False)

        self._cmd_press_time = time.time()
        print("\r\n" + "─" * 50, flush=True)
        print(f"\r🎙️  RECORDING ({self.active_mic_name})", flush=True)

        print(f"[Ear] 🔵 About to send 'listen' command to HUD", flush=True)
        threading.Thread(target=self._send_hud, args=("listen",), daemon=True).start()
        print(f"[Ear] 🔵 About to start volume sender", flush=True)
        self._start_volume_sender()

    def on_release(self, key):
        if not _is_right_cmd(key):
            return

        if self._toggle_active: 
            return

        with self._lock:
            if not self.is_recording:
                return 

        from_first_cmd_press_to_release_time_diff = time.time() - self._cmd_press_time
        if from_first_cmd_press_to_release_time_diff >= RECORDING_BUTTON_HOLD_THRESHOLD:
            # if user is pressin the cmd button for long time then from first cmd press to release time it large 
            # so if that is small that means the user is doing togglning 
            print(f"\r[Ear] ⏹️  Right CMD released - finalizing now", flush=True)
            self._stop_and_send(stop_session=True)
        else:
            self._toggle_active = True
            print(f"\r\n⏸️  Toggle mode — tap Right CMD again to stop", flush=True)

    def on_mouse_click(self, x, y, button, pressed):
        """
        Handle mouse button press/release for hold-to-record recording control.

        Press behavior (pressed=True):
            - Record press timestamp
            - Set holding flag
            - No visual feedback during hold delay

        Release behavior (pressed=False):
            - Clear holding flag
            - If recording started from this hold: stop recording
            - Early release (< 1s): silent reset (no action)

        Args:
            x, y: Mouse coordinates (unused)
            button: Which mouse button (only Button.right matters)
            pressed: True for press, False for release
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
                print("\r[Ear] ⏹️  Mouse released - finalizing now", flush=True)
                self._stop_and_send(stop_session=True)
                self._recording_from_hold = False

    def _stop_and_send(self, *, stop_session: bool = True):
        if self._is_no_streaming_mode():
            self._stop_no_streaming()
            return
        self._flush_current_chunk(stop_session=stop_session)

    def record_loop(self):
        """Main recording loop.

        Checks for hold duration >= 1.0s to start recording.
        Displays volume meter when recording.
        """
        while True:
            time.sleep(0.05)
            self._record_loop_tick()

    def _record_loop_tick(self):
        """Single iteration of record loop logic.

        Checks hold duration and starts recording if threshold met.
        Displays volume meter when recording.
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
                        print(f"\r[Ear] ❌ Failed to open brain stream", flush=True)
                        with self._lock:
                            self._is_holding = False
                        return

                self._start_recording_state(from_hold=True)

                print("\r\n" + "─" * 50, flush=True)
                print(f"\r🎙️  RECORDING via MOUSE HOLD ({self.active_mic_name})", flush=True)

                threading.Thread(target=self._send_hud, args=("listen",), daemon=True).start()
                self._start_volume_sender()

        # Display volume meter when recording
        if recording:
            meter = "█" * min(int(rms * 500), 50)
            print(f"\r  Level: [{meter:<50}]", end="", flush=True)

            if self._is_silence_streaming_mode():
                now = time.time()

                if self._utterance_gate.has_speech_started() and not self._silence_pending_logged:
                    silence_elapsed = self._utterance_gate.silence_elapsed(now)
                    if silence_elapsed > 0.0:
                        self._silence_pending_logged = True
                        print(
                            f"[Ear] 🤫 Silence pending ({silence_elapsed:.2f}s / {VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT:.2f}s)",
                            flush=True,
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
                    print(
                        f"[Ear] ⚠️  VAD has not entered speech state yet "
                        f"(max_score={max_score:.3f}, threshold={VAD_THRESHOLD:.2f}, "
                        f"last_energy={self._utterance_gate.last_energy():.4f}, "
                        f"energy_threshold={self._utterance_gate.last_dynamic_threshold():.4f})",
                        flush=True,
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
                    print(
                        f"\r[Ear] ✂️  Silence threshold hit ({silence_elapsed:.2f}s >= {VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT:.2f}s); sending chunk",
                        flush=True,
                    )
                    self._stop_and_send(stop_session=False)

    def cleanup(self):
        self._close_brain_stream()
        self._close_mic_stream()
        self.pyaudio_libaray_for_capturing_audio.terminate()


def start_ear():
    p_temp = pyaudio.PyAudio()
    selected_mic_index = select_mic(p_temp)
    p_temp.terminate()

    menu = TerminalMenu()
    menu.start()

    ear = Ear(input_device_index=selected_mic_index)

    # Keyboard listener for Right CMD shortcut
    listener = keyboard.Listener(on_press=ear.on_press, on_release=ear.on_release)
    listener.start()

    # Mouse listener for hold-to-record
    mouse_listener = mouse.Listener(on_click=ear.on_mouse_click)
    mouse_listener.start()
    print("[Ear] 🖱️  Mouse listener started - Hold RIGHT button for 1s to record", flush=True)

    backend_label = {
        "faster_whisper": "faster-whisper + distil-large-v3 (INT8)",
        "openvino":       "whisper.cpp + OpenVINO (Intel iGPU)",
    }.get(BACKEND, BACKEND)

    mic_mode = "Voice Isolation (macOS)" if os.environ.get("VOICE_ISOLATION", "0") == "1" else "Standard (Raw Audio)"

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      🎙️  PARAKEET FLOW v2 — STREAMING MODE       ║")
    print(f"║  Backend : {backend_label:<38}║")
    print(f"║  Mic Mode: {mic_mode:<38}║")
    print(f"║  Hotkey  : RIGHT CMD (hold to record)            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(" Press [1] tiny.en      [2] base.en     [3] small.en")
    print(" Press [4] distil-large [5] medium.en   [6] large-v2")
    print(" Press [7] large-v3     [8] turbo       [9] Parakeet v2")
    print(" Press [0] Parakeet v3  [t] Self-test")
    print("─" * 52)
    print(" Brain output prints directly in this terminal.\n")
    print("─" * 52)

    try:
        ear.record_loop()
    except KeyboardInterrupt:
        print("\r\n\nShutting down Ear...")
    finally:
        menu.stop()
        ear.cleanup()
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, termios.tcgetattr(sys.stdin.fileno()))


if __name__ == "__main__":
    start_ear()
