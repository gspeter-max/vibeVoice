"""
ear.py — Parakeet Flow v2 (Streaming Edition)
===============================================
Streams audio chunks to brain.py in real-time over a persistent socket.
Brain accumulates and transcribes the moment the stream ends.

Hold RIGHT CMD → speak → release → text appears instantly.
"""

import os
import sys
import socket
import subprocess
import threading
import time
import pyaudio
import math
import select
import termios
import tty
import numpy as np
from pynput import keyboard

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
VOL_PORT        = 57235
HOLD_THRESHOLD  = 0.4

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
                    print("   Is Brain running? Check logs/brain.log\n", end="", flush=True)
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
                print("   Brain might be loading model. Check logs/brain.log\n", end="", flush=True)

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


class BrainOutputTailer(threading.Thread):
    def __init__(self, log_path: str):
        super().__init__(daemon=True)
        self.log_path = log_path
        self._stop = threading.Event()

    def run(self):
        for _ in range(30):
            if os.path.exists(self.log_path):
                break
            time.sleep(0.5)
        else:
            return
        with open(self.log_path, "r") as f:
            f.seek(0, 2)
            while not self._stop.is_set():
                line = f.readline()
                if line:
                    stripped = line.rstrip()
                    if stripped:
                        print(f"\r  🧠  {stripped}", flush=True)
                else:
                    time.sleep(0.05)

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
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self._lock = threading.Lock()
        self.last_rms = 0.0
        self.gain_multiplier = 2.5 # Increased from 1.1 to fix quiet mic issues
        self._total_frames = 0
        self.last_frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.34}

        # Enable macOS Voice Isolation if requested
        if os.environ.get("VOICE_ISOLATION", "0") == "1":
            _enable_macos_voice_isolation()
        else:
            print("[Ear] Voice Isolation disabled by default (set VOICE_ISOLATION=1 to enable)", flush=True)

        if input_device_index is None:
            self.input_device_index = self.p.get_default_input_device_info().get("index")
        else:
            self.input_device_index = input_device_index

        self.active_mic_name = self.p.get_device_info_by_index(self.input_device_index).get("name")

        self.hud_proc = None
        self._cmd_press_time = 0.0
        self._toggle_active = False

        # ★ MOUSE CONTROL: Hold-to-record for voice activation
        self._mouse_press_start_time = 0.0  # When mouse button was pressed
        self._is_holding = False              # Currently holding mouse button
        self._recording_from_hold = False     # Recording started from mouse hold

        # ★ STREAMING: persistent socket connection to brain
        self._brain_sock = None
        self._brain_sock_lock = threading.Lock()

        # ★ ALWAYS LISTENING MODE: Open stream once at startup
        self.stream = None
        self._open_mic_stream()
        print(f"[Ear] Mic selected: {self.active_mic_name} ✓", flush=True)

    def _send_hud(self, cmd):
        print(f"[Ear] 📤 _send_hud() called with cmd='{cmd}'", flush=True)
        try:
            print(f"[Ear] 📤 Creating socket...", flush=True)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"[Ear] 📤 Setting timeout...", flush=True)
            s.settimeout(0.2)
            print(f"[Ear] 📤 Connecting to 127.0.0.1:57234...", flush=True)
            s.connect(('127.0.0.1', 57234))
            print(f"[Ear] 📤 Sending data...", flush=True)
            s.sendall(cmd.encode())
            print(f"[Ear] 📤 Closing socket...", flush=True)
            s.close()
            print(f"[Ear] ✅ HUD command sent successfully", flush=True)
        except Exception as e:
            print(f"[Ear] ❌ Failed to send HUD command: {e}", flush=True)
            import traceback
            traceback.print_exc()

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

    # ★ STREAMING: open a socket to brain BEFORE recording starts
    def _open_brain_stream(self) -> bool:
        """Open a persistent socket to brain for streaming chunks."""
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

    # ★ STREAMING: send chunk immediately to brain
    def _stream_chunk_to_brain(self, chunk_bytes: bytes):
        with self._brain_sock_lock:
            if self._brain_sock is None:
                return
            try:
                self._brain_sock.sendall(chunk_bytes)
            except (BrokenPipeError, ConnectionResetError):
                # Brain closed early — stop streaming silently
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

    # ★ STREAMING: close socket = signal brain to transcribe
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

            # ★ STREAMING: send each chunk immediately
            self._stream_chunk_to_brain(chunk_bytes)

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

        self.stream = self.p.open(
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

    def on_press(self, key):
        if not _is_right_cmd(key):
            return

        print(f"[Ear] 🔵 Right CMD pressed - on_press() called", flush=True)

        if self._toggle_active:
            self._toggle_active = False
            self._stop_and_send()
            return

        with self._lock:
            if self.is_recording:
                print(f"[Ear] ⚠️ Already recording, ignoring press", flush=True)
                return

        print(f"[Ear] 🔵 About to open brain stream", flush=True)
        # ★ STREAMING: open brain connection FIRST (before recording flag)
        if not self._open_brain_stream():
            print(f"[Ear] ❌ Failed to open brain stream, aborting", flush=True)
            return

        print(f"[Ear] 🔵 Brain stream opened successfully", flush=True)
        with self._lock:
            self.is_recording = True
            self.last_rms = 0.0
            self._total_frames = 0

        self._cmd_press_time = time.time()
        print("\r\n" + "─" * 50, flush=True)
        print(f"\r🎙️  RECORDING+STREAMING ({self.active_mic_name})", flush=True)

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

        held = time.time() - self._cmd_press_time

        if held >= HOLD_THRESHOLD:
            self._stop_and_send()
        else:
            self._toggle_active = True
            print(f"\r\n⏸️  Toggle mode — tap Right CMD again to stop", flush=True)

    def _stop_and_send(self):
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False
            total = self._total_frames

        duration = (total * CHUNK) / RATE
        print(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n", end="", flush=True)

        threading.Thread(target=self._send_hud, args=("process",), daemon=True).start()

        # ★ STREAMING: just close the socket — brain already has all chunks
        # This triggers brain to start transcribing immediately
        threading.Thread(target=self._close_brain_stream, daemon=True).start()

    def record_loop(self):
        while True:
            time.sleep(0.05)
            with self._lock:
                recording = self.is_recording
                rms = self.last_rms
            if recording:
                meter = "█" * min(int(rms * 500), 50)
                print(f"\r  Level: [{meter:<50}]", end="", flush=True)

    def cleanup(self):
        self._close_brain_stream()
        self._close_mic_stream()
        self.p.terminate()


def start_ear():
    # Use repo root as base for logs/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, "logs", "brain.log")

    p_temp = pyaudio.PyAudio()
    selected_mic_index = select_mic(p_temp)
    p_temp.terminate()

    tailer = BrainOutputTailer(log_path)
    tailer.start()

    menu = TerminalMenu()
    menu.start()

    ear = Ear(input_device_index=selected_mic_index)
    listener = keyboard.Listener(on_press=ear.on_press, on_release=ear.on_release)
    listener.start()

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
    print(" Brain output will appear below as you speak.\n")
    print("─" * 52)

    try:
        ear.record_loop()
    except KeyboardInterrupt:
        print("\r\n\nShutting down Ear...")
    finally:
        menu.stop()
        tailer.stop()
        ear.cleanup()
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, termios.tcgetattr(sys.stdin.fileno()))


if __name__ == "__main__":
    start_ear()