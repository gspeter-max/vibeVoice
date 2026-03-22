"""
ear.py — Parakeet Flow v2
==========================
Listens for the hotkey, records mic audio, sends to brain.py over Unix socket,
and shows the brain's transcription output live in this terminal.

Hold RIGHT CMD → speak → release → text appears in active app.
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

# ── Config ─────────────────────────────────────────────────────────────────────
SOCKET_PATH     = "/tmp/parakeet.sock"
BACKEND         = os.environ.get("BACKEND", "faster_whisper")
VOL_PORT        = 57235               # UDP port for sending mic RMS to HUD
HOLD_THRESHOLD  = 0.4                 # seconds — above this = hold mode, below = toggle mode

# macOS / pynput key matching ———————————————————————————————————————
# pynput may report Right CMD as Key.cmd_r, or as a KeyCode with name='cmd_r',
# or vk=55. We match ALL of these and explicitly reject anything that looks like
# Left CMD (Key.cmd, name='cmd', vk=None-ambiguous) so Left CMD never fires.
_RCMD_VK = 54  # macOS virtual key code for Right CMD (Left CMD = 55)

def _is_right_cmd(key) -> bool:
    """Return True ONLY if key is Right CMD, never Left CMD."""
    # Direct enum match (pynput 1.7+)
    if key == keyboard.Key.cmd_r:
        return True
    # Some pynput versions return a KeyCode for special keys
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True
    # vk code match (most reliable on macOS)
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == _RCMD_VK:
        return True
    return False

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000
CHUNK    = 1024

MODELS = [
    "tiny.en",
    "base.en",
    "small.en",
    "Systran/faster-distil-whisper-large-v3",
    "medium.en",
    "large-v2",
    "large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3"
]

# ── Volume RMS helper ──────────────────────────────────────────────────────────
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
        client.close()
    except Exception as e:
        print(f"\n❌ Failed to send switch command: {e}\n", end="", flush=True)

def run_self_test():
    """Sends a synthetic 440Hz sine wave to the Brain to verify the pipe."""
    print("\n🧪 Running SELF-TEST (synthetic audio)...\n", end="", flush=True)
    import numpy as np
    
    # 1 second of 440Hz sine wave at 16kHz
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(RATE * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16).tobytes()
    
    # Borrow Ear's _send_to_brain logic but with custom data
    temp_ear = Ear()
    temp_ear.frames = [audio_data]
    temp_ear._send_to_brain()

# ── Terminal Menu Thread ───────────────────────────────────────────────────────
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
                    if c == '1':
                        send_switch_command(MODELS[0])
                    elif c == '2':
                        send_switch_command(MODELS[1])
                    elif c == '3':
                        send_switch_command(MODELS[2])
                    elif c == '4':
                        send_switch_command(MODELS[3])
                    elif c == '5':
                        send_switch_command(MODELS[4])
                    elif c == '6':
                        send_switch_command(MODELS[5])
                    elif c == '7':
                        send_switch_command(MODELS[6])
                    elif c == '8':
                        send_switch_command(MODELS[7])
                    elif c == '9':
                        send_switch_command(MODELS[8])
                    elif c == '0':
                        send_switch_command(MODELS[9])
                    elif c.lower() == 't':
                        threading.Thread(target=run_self_test, daemon=True).start()
                    elif c == '\x03': # Ctrl+C
                        os.kill(os.getpid(), 2) # send SIGINT
                        break
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old_settings)

    def stop(self):
        self._stop.set()


# ── Brain log tail ─────────────────────────────────────────────────────────────
class BrainOutputTailer(threading.Thread):
    """
    Reads brain.log in a background thread and prints new lines to this terminal.
    This gives you live transcription output without switching windows.
    """
    def __init__(self, log_path: str):
        super().__init__(daemon=True)
        self.log_path = log_path
        self._stop = threading.Event()

    def run(self):
        # Wait for log file to appear
        for _ in range(30):
            if os.path.exists(self.log_path):
                break
            time.sleep(0.5)
        else:
            return

        with open(self.log_path, "r") as f:
            # Seek to end — only show new lines from now
            f.seek(0, 2)
            while not self._stop.is_set():
                line = f.readline()
                if line:
                    stripped = line.rstrip()
                    if stripped:
                        # Add carriage return to play nice with raw mode terminal
                        print(f"\r  🧠  {stripped}", flush=True)
                else:
                    time.sleep(0.05)

    def stop(self):
        self._stop.set()


def select_mic(p):
    """Interactive menu to select the microphone."""
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
                print("❌ Invalid index. Please choose from the list above.")
        except ValueError:
            print("❌ Please enter a valid number.")

# ── Main ear class ─────────────────────────────────────────────────────────────
class Ear:
    def __init__(self, input_device_index=None):
        self.p            = pyaudio.PyAudio()
        self.stream       = None
        self.frames       = []
        self.is_recording = False
        self._lock        = threading.Lock()
        self.last_rms     = 0.0

        if input_device_index is None:
             self.input_device_index = self.p.get_default_input_device_info().get("index")
        else:
             self.input_device_index = input_device_index

        self.active_mic_name = self.p.get_device_info_by_index(self.input_device_index).get("name")
        self.gain_multiplier = 2.0  # Boost volume by 2x
        # HUD is managed by start.sh — no subprocess here.
        self.hud_proc = None

        # Hold vs toggle mode tracking
        self._cmd_press_time = 0.0
        self._toggle_active  = False  # True while toggle mode recording is live



    def _send_hud(self, cmd):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect(('127.0.0.1', 57234))
            s.sendall(cmd.encode())
            s.close()
        except Exception:
            pass

    def _start_volume_sender(self):
        """Send mic RMS to HUD via UDP at ~25fps while recording."""
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def _sender():
            while True:
                with self._lock:
                    if not self.is_recording:
                        break
                    rms = self.last_rms
                try:
                    udp.sendto(f"vol:{rms:.4f}".encode(), ('127.0.0.1', VOL_PORT))
                except Exception:
                    pass
                time.sleep(0.04)
            udp.close()

        threading.Thread(target=_sender, daemon=True).start()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        with self._lock:
            if self.is_recording:
                # Apply digital gain boost
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                boosted_audio = (audio_data.astype(np.float32) * self.gain_multiplier).clip(-32768, 32767).astype(np.int16)
                
                self.frames.append(boosted_audio.tobytes())
                self.last_rms = get_rms(boosted_audio.tobytes())
        return (None, pyaudio.paContinue)

    # ── Hotkey handlers ────────────────────────────────────────────────────────
    def on_press(self, key):
        if not _is_right_cmd(key):
            return

        # Second tap in toggle mode → stop recording, send to brain
        if self._toggle_active:
            self._toggle_active = False
            self._stop_and_send()             # stop mic + send audio to brain
            return

        with self._lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.frames = []
            self.last_rms = 0.0

        self._cmd_press_time = time.time()
        print("\r\n" + "─" * 50, flush=True)
        print(f"\r🎙️  RECORDING ({self.active_mic_name}) — release key to stop", flush=True)
        self._send_hud("listen")

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self._start_volume_sender()  # send mic RMS to HUD in background
        except Exception as e:
            print(f"\r❌ Mic error: {e}", flush=True)
            with self._lock:
                self.is_recording = False

    def on_release(self, key):
        if not _is_right_cmd(key):
            return

        # Ignore release if already handled by second-tap logic
        if self._toggle_active:
            return

        with self._lock:
            if not self.is_recording:
                return

        held = time.time() - self._cmd_press_time

        if held >= HOLD_THRESHOLD:
            # ── HOLD MODE: stop now and send immediately ───────────────────
            self._stop_and_send()
        else:
            # TOGGLE MODE: keep recording silently
            self._toggle_active = True
            print(f"\r\n⏸️  Toggle mode — tap Right CMD again to stop", flush=True)

    def _stop_and_send(self):
        """Stop the mic stream and dispatch audio to brain. Called by both modes."""
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        n_frames = len(self.frames)
        duration  = (n_frames * CHUNK) / RATE
        print(f"\r\n⏹️  Recorded {duration:.1f}s — sending to Brain...\n", end="", flush=True)
        self._send_hud("hide")

        threading.Thread(target=self._send_to_brain, daemon=True).start()

    # ── Audio capture loop ─────────────────────────────────────────────────────
    def record_loop(self):
        while True:
            time.sleep(0.05)
            with self._lock:
                recording = self.is_recording
                rms = self.last_rms

            if recording:
                meter = "█" * min(int(rms * 120), 50)
                print(f"\r  Level: [{meter:<50}]", end="", flush=True)

    # ── Send audio to brain ────────────────────────────────────────────────────
    def _send_to_brain(self):
        if not self.frames:
            print("\r⚠️  No audio captured\n", end="", flush=True)
            self._send_hud("hide")
            return

        audio_data = b"".join(self.frames)

        # Retry up to 3 times in case brain is still starting
        for attempt in range(3):
            if not os.path.exists(SOCKET_PATH):
                print(f"\r❌ Brain socket not found (attempt {attempt+1}/3)\n", end="", flush=True)
                time.sleep(1)
                continue
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(120)
                client.connect(SOCKET_PATH)
                client.sendall(audio_data)
                client.shutdown(socket.SHUT_WR)
                client.close()
                print("\r✅ Sent to Brain — waiting for transcription...\n", end="", flush=True)
                # We don't hide here! Brain will send "done" when finished.
                return
            except ConnectionRefusedError:
                print(f"\r❌ Brain refused connection (attempt {attempt+1}/3) — is it running?\n", end="", flush=True)
                time.sleep(1)
            except Exception as e:
                print(f"\r❌ Socket error: {e}\n", end="", flush=True)
                self._send_hud("hide")
                return

        print("\r❌ Could not reach Brain after 3 attempts. Run ./start.sh to restart.\n", end="", flush=True)
        self._send_hud("hide")

    def cleanup(self):
        self.p.terminate()


# ── Entrypoint ─────────────────────────────────────────────────────────────────
def start_ear():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain.log")

    # Interactive Mic Selection
    p_temp = pyaudio.PyAudio()
    selected_mic_index = select_mic(p_temp)
    p_temp.terminate()

    # Start brain log tailer in background
    tailer = BrainOutputTailer(log_path)
    tailer.start()

    # Start terminal input menu (calls tty.setcbreak)
    menu = TerminalMenu()
    menu.start()

    ear = Ear(input_device_index=selected_mic_index)
    listener = keyboard.Listener(on_press=ear.on_press, on_release=ear.on_release)
    listener.start()


    backend_label = {
        "faster_whisper": "faster-whisper + distil-large-v3 (INT8)",
        "openvino":       "whisper.cpp + OpenVINO (Intel iGPU)",
    }.get(BACKEND, BACKEND)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        🎙️  PARAKEET FLOW  v2  — ACTIVE           ║")
    print(f"║  Backend : {backend_label:<38}║")
    print(f"║  Hotkey  : RIGHT CMD (hold to record)            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(" Press [1] tiny.en      (Fastest)")
    print(" Press [2] base.en      (Default)")
    print(" Press [3] small.en     (Accurate)")
    print(" Press [4] distil-large (Fast Max Accuracy)")
    print(" Press [5] medium.en    (High Accuracy)")
    print(" Press [6] large-v2     (Very High Accuracy)")
    print(" Press [7] large-v3     (Max Accuracy)")
    print(" Press [8] turbo        (Ultra-Fast Max)")
    print(" Press [9] Parakeet v2  (Vibe Coding English)")
    print(" Press [0] Parakeet v3  (Multilingual TDT)")
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
        
        # Restore terminal settings just in case
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, termios.tcgetattr(sys.stdin.fileno()))


if __name__ == "__main__":
    start_ear()