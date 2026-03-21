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
from pynput import keyboard

# ── Config ─────────────────────────────────────────────────────────────────────
SOCKET_PATH = "/tmp/parakeet.sock"
HOTKEY      = keyboard.Key.cmd_r   # Right Command key — change if needed
BACKEND     = os.environ.get("BACKEND", "faster_whisper")

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


# ── Main ear class ─────────────────────────────────────────────────────────────
class Ear:
    def __init__(self):
        self.p            = pyaudio.PyAudio()
        self.stream       = None
        self.frames       = []
        self.is_recording = False
        self._lock        = threading.Lock()
        self.last_rms     = 0.0

    def _audio_callback(self, in_data, frame_count, time_info, status):
        with self._lock:
            if self.is_recording:
                self.frames.append(in_data)
                self.last_rms = get_rms(in_data)
        return (None, pyaudio.paContinue)

    # ── Hotkey handlers ────────────────────────────────────────────────────────
    def on_press(self, key):
        if key != HOTKEY:
            return
        with self._lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.frames = []
            self.last_rms = 0.0

        print("\r\n" + "─" * 50, flush=True)
        print("\r🎙️  RECORDING — release key to stop", flush=True)

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"\r❌ Mic error: {e}", flush=True)
            with self._lock:
                self.is_recording = False

    def on_release(self, key):
        if key != HOTKEY:
            return
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
                return
            except ConnectionRefusedError:
                print(f"\r❌ Brain refused connection (attempt {attempt+1}/3) — is it running?\n", end="", flush=True)
                time.sleep(1)
            except Exception as e:
                print(f"\r❌ Socket error: {e}\n", end="", flush=True)
                return

        print("\r❌ Could not reach Brain after 3 attempts. Run ./start.sh to restart.\n", end="", flush=True)

    def cleanup(self):
        self.p.terminate()


# ── Entrypoint ─────────────────────────────────────────────────────────────────
def start_ear():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain.log")

    # Start brain log tailer in background
    tailer = BrainOutputTailer(log_path)
    tailer.start()
    
    # Start terminal input menu
    menu = TerminalMenu()
    menu.start()

    ear = Ear()
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