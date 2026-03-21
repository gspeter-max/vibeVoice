"""
brain.py — Parakeet Flow v2
============================
Receives raw 16-bit PCM audio over a Unix socket, transcribes it,
and auto-types the result into whatever app has focus.

Backend selection (set BACKEND env var before running):
  BACKEND=faster_whisper ./start.sh   ← default, no extra setup needed
  BACKEND=openvino       ./start.sh   ← requires OpenVINO setup (see backend_openvino.py)
"""

import os
import sys
import socket
import time
import numpy as np
from pynput.keyboard import Controller

# ── Socket config ──────────────────────────────────────────────────────────────
SOCKET_PATH = "/tmp/parakeet.sock"

# ── Keyboard ───────────────────────────────────────────────────────────────────
keyboard = Controller()

# ── Backend selection ──────────────────────────────────────────────────────────
BACKEND = os.environ.get("BACKEND", "faster_whisper").lower().strip()

def load_backend(model_name="base.en"):
    if "parakeet-tdt" in model_name:
        print(f"[Brain] Backend: NVIDIA Parakeet-TDT (via sherpa-onnx)")
        import backend_parakeet as backend
        model = backend.load_model(model_name)
        return backend, model

    if BACKEND == "openvino":
        print("[Brain] Backend: whisper.cpp + OpenVINO (Intel iGPU)")
        try:
            import backend_openvino as backend
            model = backend.load_model(model_name)
            return backend, model
        except RuntimeError as e:
            print(f"[Brain] ⚠️  OpenVINO unavailable: {e}")
            print("[Brain] ↩️  Falling back to faster-whisper...")

    # Default: faster-whisper
    print(f"[Brain] Backend: faster-whisper + {model_name} + INT8 (CPU)")
    import backend_faster_whisper as backend
    model = backend.load_model(model_name)
    return backend, model


# ── Socket server ──────────────────────────────────────────────────────────────
def start_server():
    backend, model = load_backend("base.en")

    # Clean up any stale socket
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)

    print(f"[Brain] ✅ Socket ready at {SOCKET_PATH}")
    print("[Brain] Waiting for audio from Ear...")
    sys.stdout.flush()

    try:
        while True:
            conn, _ = server.accept()
            t_start = time.perf_counter()

            # Read all audio data
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            conn.close()

            if not data:
                continue

            # Check if it might be a text command instead of audio
            if len(data) < 256:
                try:
                    command = data.decode("utf-8").strip()
                    if command.startswith("CMD_SWITCH_MODEL:"):
                        new_model_name = command.split(":", 1)[1]
                        print(f"\n[Brain] 🔄 Switch model requested: {new_model_name}")
                        sys.stdout.flush()

                        try:
                            # Force garbage collection to free up memory before loading new model
                            import gc
                            model = None
                            gc.collect()

                            # Re-load backend dynamically to allow switching between Whisper and Parakeet
                            backend, model = load_backend(new_model_name)
                            print(f"[Brain] ✅ Successfully switched to {new_model_name}")
                        except Exception as e:
                            print(f"[Brain] ❌ Failed to switch model: {e}")

                        sys.stdout.flush()
                        continue
                except UnicodeDecodeError:
                    pass

            # Convert raw 16-bit PCM → float32 numpy array
            try:
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            except ValueError as e:
                print(f"[Brain] ❌ Invalid audio data received: {e}")
                sys.stdout.flush()
                continue

            # Skip clips shorter than 0.3 seconds
            if len(audio_array) < 4800:
                print("[Brain] ⚠️  Audio too short — skipped")
                sys.stdout.flush()
                continue

            duration_sec = len(audio_array) / 16000.0
            print(f"[Brain] 🎙️  Audio: {duration_sec:.1f}s — transcribing...")
            sys.stdout.flush()

            # Transcribe
            text = backend.transcribe(model, audio_array)

            t_elapsed = time.perf_counter() - t_start

            if text:
                print(f"[Brain] 📝 [{t_elapsed:.2f}s] → \"{text}\"")
                sys.stdout.flush()
                # Auto-type into active app + trailing space
                keyboard.type(text + " ")
            else:
                print(f"[Brain] 🔇 [{t_elapsed:.2f}s] Nothing detected")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[Brain] Shutting down...")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


if __name__ == "__main__":
    start_server()