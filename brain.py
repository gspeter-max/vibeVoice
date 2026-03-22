"""
brain.py — Parakeet Flow v2 (Multi-threaded Edition)
====================================================
Receives audio via Unix socket, queues it, and transcribes in a background thread.
This prevents "Connection Refused" errors when sending audio rapidly.
"""

import os
import sys
import socket
import time
import queue
import threading
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


# ── Global State ──────────────────────────────────────────────────────────────
audio_queue = queue.Queue()
backend_info = {"backend": None, "model": None}
backend_lock = threading.Lock()

def send_hud(cmd: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.connect(('127.0.0.1', 57234))
        s.sendall(cmd.encode())
        s.close()
    except Exception:
        pass

def worker():
    """Background thread that pulls from queue and transcribes."""
    while True:
        try:
            item = audio_queue.get()
            if item is None:
                break

            t_start = time.perf_counter()
            data, is_command = item

            if is_command:
                command = data.decode("utf-8").strip()
                if command.startswith("CMD_SWITCH_MODEL:"):
                    new_model_name = command.split(":", 1)[1]
                    print(f"\n[Brain] 🔄 Switch model requested: {new_model_name}")
                    sys.stdout.flush()
                    with backend_lock:
                        try:
                            import gc
                            backend_info["model"] = None
                            gc.collect()
                            new_backend, new_model = load_backend(new_model_name)
                            backend_info["backend"] = new_backend
                            backend_info["model"] = new_model
                            print(f"[Brain] ✅ Switched to {new_model_name}")
                        except Exception as e:
                            print(f"[Brain] ❌ Failed: {e}")
                            print(f"[Brain] ↩️  Falling back to base.en...")
                            try:
                                fb_backend, fb_model = load_backend("base.en")
                                backend_info["backend"] = fb_backend
                                backend_info["model"] = fb_model
                                print(f"[Brain] ✅ Fallback to base.en OK")
                            except Exception as e2:
                                print(f"[Brain] 💥 Fallback failed: {e2}")
                        sys.stdout.flush()
                audio_queue.task_done()
                continue

            # ── Process audio ──────────────────────────────────────────
            # Grab backend reference under lock (fast)
            with backend_lock:
                backend = backend_info["backend"]
                model = backend_info["model"]

            if backend is None or model is None:
                print("[Brain] ⚠️  No model loaded — skipping")
                audio_queue.task_done()
                send_hud("hide")
                continue

            try:
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                print(f"[Brain] ❌ Audio decode error: {e}")
                audio_queue.task_done()
                send_hud("hide")
                continue

            if len(audio_array) < 4800:
                print("[Brain] ⚠️  Audio too short — skipped")
                audio_queue.task_done()
                send_hud("hide")
                continue

            duration_sec = len(audio_array) / 16000.0
            print(f"[Brain] 🎙️  Processing: {duration_sec:.1f}s...")
            sys.stdout.flush()
            send_hud("process")

            # Transcribe WITHOUT holding the lock (so commands aren't blocked)
            try:
                text = backend.transcribe(model, audio_array)
            except Exception as e:
                print(f"[Brain] ❌ Transcription error: {e}")
                audio_queue.task_done()
                send_hud("hide")
                continue

            t_elapsed = time.perf_counter() - t_start

            if text:
                print(f"[Brain] 📝 [{t_elapsed:.2f}s] → \"{text}\"")
                sys.stdout.flush()
                keyboard.type(text + " ")
                send_hud("done")
            else:
                print(f"[Brain] 🔇 [{t_elapsed:.2f}s] Nothing detected")
                send_hud("hide")

            sys.stdout.flush()
            audio_queue.task_done()

        except Exception as e:
            print(f"[Brain] 💥 Worker error: {e}")
            sys.stdout.flush()
            send_hud("hide")

# ── Socket server ──────────────────────────────────────────────────────────────
def start_server():
    # Initial load
    initial_backend, initial_model = load_backend("base.en")
    backend_info["backend"] = initial_backend
    backend_info["model"] = initial_model

    # Start worker thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Clean up any stale socket
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(10) # Increased backlog

    print(f"[Brain] ✅ Multi-threaded Socket ready at {SOCKET_PATH}")
    print("[Brain] Ready to receive audio...")
    sys.stdout.flush()

    try:
        while True:
            conn, _ = server.accept()
            
            # Read all audio data immediately
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            conn.close() # Close immediately so Ear can move on

            if not data:
                continue

            # Identify if it's a command or audio
            is_command = (len(data) < 256)
            
            # Put into queue for the worker to process
            audio_queue.put((data, is_command))

    except KeyboardInterrupt:
        print("\n[Brain] Shutting down...")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


if __name__ == "__main__":
    start_server()
