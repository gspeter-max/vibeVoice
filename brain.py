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

def worker():
    """Background thread that pulls from queue and transcribes."""
    while True:
        try:
            item = audio_queue.get()
            if item is None: break # Shutdown signal

            t_start = time.perf_counter()
            data, is_command = item

            with backend_lock:
                backend = backend_info["backend"]
                model = backend_info["model"]

                if is_command:
                    command = data.decode("utf-8").strip()
                    if command.startswith("CMD_SWITCH_MODEL:"):
                        new_model_name = command.split(":", 1)[1]
                        print(f"\n[Brain] 🔄 Switch model requested: {new_model_name}")
                        try:
                            import gc
                            backend_info["model"] = None
                            gc.collect()
                            
                            # Re-load backend dynamically
                            new_backend, new_model = load_backend(new_model_name)
                            backend_info["backend"] = new_backend
                            backend_info["model"] = new_model
                            print(f"[Brain] ✅ Successfully switched to {new_model_name}")
                        except Exception as e:
                            print(f"[Brain] ❌ Failed to switch model: {e}")
                    audio_queue.task_done()
                    continue

                # Process Audio
                try:
                    audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                except Exception as e:
                    print(f"[Brain] ❌ Audio error: {e}")
                    audio_queue.task_done()
                    continue

                if len(audio_array) < 4800:
                    print("[Brain] ⚠️  Audio too short — skipped")
                    audio_queue.task_done()
                    continue

                duration_sec = len(audio_array) / 16000.0
                print(f"[Brain] 🎙️  Processing: {duration_sec:.1f}s...")
                sys.stdout.flush()

                # Actual Transcription
                text = backend.transcribe(model, audio_array)
                t_elapsed = time.perf_counter() - t_start

                if text:
                    print(f"[Brain] 📝 [{t_elapsed:.2f}s] → \"{text}\"")
                    keyboard.type(text + " ")
                else:
                    print(f"[Brain] 🔇 [{t_elapsed:.2f}s] Nothing detected")
                
                sys.stdout.flush()
            
            audio_queue.task_done()
        except Exception as e:
            print(f"[Brain] 💥 Worker error: {e}")

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
