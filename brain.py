"""
brain.py — Parakeet Flow v2 (Streaming Edition)
=================================================
Receives streamed audio chunks in real-time via Unix socket.
Accumulates until the connection closes, then transcribes immediately.
No more waiting for the entire blob to arrive after recording stops.
"""

import os
import sys
import socket
import time
import threading
import numpy as np
from pynput.keyboard import Controller

SOCKET_PATH = "/tmp/parakeet.sock"
keyboard = Controller()
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

    print(f"[Brain] Backend: faster-whisper + {model_name} + INT8 (CPU)")
    import backend_faster_whisper as backend
    model = backend.load_model(model_name)
    return backend, model


# Global state
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


def handle_connection(conn):
    """Handle one streaming connection: accumulate chunks → transcribe."""
    t_connect = time.perf_counter()
    chunks = []
    total_bytes = 0

    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            chunks.append(data)
            total_bytes += len(data)
    except Exception as e:
        print(f"[Brain] ⚠️  Recv error: {e}")
    finally:
        conn.close()

    t_received = time.perf_counter()

    if total_bytes == 0:
        return

    raw = b"".join(chunks)

    # Check if it's a command
    if total_bytes < 256:
        try:
            text = raw.decode("utf-8").strip()
            if text.startswith("CMD_SWITCH_MODEL:"):
                new_model = text.split(":", 1)[1]
                print(f"\n[Brain] 🔄 Switch model: {new_model}")
                sys.stdout.flush()
                with backend_lock:
                    try:
                        import gc
                        backend_info["model"] = None
                        gc.collect()
                        nb, nm = load_backend(new_model)
                        backend_info["backend"] = nb
                        backend_info["model"] = nm
                        print(f"[Brain] ✅ Switched to {new_model}")
                    except Exception as e:
                        print(f"[Brain] ❌ Failed: {e}, falling back to base.en")
                        try:
                            nb, nm = load_backend("base.en")
                            backend_info["backend"] = nb
                            backend_info["model"] = nm
                            print(f"[Brain] ✅ Fallback OK")
                        except Exception as e2:
                            print(f"[Brain] 💥 Fallback failed: {e2}")
                    sys.stdout.flush()
                return
        except UnicodeDecodeError:
            pass  # Not a command, treat as audio

    # ── Process audio ──────────────────────────────────────────────
    with backend_lock:
        be = backend_info["backend"]
        mo = backend_info["model"]

    if be is None or mo is None:
        print("[Brain] ⚠️  No model loaded — skipping")
        send_hud("hide")
        return

    try:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        print(f"[Brain] ❌ Audio decode error: {e}")
        send_hud("hide")
        return

    if len(audio) < 4800:
        print("[Brain] ⚠️  Audio too short — skipped")
        send_hud("hide")
        return

    duration = len(audio) / 16000.0
    stream_time = (t_received - t_connect) * 1000
    print(f"[Brain] 🎙️  {duration:.1f}s audio (streamed in {stream_time:.0f}ms)")
    sys.stdout.flush()
    send_hud("process")

    t_start = time.perf_counter()

    try:
        text = be.transcribe(mo, audio)
    except Exception as e:
        print(f"[Brain] ❌ Transcription error: {e}")
        send_hud("hide")
        return

    t_elapsed = time.perf_counter() - t_start
    total_latency = time.perf_counter() - t_connect

    if text:
        print(f"[Brain] 📝 [{t_elapsed:.2f}s transcribe | {total_latency:.2f}s total] → \"{text}\"")
        sys.stdout.flush()
        keyboard.type(text + " ")
        send_hud("done")
    else:
        print(f"[Brain] 🔇 [{t_elapsed:.2f}s] Nothing detected")
        send_hud("hide")

    sys.stdout.flush()


def start_server():
    # Load model
    initial_backend, initial_model = load_backend("base.en")
    backend_info["backend"] = initial_backend
    backend_info["model"] = initial_model

    # ★ WARM UP — first inference is always slow
    print("[Brain] Warming up model...")
    dummy = np.zeros(8000, dtype=np.float32)
    try:
        initial_backend.transcribe(initial_model, dummy)
    except Exception:
        pass
    print("[Brain] Warm-up done ✓")

    # Clean up stale socket
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(10)

    print(f"[Brain] ✅ Streaming server ready at {SOCKET_PATH}")
    print("[Brain] Audio streams in real-time — transcription starts on disconnect")
    sys.stdout.flush()

    try:
        while True:
            conn, _ = server.accept()

            # ★ Each connection handled in its own thread
            # So brain can accept the NEXT recording while still transcribing
            t = threading.Thread(target=handle_connection, args=(conn,), daemon=True)
            t.start()

    except KeyboardInterrupt:
        print("\n[Brain] Shutting down...")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


if __name__ == "__main__":
    start_server()