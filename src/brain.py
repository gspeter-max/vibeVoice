"""
brain.py — Parakeet Flow v2 (Streaming Edition)
=================================================
Receives streamed audio chunks in real-time via Unix socket.
Accumulates until the connection closes, then transcribes immediately.
Optimized for Intel Mac with Silero VAD (Voice Activity Detection).
"""

import os
import sys
import socket
import time
import threading
import numpy as np
from pynput.keyboard import Controller

# VAD settings
VAD_MODEL_PATH = os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx")
VAD_THRESHOLD = 0.5
VAD_SILENCE_TIMEOUT = 0.4  # seconds of silence before auto-stop

SOCKET_PATH = "/tmp/parakeet.sock"
keyboard = Controller()
BACKEND = os.environ.get("BACKEND", "faster_whisper").lower().strip()

# ── Silero VAD Engine ────────────────────────────────────────────────────────
class SileroVAD:
    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
        
        # Auto-detect model version by checking input names
        input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"[VAD] Model inputs: {input_names}")
        
        if 'state' in input_names:
            # v5 — single state tensor
            self._version = 5
            # Find state shape from model metadata
            for inp in self.session.get_inputs():
                if inp.name == 'state':
                    state_shape = inp.shape  # e.g. [2, 1, 128]
                    break
            self._state = np.zeros(state_shape, dtype=np.float32)
        else:
            # v3/v4 — separate h and c
            self._version = 3
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        """Reset VAD state between recordings."""
        if self._version == 5:
            self._state = np.zeros_like(self._state)
        else:
            self._h = np.zeros_like(self._h)
            self._c = np.zeros_like(self._c)

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        if len(audio_chunk) == 0:
            return 0.0
        
        # Silero expects exactly 512 samples for 16kHz
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        elif len(audio_chunk) > 512:
            audio_chunk = audio_chunk[:512]

        if self._version == 5:
            ort_inputs = {
                'input': audio_chunk.reshape(1, -1),
                'sr': np.array([sample_rate], dtype=np.int64),
                'state': self._state,
            }
            out, new_state = self.session.run(None, ort_inputs)
            self._state = new_state
        else:
            ort_inputs = {
                'input': audio_chunk.reshape(1, -1),
                'sr': np.array([sample_rate], dtype=np.int64),
                'h': self._h,
                'c': self._c,
            }
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c
        
        return out[0][0]

# Global VAD instance
vad_engine = None

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
    """Handle one streaming connection: accumulate → transcribe when ear closes."""
    t_connect = time.perf_counter()
    chunks = []
    total_bytes = 0

    # VAD State — HUD feedback ONLY, never closes connection
    speech_detected = False
    last_speech_time = time.perf_counter()
    thinking_sent = False

    # Reset VAD state for fresh recording
    if vad_engine:
        vad_engine.reset()

    try:
        while True:
            # VAD silence → just update HUD (NEVER break)
            if (speech_detected
                and not thinking_sent
                and (time.perf_counter() - last_speech_time > VAD_SILENCE_TIMEOUT)):
                print("[Brain] 💤 VAD: Speech paused (HUD only)")
                send_hud("thinking")
                thinking_sent = True

            conn.settimeout(0.1)
            try:
                data = conn.recv(32768)
            except socket.timeout:
                continue

            if not data:
                break  # ← ONLY exit: ear closed the socket

            chunks.append(data)
            total_bytes += len(data)

            # --- Real-time VAD (wrapped in try to never kill connection) ---
            try:
                if vad_engine and len(data) >= 1024:
                    audio_chunk = np.frombuffer(data[-1024:], dtype=np.int16).astype(np.float32) / 32768.0
                    score = vad_engine.is_speech(audio_chunk)
                    if score > VAD_THRESHOLD:
                        if not speech_detected:
                            print("[Brain] 🗣️  VAD: Speech started")
                        speech_detected = True
                        last_speech_time = time.perf_counter()
                        thinking_sent = False  # Reset if speech resumes
            except Exception as ve:
                print(f"[Brain] ⚠️  VAD error (non-fatal): {ve}")

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
                        except Exception as e2:
                            print(f"[Brain] 💥 Fallback failed: {e2}")
                    sys.stdout.flush()
                return
        except UnicodeDecodeError:
            pass

    # ── Process audio ──
    with backend_lock:
        be = backend_info["backend"]
        mo = backend_info["model"]

    if be is None or mo is None:
        print("[Brain] ⚠️  No model loaded — skipping")
        send_hud("hide")
        return

    try:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        
        # ── DEBUG: Show actual audio level ──
        max_val = np.max(np.abs(audio))
        rms_val = np.sqrt(np.mean(audio ** 2))
        nonzero = np.count_nonzero(audio)
        print(f"[Brain] 📊 max={max_val:.4f} rms={rms_val:.4f} nonzero={nonzero}/{len(audio)}")

        if max_val < 0.005:
            print("[Brain] ⚠️  Audio is silence — mic not capturing")
            send_hud("hide")
            return
            
        if max_val > 0.01:
            audio = audio / max_val * 0.9
    except Exception as e:
        print(f"[Brain] ❌ Audio decode error: {e}")
        send_hud("hide")
        return

    if len(audio) < 4800:
        print("[Brain] ⚠️  Audio too short — skipped")
        send_hud("hide")
        return

    duration = len(audio) / 16000.0
    print(f"[Brain] 🎙️  {duration:.1f}s audio received")
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
        print(f"[Brain] 📝 [{t_elapsed:.2f}s | {total_latency:.2f}s total] → \"{text}\"")
        sys.stdout.flush()
        keyboard.type(text + " ")
        send_hud("done")
    else:
        print(f"[Brain] 🔇 [{t_elapsed:.2f}s] Nothing detected")
        send_hud("hide")

    sys.stdout.flush()


def start_server():
    global vad_engine
    
    # Initialize VAD
    print("[Brain] Loading Silero VAD...")
    try:
        vad_engine = SileroVAD(VAD_MODEL_PATH)
        print("[Brain] ✅ VAD Engine ready.")
    except Exception as e:
        print(f"[Brain] ❌ VAD initialization failed: {e}")

    # Load model
    initial_backend, initial_model = load_backend("base.en")
    backend_info["backend"] = initial_backend
    backend_info["model"] = initial_model

    # ★ WARM UP
    print("[Brain] Warming up model...")
    dummy = np.zeros(8000, dtype=np.float32)
    try:
        initial_backend.transcribe(initial_model, dummy)
    except Exception:
        pass
    print("[Brain] Warm-up done ✓")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(10)

    print(f"[Brain] ✅ Streaming server ready at {SOCKET_PATH}")
    sys.stdout.flush()

    try:
        while True:
            conn, _ = server.accept()
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
