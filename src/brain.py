"""
brain.py — Parakeet Flow v2 (Streaming Edition)
=================================================
Receives chunk and commit commands via Unix socket, decodes chunks as they arrive,
and pastes the stitched final text after session commit.
"""

import os
import sys
import socket
import time
import threading
from dataclasses import dataclass, field
import numpy as np

try:
    from pynput.keyboard import Controller
except Exception:  # pragma: no cover - test environments may not support pynput backends
    class Controller:  # type: ignore[override]
        def type(self, _text):
            return None

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
session_store = {}
session_store_lock = threading.Lock()


@dataclass
class SessionState:
    backend: object
    model: object
    received_count: int = 0
    done_count: int = 0
    closed: bool = False
    finalized: bool = False
    transcript_parts: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def send_hud(cmd: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.connect(('127.0.0.1', 57234))
        s.sendall(cmd.encode())
        s.close()
    except Exception:
        pass


def _get_or_create_session(session_id: str) -> SessionState:
    with session_store_lock:
        session = session_store.get(session_id)
        if session is not None:
            return session

        with backend_lock:
            backend = backend_info["backend"]
            model = backend_info["model"]

        session = SessionState(backend=backend, model=model)
        session_store[session_id] = session
        return session


def _split_text_into_comparable_words(text: str) -> list[str]:
    return [word for word in text.strip().split() if word]


def _remove_duplicate_chunk_prefix(
    previous_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 8,
) -> str:
    if not previous_chunk_text or not current_chunk_text:
        return current_chunk_text.strip()

    previous_chunk_words = _split_text_into_comparable_words(previous_chunk_text)
    current_chunk_words = _split_text_into_comparable_words(current_chunk_text)
    largest_possible_overlap = min(
        len(previous_chunk_words),
        len(current_chunk_words),
        max_overlap_words,
    )

    # Remove only exact repeated word prefixes from the current chunk.
    for overlap_word_count in range(largest_possible_overlap, 0, -1):
        if previous_chunk_words[-overlap_word_count:] == current_chunk_words[:overlap_word_count]:
            return " ".join(current_chunk_words[overlap_word_count:]).strip()

    return current_chunk_text.strip()


def _finalize_session_if_ready(session_id: str) -> None:
    with session_store_lock:
        session = session_store.get(session_id)
    if session is None:
        return

    text = None
    with session.lock:
        if session.finalized:
            return
        if not session.closed or session.done_count != session.received_count:
            return

        parts = [session.transcript_parts[idx] for idx in sorted(session.transcript_parts) if session.transcript_parts[idx]]
        text = " ".join(parts).strip()
        session.finalized = True

    short_id = session_id[:8]
    if text:
        print(f"[Brain] 📝 [session {short_id} | {session.received_count} chunks] → \"{text}\"", flush=True)
        send_hud("process")
        paste_instantly(text + " ")
        send_hud("done")
    else:
        print(f"[Brain] 🔇 [session {short_id}] Nothing detected", flush=True)
        send_hud("hide")

    with session_store_lock:
        session_store.pop(session_id, None)


def _handle_audio_chunk(session_id: str, seq: int, audio_bytes: bytes) -> None:
    session = _get_or_create_session(session_id)

    with session.lock:
        session.received_count += 1

    if len(audio_bytes) % 2:
        audio_bytes = audio_bytes[:-1]
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio = _normalize_audio(audio_int16)

    text = ""
    if audio is not None:
        duration_s = len(audio_int16) / 16000.0
        print(f"[Brain] 🎙️  [session {session_id[:8]} chunk {seq}] decode ({duration_s:.2f}s)", flush=True)
        try:
            backend = session.backend
            model = session.model
            if backend is None or model is None:
                print("[Brain] ⚠️  No model loaded — skipping chunk", flush=True)
            else:
                text = backend.transcribe(model, audio).strip()
        except Exception as e:
            print(f"[Brain] ❌ Chunk decode error: {e}", flush=True)
            text = ""
    else:
        print(f"[Brain] 🔇 [session {session_id[:8]} chunk {seq}] Nothing detected", flush=True)

    with session.lock:
        previous_chunk_text = session.transcript_parts.get(seq - 1, "")
        cleaned_chunk_text = _remove_duplicate_chunk_prefix(previous_chunk_text, text)
        session.transcript_parts[seq] = cleaned_chunk_text
        session.done_count += 1

    _finalize_session_if_ready(session_id)


def _mark_session_closed(session_id: str) -> None:
    session = _get_or_create_session(session_id)
    with session.lock:
        session.closed = True
    _finalize_session_if_ready(session_id)


import subprocess

def paste_instantly(text: str):
    """
    Pastes text instantly by injecting it into the macOS clipboard and simulating Cmd+V.
    Restores the original clipboard contents immediately afterward.
    """
    try:
        # 1. Save current clipboard contents (if any)
        try:
            old_clipboard = subprocess.check_output(['pbpaste'], stderr=subprocess.DEVNULL).decode('utf-8')
        except Exception:
            old_clipboard = ""

        # 2. Put the new text into the clipboard
        process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        process.communicate(input=text.encode('utf-8'))

        # 3. Simulate Cmd+V using AppleScript
        applescript = 'tell application "System Events" to keystroke "v" using command down'
        subprocess.run(['osascript', '-e', applescript], check=True)

        # 4. Give the OS a tiny moment to process the paste, then restore the old clipboard
        time.sleep(0.05)
        if old_clipboard:
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            process.communicate(input=old_clipboard.encode('utf-8'))
        else:
            pass 
            
    except Exception as e:
        print(f"[Brain] ⚠️  Paste failed: {e}. Falling back to slow typing.")
        keyboard.type(text)


def _normalize_audio(int16_audio: np.ndarray) -> np.ndarray | None:
    """Convert int16 PCM to normalized float32 audio and reject silence."""
    if len(int16_audio) == 0:
        return None

    audio = int16_audio.astype(np.float32) / 32768.0
    max_val = float(np.max(np.abs(audio)))
    if max_val < 0.001:
        return None
    if max_val > 0.01:
        audio = audio / max_val * 0.9
    return audio


def handle_connection(conn):
    """Handle one payload (chunk/commit/switch/raw audio fallback)."""
    t_connect = time.perf_counter()
    raw_audio = bytearray()
    total_bytes = 0

    try:
        while True:
            conn.settimeout(0.1)
            try:
                data = conn.recv(32768)
            except socket.timeout:
                continue

            if not data:
                break

            total_bytes += len(data)
            raw_audio.extend(data)
    except Exception as e:
        print(f"[Brain] ⚠️  Recv error: {e}")
    finally:
        conn.close()

    if total_bytes == 0:
        return

    print(f"[Brain] 📥 Received utterance: {total_bytes} bytes", flush=True)

    blob = bytes(raw_audio)

    if blob.startswith(b"CMD_SWITCH_MODEL:"):
        try:
            text_probe = blob.decode("utf-8").strip()
            new_model = text_probe.split(":", 1)[1]
        except Exception:
            print("[Brain] ⚠️  Bad switch command — skipping", flush=True)
            return

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

    if blob.startswith(b"CMD_SESSION_COMMIT:"):
        try:
            text_probe = blob.decode("utf-8").strip()
            session_id = text_probe.split(":", 1)[1]
        except Exception:
            print("[Brain] ⚠️  Bad commit command — skipping", flush=True)
            return

        print(f"[Brain] ✅ Commit received for session {session_id[:8]}", flush=True)
        _mark_session_closed(session_id)
        return

    if blob.startswith(b"CMD_AUDIO_CHUNK:") and b"\n\n" in blob:
        header, audio_bytes = blob.split(b"\n\n", 1)
        try:
            header_text = header.decode("utf-8").strip()
            _, session_id, seq_text = header_text.split(":", 2)
            seq = int(seq_text)
        except Exception as e:
            print(f"[Brain] ⚠️  Bad chunk header: {e}", flush=True)
            return

        _handle_audio_chunk(session_id, seq, audio_bytes)
        return

    with backend_lock:
        backend = backend_info["backend"]
        model = backend_info["model"]

    if backend is None or model is None:
        print("[Brain] ⚠️  No model loaded — skipping")
        send_hud("hide")
        return

    try:
        if len(blob) % 2:
            blob = blob[:-1]
        audio_int16 = np.frombuffer(blob, dtype=np.int16)
        if len(audio_int16) == 0:
            print("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return
        audio = _normalize_audio(audio_int16)
        if audio is None:
            print("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return

        duration_s = len(audio_int16) / 16000.0
        print(f"[Brain] 🎙️  Final utterance decode ({duration_s:.2f}s)", flush=True)
        send_hud("process")
        final_text = backend.transcribe(model, audio).strip()
        if not final_text:
            print("[Brain] 🔇 Nothing detected")
            send_hud("hide")
            return
    except Exception as e:
        print(f"[Brain] ❌ Audio decode error: {e}")
        sys.stdout.flush()
        send_hud("hide")
        return

    total_latency = time.perf_counter() - t_connect
    print(f"[Brain] 📝 [utterance | {total_latency:.2f}s total] → \"{final_text}\"")
    sys.stdout.flush()
    paste_instantly(final_text + " ")
    send_hud("done")
    sys.stdout.flush()


def start_server():
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
