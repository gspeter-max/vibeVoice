# ASR Mode Toggle and Two-Pass Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add startup-selectable `streaming` and `twopass` ASR modes so the app can show low-latency draft text during speech while still committing a single final transcription to the active app.

**Architecture:** Keep `ear.py` as the audio transport and control layer. Resolve `ASR_MODE` once at startup, run a Sherpa online recognizer for draft text when the mode requires it, keep the existing offline recognizers for final text, and send partial text to the HUD instead of typing partial corrections into arbitrary apps.

**Tech Stack:** Python 3.11, Bash, `sherpa-onnx`, `faster-whisper`, NumPy, PySide6, Pytest, Unix sockets, `apply_patch`

---

### Task 1: Add startup ASR mode selection and compatibility checks

**Files:**
- Create: `src/asr_mode.py`
- Modify: `start.sh`
- Modify: `src/brain.py`
- Test: `tests/test_asr_mode.py`

- [ ] **Step 1: Write the failing mode-resolution tests**

Create `tests/test_asr_mode.py`:

```python
from asr_mode import resolve_asr_mode


def test_resolve_asr_mode_defaults_to_twopass():
    cfg = resolve_asr_mode(requested_mode="", backend="faster_whisper", model_name="base.en")
    assert cfg.mode == "twopass"
    assert cfg.partial_preview is True


def test_streaming_requires_streaming_capable_final_backend():
    try:
        resolve_asr_mode(
            requested_mode="streaming",
            backend="faster_whisper",
            model_name="base.en",
        )
    except ValueError as exc:
        assert "streaming" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported streaming combination")


def test_twopass_allows_offline_final_backend():
    cfg = resolve_asr_mode(
        requested_mode="twopass",
        backend="faster_whisper",
        model_name="base.en",
    )
    assert cfg.mode == "twopass"
    assert cfg.use_online_draft is True
    assert cfg.use_offline_final is True
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:
```bash
pytest tests/test_asr_mode.py -v
```

Expected:
- The run fails because `asr_mode.py` and `resolve_asr_mode()` do not exist yet.

- [ ] **Step 3: Implement the mode resolver**

Create `src/asr_mode.py` with a focused config object and validation logic:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AsrModeConfig:
    mode: str
    use_online_draft: bool
    use_offline_final: bool
    partial_preview: bool


def resolve_asr_mode(requested_mode: str, backend: str, model_name: str) -> AsrModeConfig:
    mode = (requested_mode or "twopass").strip().lower()
    is_parakeet_model = "parakeet-tdt" in model_name

    if mode == "streaming":
        if not is_parakeet_model:
            raise ValueError("streaming mode requires a streaming-capable Sherpa model")
        return AsrModeConfig(mode, True, False, True)

    if mode == "twopass":
        return AsrModeConfig(mode, True, True, True)

    if mode == "batch":
        return AsrModeConfig(mode, False, True, False)

    raise ValueError(f"Unsupported ASR mode: {requested_mode}")
```

- [ ] **Step 4: Wire startup export and mode display**

Update `start.sh` so it exports and prints the selected mode:

```bash
export ASR_MODE="${ASR_MODE:-twopass}"

echo "  ASR Mode : $ASR_MODE"

BACKEND="$BACKEND" ASR_MODE="$ASR_MODE" "$VENV_PYTHON" src/brain.py > logs/brain.log 2>&1 &
BACKEND="$BACKEND" ASR_MODE="$ASR_MODE" "$VENV_PYTHON" src/ear.py
```

Also update `src/brain.py` startup to read:

```python
ASR_MODE = os.environ.get("ASR_MODE", "twopass").lower().strip()
```

- [ ] **Step 5: Re-run the targeted tests**

Run:
```bash
pytest tests/test_asr_mode.py tests/test_brain.py -v
```

Expected:
- `tests/test_asr_mode.py` passes.
- Existing `tests/test_brain.py` may still fail until later tasks wire the new session flow.

- [ ] **Step 6: Commit the mode-selection groundwork**

Run:
```bash
git add start.sh src/asr_mode.py src/brain.py tests/test_asr_mode.py
git commit -m "feat: add startup ASR mode selection"
```

---

### Task 2: Add a Sherpa online draft recognizer backend

**Files:**
- Create: `src/backend_sherpa_online.py`
- Modify: `src/backend_parakeet.py`
- Test: `tests/test_backend_sherpa_online.py`

- [ ] **Step 1: Write the failing online-backend tests**

Create `tests/test_backend_sherpa_online.py`:

```python
from unittest.mock import MagicMock, patch

import backend_sherpa_online


def test_load_model_uses_online_recognizer_factory():
    with patch("backend_sherpa_online.sherpa_onnx.OnlineRecognizer") as mock_cls:
        mock_cls.from_transducer.return_value = MagicMock()
        model = backend_sherpa_online.load_model("/tmp/draft-model", num_threads=2)

    assert model is mock_cls.from_transducer.return_value
    assert mock_cls.from_transducer.called


def test_read_partial_returns_string_result():
    recognizer = MagicMock()
    stream = MagicMock()
    recognizer.is_ready.return_value = False
    recognizer.get_result.return_value = "draft text"

    text = backend_sherpa_online.read_partial(recognizer, stream, [0.1, 0.2, 0.3])
    assert text == "draft text"
```

- [ ] **Step 2: Run the backend tests and confirm they fail**

Run:
```bash
pytest tests/test_backend_sherpa_online.py -v
```

Expected:
- The run fails because `backend_sherpa_online.py` does not exist yet.

- [ ] **Step 3: Implement the online backend wrapper**

Create `src/backend_sherpa_online.py`:

```python
import os
import numpy as np
import sherpa_onnx


def load_model(model_dir: str, num_threads: int = 2):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder.int8.onnx",
        decoder=f"{model_dir}/decoder.int8.onnx",
        joiner=f"{model_dir}/joiner.int8.onnx",
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        decoding_method="greedy_search",
        model_type="zipformer",
        provider="cpu",
    )


def create_stream(model):
    return model.create_stream()


def read_partial(model, stream, samples) -> str:
    audio = np.asarray(samples, dtype=np.float32)
    stream.accept_waveform(16000, audio)
    while model.is_ready(stream):
        model.decode_stream(stream)
    return model.get_result(stream).strip()
```

- [ ] **Step 4: Keep the offline Parakeet backend unchanged for final-pass transcription**

Verify `src/backend_parakeet.py` still exposes the current offline contract:

```python
def transcribe(model, audio_array):
    stream = model.create_stream()
    stream.accept_waveform(16000, audio_array)
    model.decode_stream(stream)
    return stream.result.text.strip()
```

Expected:
- The online and offline Sherpa paths stay separate.
- The existing Parakeet final transcription path remains valid for pass 2.

- [ ] **Step 5: Run the Sherpa backend tests**

Run:
```bash
pytest tests/test_backend_sherpa_online.py tests/test_backend.py -v
```

Expected:
- The new online backend tests pass.
- Existing backend tests still pass for the offline/faster-whisper path.

- [ ] **Step 6: Commit the online draft backend**

Run:
```bash
git add src/backend_sherpa_online.py src/backend_parakeet.py tests/test_backend_sherpa_online.py
git commit -m "feat: add sherpa online draft backend"
```

---

### Task 3: Refactor Brain into mode-aware streaming and two-pass sessions

**Files:**
- Create: `src/asr_session.py`
- Modify: `src/brain.py`
- Modify: `src/hud.py`
- Test: `tests/test_brain.py`

- [ ] **Step 1: Write the failing Brain session tests**

Add to `tests/test_brain.py`:

```python
def test_handle_connection_in_twopass_sends_partial_and_final(sample_audio_bytes):
    mock_online_backend = MagicMock()
    mock_online_backend.read_partial.return_value = "draft hello"
    mock_online_backend.create_stream.return_value = MagicMock()

    mock_offline_backend = MagicMock()
    mock_offline_backend.transcribe.return_value = "final hello"

    brain.backend_info["backend"] = mock_offline_backend
    brain.backend_info["model"] = MagicMock()
    brain.vad_engine = None

    conn = MockConn(sample_audio_bytes)

    with patch("brain.send_hud") as mock_hud, patch("brain.paste_instantly") as mock_paste:
        with patch("brain.create_asr_session") as mock_factory:
            session = mock_factory.return_value
            session.run_connection.side_effect = lambda _conn: ("draft hello", "final hello")
            brain.handle_connection(conn)

    mock_hud.assert_any_call("partial:draft hello")
    mock_paste.assert_called_once_with("final hello ")


def test_handle_connection_in_streaming_mode_skips_offline_final(sample_audio_bytes):
    conn = MockConn(sample_audio_bytes)

    with patch("brain.send_hud"), patch("brain.paste_instantly") as mock_paste:
        with patch("brain.create_asr_session") as mock_factory:
            session = mock_factory.return_value
            session.run_connection.side_effect = lambda _conn: ("draft only", "draft only")
            brain.handle_connection(conn)

    mock_paste.assert_called_once_with("draft only ")
```

- [ ] **Step 2: Run the Brain tests and confirm they fail**

Run:
```bash
pytest tests/test_brain.py -v
```

Expected:
- The new tests fail because `create_asr_session()` and the new HUD partial protocol do not exist yet.

- [ ] **Step 3: Implement a session object that owns chunk-by-chunk decoding**

Create `src/asr_session.py`:

```python
class AsrSession:
    def __init__(self, mode_config, online_backend, online_model, offline_backend, offline_model, send_hud):
        self.mode_config = mode_config
        self.online_backend = online_backend
        self.online_model = online_model
        self.offline_backend = offline_backend
        self.offline_model = offline_model
        self.send_hud = send_hud
        self.partial_text = ""
        self.audio_chunks = []

    def on_chunk(self, pcm_bytes: bytes):
        self.audio_chunks.append(pcm_bytes)

    def on_partial(self, text: str):
        if text and text != self.partial_text:
            self.partial_text = text
            self.send_hud(f"partial:{text}")

    def finalize(self):
        raw = b"".join(self.audio_chunks)
        final_text = self.partial_text
        if self.mode_config.use_offline_final:
            final_text = self.offline_backend.transcribe(self.offline_model, decode_pcm(raw))
        self.send_hud("clear")
        return self.partial_text, final_text.strip()
```

- [ ] **Step 4: Update `brain.py` to use the session object**

Refactor `handle_connection()` so it delegates chunk handling and finalization:

```python
def create_asr_session():
    mode_config = resolve_asr_mode(ASR_MODE, BACKEND, current_model_name())
    return AsrSession(
        mode_config=mode_config,
        online_backend=online_backend_info["backend"],
        online_model=online_backend_info["model"],
        offline_backend=backend_info["backend"],
        offline_model=backend_info["model"],
        send_hud=send_hud,
    )
```

Then change the receive loop shape:

```python
session = create_asr_session()

while True:
    data = conn.recv(32768)
    if not data:
        break
    session.on_chunk(data)

partial_text, final_text = session.finalize()
```

- [ ] **Step 5: Extend the HUD command protocol for draft text**

Update `src/hud.py` to recognize partial text commands in `_on_command()`:

```python
elif c.startswith("partial:"):
    self.show_partial_text(c.split(":", 1)[1])
elif c == "clear":
    self.clear_partial_text()
```

Add minimal widget state:

```python
self._partial_text = ""

def show_partial_text(self, text: str):
    self._partial_text = text
    self.update()

def clear_partial_text(self):
    self._partial_text = ""
    self.update()
```

- [ ] **Step 6: Run the Brain and HUD-adjacent tests**

Run:
```bash
pytest tests/test_brain.py tests/test_integration.py -v
```

Expected:
- Brain tests pass with partial-preview and final-commit behavior.
- Integration tests may still need one more update in the next task for end-to-end mode coverage.

- [ ] **Step 7: Commit the mode-aware session refactor**

Run:
```bash
git add src/asr_session.py src/brain.py src/hud.py tests/test_brain.py
git commit -m "feat: add streaming and twopass session flow"
```

---

### Task 4: Update integration tests and startup/help text for the new workflow

**Files:**
- Modify: `tests/test_integration.py`
- Modify: `README.md`
- Modify: `start.sh`

- [ ] **Step 1: Update the integration test to assert draft-preview behavior**

Edit `tests/test_integration.py` so it captures HUD partial updates as well as final paste:

```python
def test_socket_communication_emits_partial_then_final(sample_audio_bytes):
    partial_messages = []

    with patch("brain.send_hud", side_effect=partial_messages.append), \
         patch("brain.paste_instantly") as mock_paste:
        with patch("brain.create_asr_session") as mock_factory:
            session = mock_factory.return_value
            session.run_connection.side_effect = lambda _conn: ("draft text", "final text")

            parent_sock, child_sock = socket.socketpair()
            thread = threading.Thread(target=brain.handle_connection, args=(child_sock,), daemon=True)
            thread.start()

            parent_sock.sendall(sample_audio_bytes)
            parent_sock.shutdown(socket.SHUT_WR)
            parent_sock.close()
            thread.join(timeout=5.0)

    assert "partial:draft text" in partial_messages
    mock_paste.assert_called_once_with("final text ")
```

- [ ] **Step 2: Update startup text and README mode documentation**

Add mode-specific usage to `start.sh` and `README.md`:

```bash
ASR_MODE=streaming ./start.sh
ASR_MODE=twopass ./start.sh
ASR_MODE=batch ./start.sh
```

Document these rules:
- `streaming` uses Sherpa draft text only and commits once.
- `twopass` uses Sherpa draft plus the selected offline final backend.
- Runtime number keys still switch final models, not the ASR mode.

- [ ] **Step 3: Run the targeted regression suite**

Run:
```bash
pytest tests/test_asr_mode.py tests/test_backend.py tests/test_backend_sherpa_online.py tests/test_brain.py tests/test_integration.py -v
```

Expected:
- All targeted tests pass.
- The suite proves startup mode selection, online draft decoding, Brain session orchestration, and end-to-end socket behavior.

- [ ] **Step 4: Run manual smoke tests**

Run:
```bash
ASR_MODE=twopass ./start.sh
```

Then verify:
- The HUD shows draft text while speaking.
- Releasing the key or mouse commits one final transcription.
- No partial text is typed into the target app.

Run:
```bash
ASR_MODE=streaming ./start.sh
```

Then verify:
- A streaming-capable model works without invoking the offline final pass.
- Switching to a non-streaming final model is rejected or clearly warned.

- [ ] **Step 5: Commit docs and integration coverage**

Run:
```bash
git add README.md start.sh tests/test_integration.py
git commit -m "docs: describe ASR modes and verify integration flow"
```

---

## Self-Review Checklist

- [ ] The plan keeps `ear.py` mostly as transport and moves complexity into Brain/HUD/backend helpers.
- [ ] The plan does not rely on live deletion/retyping of partial text in arbitrary apps.
- [ ] The plan keeps Sherpa online draft decoding separate from the existing offline Parakeet final backend.
- [ ] The plan allows `twopass` to use an offline final backend while reserving `streaming` for streaming-capable models.
- [ ] The plan includes both targeted automated tests and manual smoke tests before completion.
