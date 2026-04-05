# Silero VAD Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace overlap-based draft streaming with Silero VAD-gated utterance capture so `ear.py` buffers complete speech segments, `brain.py` transcribes once per utterance, and the HUD stays silent except for menu-bar status.

**Architecture:** `ear.py` owns microphone capture, Silero VAD scoring, utterance buffering, and the stop condition. `brain.py` becomes final-only: it receives a completed utterance, transcribes it once, pastes the final text, and exits cleanly. `hud.py` keeps only status states (`listen`, `thinking`, `process`, `done`, `hide`) and ignores transcript payloads. A small helper module isolates the VAD state machine so the speech-boundary logic can be tested without PyAudio or macOS UI.

**Tech Stack:** Python 3.12, NumPy, PyAudio, onnxruntime, the existing local Silero VAD ONNX model, the existing Parakeet/Faster-Whisper/OpenVINO backends, PySide6/AppKit.

---

## File Map

- Create: `src/vad_segmenter.py`
- Modify: `src/ear.py`
- Modify: `src/brain.py`
- Modify: `src/hud.py`
- Modify: `tests/test_integration.py`
- Modify: `tests/test_brain.py`
- Modify: `tests/test_hud_menu_bar.py`
- Create: `tests/test_vad_segmenter.py`
- Create: `streaming.md`
- Delete: `stramning.md`

The new `src/vad_segmenter.py` module should also own the current Silero ONNX wrapper that lives in `brain.py` today, so `ear.py` can import the VAD engine without depending on the transcription server.

Add this constant near the other `ear.py` settings:

```python
VAD_SILENCE_TIMEOUT = 0.4   # seconds of silence before utterance closes
```

## Design Choices

The recommended approach is to gate speech in `ear.py` rather than in `brain.py`.

- `ear.py` already owns the microphone stream and the user-facing recording lifecycle.
- `brain.py` already has the final paste step and model switching logic.
- Keeping VAD on the capture side lets us remove overlap, draft stitching, and per-chunk decode from the inference side.
- This avoids a multi-segment socket protocol and keeps the pipeline easy to reason about.

Alternative approaches were considered:

- VAD in `brain.py` with repeated draft decodes: rejected because it keeps the overlap/draft complexity and still risks repeated text.
- Fixed-size chunk streaming with deduplication: rejected because it keeps cutting words at arbitrary boundaries.
- VAD in `ear.py` with one final utterance per session: chosen because it matches the current hold-to-record flow and the user’s requirement for silent processing.

---

### Task 1: Add a reusable Silero VAD segmenter and wire `ear.py` to buffer whole utterances

**Files:**
- Create: `src/vad_segmenter.py`
- Modify: `src/ear.py`
- Test: `tests/test_vad_segmenter.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
from src.vad_segmenter import SileroUtteranceGate


class FakeVAD:
    def __init__(self, scores):
        self._scores = iter(scores)

    def is_speech(self, _audio_chunk, sample_rate=16000):
        return next(self._scores)


def test_gate_flushes_only_after_voice_then_silence():
    gate = SileroUtteranceGate(
        vad_engine=FakeVAD([0.95, 0.91, 0.12, 0.08]),
        voice_threshold=0.5,
        silence_timeout_s=0.4,
        min_utterance_bytes=16,
    )

    gate.push(b"\x01\x00" * 8, now=0.0)
    gate.push(b"\x02\x00" * 8, now=0.1)

    assert gate.should_finalize(now=0.2) is False
    assert gate.should_finalize(now=0.6) is True
    assert gate.flush() == (b"\x01\x00" * 8) + (b"\x02\x00" * 8)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_vad_segmenter.py::test_gate_flushes_only_after_voice_then_silence -v
```

Expected:

```text
FAIL: ModuleNotFoundError or attribute error for SileroUtteranceGate
```

- [ ] **Step 3: Write the minimal implementation**

```python
class SileroUtteranceGate:
    def __init__(
        self,
        vad_engine,
        *,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        voice_threshold: float = 0.5,
        silence_timeout_s: float = 0.4,
        min_utterance_bytes: int = 8000,
    ):
        self.vad_engine = vad_engine
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.voice_threshold = voice_threshold
        self.silence_timeout_s = silence_timeout_s
        self.min_utterance_bytes = min_utterance_bytes
        self._buffer = bytearray()
        self._speech_started = False
        self._last_voice_time = 0.0

    def push(self, pcm16_bytes: bytes, now: float) -> None:
        self._buffer.extend(pcm16_bytes)
        if len(pcm16_bytes) < 1024:
            return
        frame = np.frombuffer(pcm16_bytes[-1024:], dtype=np.int16).astype(np.float32) / 32768.0
        score = self.vad_engine.is_speech(frame, sample_rate=self.sample_rate)
        if score > self.voice_threshold:
            self._speech_started = True
            self._last_voice_time = now

    def should_finalize(self, now: float) -> bool:
        return (
            self._speech_started
            and len(self._buffer) >= self.min_utterance_bytes
            and (now - self._last_voice_time) >= self.silence_timeout_s
        )

    def flush(self) -> bytes:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self._speech_started = False
        self._last_voice_time = 0.0
        return audio
```

Wire it into `ear.py` with a single utterance buffer and an on-demand Brain send path:

```python
self._vad_engine = SileroVAD(VAD_MODEL_PATH)
self._utterance_gate = SileroUtteranceGate(
    vad_engine=self._vad_engine,
    silence_timeout_s=VAD_SILENCE_TIMEOUT,
)
self._finalize_requested = False

def _audio_callback(self, in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    boosted = (audio_data.astype(np.float32) * self.gain_multiplier).clip(-32768, 32767).astype(np.int16)
    chunk_bytes = boosted.tobytes()
    self.last_rms = get_rms(chunk_bytes)
    self._utterance_gate.push(chunk_bytes, now=time.time())
    return (None, pyaudio.paContinue)

def _record_loop_tick(self):
    with self._lock:
        finalize_requested = self._finalize_requested
    if finalize_requested and self._utterance_gate.should_finalize(time.time()):
        self._stop_and_send()

def _stop_and_send(self):
    with self._lock:
        if not self.is_recording:
            return
        self.is_recording = False
        self._finalize_requested = False
    utterance = self._utterance_gate.flush()
    if utterance:
        threading.Thread(target=self._send_hud, args=("process",), daemon=True).start()
        self._send_utterance_to_brain(utterance)
```

Replace the persistent Brain socket with a one-shot sender:

```python
def _send_utterance_to_brain(self, utterance_bytes: bytes):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)
    client.sendall(utterance_bytes)
    client.shutdown(socket.SHUT_WR)
    client.close()
```

On button release, do not transcribe immediately. Mark the finalize request and let the VAD gate close the utterance only after silence has been stable long enough:

```python
def on_release(self, key):
    ...
    with self._lock:
        if not self.is_recording:
            return
        self._finalize_requested = True
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
pytest tests/test_vad_segmenter.py::test_gate_flushes_only_after_voice_then_silence -v
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/vad_segmenter.py src/ear.py tests/test_vad_segmenter.py tests/test_integration.py
git commit -m "feat: gate utterances with silero vad"
```

---

### Task 2: Remove overlap, draft stitching, and repeated decode from `brain.py`

**Files:**
- Modify: `src/brain.py`
- Modify: `tests/test_brain.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_handle_connection_transcribes_one_full_utterance(sample_audio_bytes):
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_backend.transcribe.return_value = "hello world"

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    conn = MockConn(sample_audio_bytes)

    with patch("brain.send_hud") as mock_hud, patch("brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_called_once()
    args, _ = mock_backend.transcribe.call_args
    assert args[0] == mock_model
    assert isinstance(args[1], np.ndarray)
    assert args[1].dtype == np.float32
    mock_paste.assert_called_once_with("hello world ")
    assert [call.args[0] for call in mock_hud.call_args_list] == ["done"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_brain.py::test_handle_connection_transcribes_one_full_utterance -v
```

Expected:

```text
FAIL: still using draft/overlap state or still emitting transcript payloads
```

- [ ] **Step 3: Write the minimal implementation**

```python
def handle_connection(conn):
    t_connect = time.perf_counter()
    command_probe = bytearray()
    pending_audio = bytearray()
    total_bytes = 0

    with backend_lock:
        backend = backend_info["backend"]
        model = backend_info["model"]

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
            if len(command_probe) < 256:
                command_probe.extend(data[: 256 - len(command_probe)])
            pending_audio.extend(data)
    finally:
        conn.close()

    if total_bytes == 0:
        return

    try:
        text = bytes(command_probe).decode("utf-8").strip()
        if text.startswith("CMD_SWITCH_MODEL:"):
            new_model = text.split(":", 1)[1]
            print(f"\n[Brain] 🔄 Switch model: {new_model}")
            sys.stdout.flush()
            with backend_lock:
                import gc
                backend_info["model"] = None
                gc.collect()
                nb, nm = load_backend(new_model)
                backend_info["backend"] = nb
                backend_info["model"] = nm
                print(f"[Brain] ✅ Switched to {new_model}")
                sys.stdout.flush()
            return
    except UnicodeDecodeError:
        pass

    raw_audio = bytes(pending_audio)
    if backend is None or model is None:
        send_hud("hide")
        return

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    text = backend.transcribe(model, audio).strip()
    if not text:
        send_hud("hide")
        return

    print(f"[Brain] 📝 [utterance | {time.perf_counter() - t_connect:.2f}s] → \"{text}\"")
    send_hud("done")
    paste_instantly(text + " ")
```

This step also deletes the old overlap-only state from `brain.py`:

```python
stitch_draft(...)
prev_tail_audio
last_draft_text
MIN_DECODE_SECONDS
OVERLAP_SECONDS
draft:<text>
final:<text>
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
pytest tests/test_brain.py::test_handle_connection_transcribes_one_full_utterance -v
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/brain.py tests/test_brain.py tests/test_integration.py
git commit -m "feat: transcribe full utterances in brain"
```

---

### Task 3: Strip transcript payload handling out of `hud.py`

**Files:**
- Modify: `src/hud.py`
- Modify: `tests/test_hud_menu_bar.py`

- [ ] **Step 1: Write the failing test**

```python
def test_hud_ignores_transcript_payloads_and_keeps_status_only():
    hud_widget = hud.MenuBarWaveformController.__new__(hud.MenuBarWaveformController)
    hud_widget._state = hud.HIDDEN
    hud_widget._request_menu_bar_view_redraw = lambda: None
    hud_widget.show_listening = lambda: None
    hud_widget.show_done = lambda: None
    hud_widget.hide_hud = lambda: None
    hud_widget.show_thinking = lambda: None
    hud_widget.show_processing = lambda: None

    hud.MenuBarWaveformController._on_command(hud_widget, "draft:Hello World")
    hud.MenuBarWaveformController._on_command(hud_widget, "final:Hello World")

    assert hud_widget._state == hud.HIDDEN
    assert not hasattr(hud_widget, "_draft_text")
    assert not hasattr(hud_widget, "_final_text")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_hud_menu_bar.py::test_hud_ignores_transcript_payloads_and_keeps_status_only -v
```

Expected:

```text
FAIL: _draft_text / _final_text still exist or draft/final handling is still active
```

- [ ] **Step 3: Write the minimal implementation**

```python
def _on_command(self, cmd):
    c = cmd.strip().lower()
    if c == "listen":
        self.show_listening()
    elif c == "thinking":
        self.show_thinking()
    elif c == "process":
        self.show_processing()
    elif c == "done":
        self.show_done()
    elif c == "hide":
        self.hide_hud()
```

Delete the transcript helper and state:

```python
normalize_transcript_text
_draft_text
_final_text
"draft:<text>"
"final:<text>"
```

Update the module docstring so IPC only lists status commands.

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
pytest tests/test_hud_menu_bar.py::test_hud_ignores_transcript_payloads_and_keeps_status_only -v
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/hud.py tests/test_hud_menu_bar.py
git commit -m "feat: remove transcript payload handling from hud"
```

---

### Task 4: Rewrite the streaming note so the repo documents the VAD-only flow

**Files:**
- Create: `streaming.md`
- Delete: `stramning.md`

- [ ] **Step 1: Write the failing documentation check**

```markdown
# Streaming Notes

The app uses Silero VAD to detect speech boundaries.
It does not rely on overlap-based draft stitching.
Audio is buffered until silence, then one complete utterance is sent to Brain.
```

If the repo keeps a doc check, verify that the old overlap language no longer appears:

```bash
rg -n "overlap|draft|stitch|chunk boundaries" streaming.md
```

Expected:

```text
no matches
```

- [ ] **Step 2: Run the check and verify it fails before the rewrite**

Run:

```bash
rg -n "overlap|draft|stitch|chunk boundaries" stramning.md
```

Expected:

```text
matches, because the old note still references the previous streaming strategy
```

- [ ] **Step 3: Write the minimal documentation rewrite**

```markdown
# Streaming Notes

Silero VAD is the speech boundary detector for this project.

Flow:
1. Capture microphone audio continuously.
2. Feed short frames into Silero VAD.
3. Buffer audio while speech is active.
4. When silence lasts long enough, finalize the utterance.
5. Send the complete utterance to `brain.py`.
6. Transcribe once, paste once, and reset.

There is no overlap-based draft stitching in the live path.
The HUD stays menu-bar only, and the detailed logs stay available through `DEBUG_STREAMING=1 ./start.sh`.
```

- [ ] **Step 4: Run the check and verify it passes**

Run:

```bash
rg -n "overlap|draft|stitch|chunk boundaries" streaming.md
```

Expected:

```text
no matches
```

- [ ] **Step 5: Commit**

```bash
git add streaming.md stramning.md
git commit -m "docs: rewrite streaming notes for silero vad"
```

---

## Verification Checklist

Before calling this done, run the focused verification commands:

```bash
pytest tests/test_vad_segmenter.py tests/test_brain.py tests/test_hud_menu_bar.py tests/test_integration.py -q
bash -n start.sh
```

Expected:

```text
all focused tests pass
start.sh parses cleanly
```

Manual smoke test:

```bash
DEBUG_STREAMING=1 ./start.sh
```

Expected:

```text
terminal shows Brain logs, no transcript box appears, and the final text is pasted only after the utterance closes
```

## Self-Review

- Spec coverage: Task 1 covers VAD gating and utterance buffering in `ear.py`; Task 2 removes overlap and draft decoding from `brain.py`; Task 3 removes transcript payload handling from `hud.py`; Task 4 rewrites the streaming note so the repo documents the new flow.
- Placeholder scan: no `TBD`, `TODO`, or “implement later” text remains in the plan.
- Type consistency: `SileroUtteranceGate`, `push()`, `should_finalize()`, and `flush()` are used consistently across the plan, and the HUD contract is status-only.
