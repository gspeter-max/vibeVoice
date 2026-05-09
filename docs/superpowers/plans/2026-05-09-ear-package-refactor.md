# Ear Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent** :
- This plan refactors `src/audio/ear.py`, which is currently a shallow catch-all module.
- The current file mixes microphone runtime, input control, streaming session state, IPC protocol, IPC transport, HUD client code, platform helpers, operator menu code, and startup wiring.
- The goal is to replace that shallow module with a deeper package shape that has clearer seams and better locality.
- The target design is:
  - `src/input` owns trigger semantics.
  - `src/audio/ear/*` owns microphone runtime and audio-local helpers.
  - `src/streaming` owns chunk/session state and chunk split policy.
  - `src/ipc` owns wire protocol and socket transport.
  - `src/ui` owns HUD server plus HUD client protocol helpers.
- The fresh agent must preserve public import compatibility for `src.audio.ear` during migration because tests and runtime code still import from it.
- The fresh agent must not assume anything. If any import path, patch target, or runtime invariant is unclear, ask the user before changing it.
- Files to read first to understand current behavior:
  - `/Users/apple/.codex/AGENTS.md` - repo-specific communication and implementation rules.
  - `src/audio/ear.py` - current large runtime and duplicated ownership.
  - `src/input/hotkeys.py` - existing input seam that should become the single input owner.
  - `src/ipc/messenger.py` - existing protocol formatting/parsing helpers.
  - `src/streaming/streaming_shared_logic.py` - existing stateless chunking policy.
  - `src/streaming/session.py` - partial session-state module that should be deepened or complemented.
  - `src/backend/brain.py` - receiver side that currently still understands raw protocol details.
  - `src/ui/hud.py` - HUD server and state vocabulary.
  - `src/utils/wizard_tui.py` - imports `get_active_models` from `src.audio.ear`.
  - `tests/test_ear_hold_state.py` - highest-value behavior lock for Ear runtime semantics.
  - `tests/test_ear_recording_mode.py` - model-list and terminal-menu compatibility.
  - `tests/test_ear_nemotron.py` - special Nemotron heartbeat rule.
  - `tests/test_ear_fft.py` - frequency-analysis behavior.
  - `tests/test_input_hotkeys.py` - current input seam behavior.
  - `tests/test_ipc_messenger.py` - exact protocol header assertions.

**Architecture:** The refactor keeps one compatibility façade at `src/audio/ear.py`, but moves real ownership into deeper modules with smaller interfaces. The runtime will call a streaming capture-session module for state transitions, an IPC client module for socket transport, and a HUD client module for UI signaling. The input seam remains `InputTrigger`. This plan is intentionally incremental so that each step is testable and reversible.

**Important Rule to follow :**
- **CRITICAL:** add detailed docs in functions and explain the code and logic in comments.
- **CRITICAL:** make code function names and variable names clear and easy to understand instead of short and confusing names.
- Explain like a fresher.
- Write docs in a step-by-step simple style.
- Make the docs in function and file headers human-readable and literal.

---

## File Structure

### Current relevant file tree

```text
src/
  audio/
    __init__.py
    ear.py
    vad_segmenter.py
  input/
    __init__.py
    hotkeys.py
  ipc/
    __init__.py
    messenger.py
  streaming/
    __init__.py
    nemotron.py
    session.py
    streaming_shared_logic.py
  backend/
    brain.py
    state.py
  ui/
    __init__.py
    hud.py
  utils/
    wizard_tui.py
```

### Target file tree after this plan

```text
src/
  audio/
    __init__.py
    ear.py                  # compatibility facade only
    vad_segmenter.py
    ear_runtime/
      __init__.py
      controller.py         # thin Ear runtime
      analysis.py           # RMS, FFT, gain helpers
      devices.py            # mic selection and stream/device helpers
      platform.py           # sound effect and macOS voice isolation
      menu.py               # terminal menu and self-test
      runtime.py            # start_ear and composition root
  input/
    __init__.py
    hotkeys.py
  ipc/
    __init__.py
    protocol.py            # canonical message builders/parsers
    client.py              # one-shot send + raw stream transport
    messenger.py           # compatibility wrapper/re-export layer
  streaming/
    __init__.py
    capture_session.py     # ear-side chunk/session state
    nemotron.py
    session.py
    streaming_shared_logic.py
  ui/
    __init__.py
    hud.py
    hud_client.py          # Ear-facing HUD client helpers
```

### Module responsibilities

- `src/audio/ear.py`
  - Keep import compatibility for `Ear`, `start_ear`, `get_active_models`, `TerminalMenu`, and temporary patch points used by tests.
- `src/audio/ear_runtime/controller.py`
  - Own the thin `Ear` runtime class.
  - Own microphone callback orchestration, record loop, and mode branching.
  - Do not own raw protocol string formatting.
- `src/audio/ear_runtime/analysis.py`
  - Own `get_rms`, gain boosting, and frequency-band helpers.
- `src/audio/ear_runtime/devices.py`
  - Own `select_mic` and microphone/device setup helpers.
- `src/audio/ear_runtime/platform.py`
  - Own start sound loading/playback and macOS voice isolation helpers.
- `src/audio/ear_runtime/menu.py`
  - Own `TerminalMenu`, `run_self_test`, and model-switch operator logic.
- `src/audio/ear_runtime/runtime.py`
  - Own `start_ear()` and object wiring.
- `src/ipc/protocol.py`
  - Own message shapes and exact wire-format encoding and parsing.
- `src/ipc/client.py`
  - Own one-shot send helpers and raw no-streaming socket lifecycle.
- `src/ipc/messenger.py`
  - Re-export protocol helpers so current tests keep passing during migration.
- `src/streaming/capture_session.py`
  - Own `_current_session_id`, `_recording_index`, `_chunk_seq`, `_chunk_started_at`, overlap state, flush state transitions, and commit decisions.
- `src/ui/hud_client.py`
  - Own Ear-facing HUD command send and volume sender thread.

## Test design audit

### Existing test style in this feature area

- `pytest` is the test framework.
- Tests use `unittest.mock.patch`, `MagicMock`, and `monkeypatch`.
- Tests prefer direct unit-level assertions over large fixtures.
- Tests often monkeypatch import targets such as `src.audio.ear.SileroVAD`, `src.audio.ear.socket.socket`, and `src.audio.ear.RECORDING_MODE`.
- Several tests assert exact protocol bytes and exact public import paths.

### Five most complex edge cases and failure states to preserve

1. Session lifetime invariants:
   - session id is created once for app lifetime.
   - recording index increments only on commit.
   - chunk sequence resets only after final stop.
2. Audio-path invariants:
   - VAD receives raw audio chunk plus boosted analysis chunk.
   - Brain receives boosted audio.
   - final stop skips overlap prepend.
3. Mode split invariants:
   - `no_streaming` and `silence_streaming` have materially different transport paths.
   - `no_streaming` must keep raw open-socket streaming behavior.
4. Input semantics:
   - toggle mode, hold mode, and mouse hold threshold behavior must not drift.
   - duplicate old input handlers should be removed only after compatibility coverage is preserved.
5. Nemotron special-case chunk policy:
   - `nemotron-streaming-0.6b` must keep the 1.12 second forced heartbeat path.

### Scope check

This plan produces working, testable software on its own. It intentionally does not redesign backend session storage, replace VAD behavior, or change external command semantics. It only removes duplicate ownership and improves module locality.

---

### Task 1 : Read instruction and behavior lock files

**Files:**
- Read: `/Users/apple/.gemini/GEMINI.md`
- Read: `src/audio/ear.py`
- Read: `src/input/hotkeys.py`
- Read: `src/ipc/messenger.py`
- Read: `src/streaming/streaming_shared_logic.py`
- Read: `src/streaming/session.py`
- Read: `src/backend/brain.py`
- Read: `src/ui/hud.py`
- Read: `src/utils/wizard_tui.py`
- Read: `tests/test_ear_hold_state.py`
- Read: `tests/test_ear_recording_mode.py`
- Read: `tests/test_ear_nemotron.py`
- Read: `tests/test_ear_fft.py`
- Read: `tests/test_ipc_messenger.py`
- Read: `tests/test_input_hotkeys.py`

- [ ] **Step 1: Read the repo instructions and the Ear/IPC/input tests**

Run:

```bash
sed -n '1,260p' /Users/apple/.codex/AGENTS.md
sed -n '1,220p' tests/test_ipc_messenger.py
sed -n '1,220p' tests/test_input_hotkeys.py
sed -n '1,220p' tests/test_ear_recording_mode.py
sed -n '1,260p' tests/test_ear_nemotron.py
sed -n '1,620p' tests/test_ear_hold_state.py
```

Expected:
- You have a verified list of compatibility seams and patch targets before editing.

- [ ] **Step 2: Write a short scratch checklist of invariants**

Document in your local notes:
- `src.audio.ear` import compatibility symbols
- test patch targets that must still exist during migration
- exact message headers asserted by tests
- Nemotron heartbeat rule

- [ ] **Step 3: Commit nothing yet**

Expected:
- No code changes in this task.

### Task 2: Introduce protocol authority in `src/ipc`

**Files:**
- Create: `src/ipc/protocol.py`
- Modify: `src/ipc/messenger.py`
- Test: `tests/test_ipc_messenger.py`

- [ ] **Step 1: Write the failing test**

Add tests that import through the new module while preserving old compatibility:

```python
from src.ipc.protocol import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
    format_switch_model_message,
    parse_incoming_message,
)

def test_protocol_module_matches_existing_messenger_contract():
    assert format_audio_chunk_message("session123", 0, 5, b"audio") == b"CMD_AUDIO_CHUNK:session123:0:5\n\naudio"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ipc_messenger.py -q
```

Expected:
- FAIL because `src.ipc.protocol` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `src/ipc/protocol.py` with the formatting and parsing helpers now living in `src/ipc/messenger.py`.

Implementation notes:
- Copy behavior exactly.
- Keep function names identical.
- Keep docs literal and detailed.
- Do not change protocol bytes.

Then update `src/ipc/messenger.py` to re-export:

```python
from src.ipc.protocol import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
    format_switch_model_message,
    parse_incoming_message,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_ipc_messenger.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/ipc/protocol.py src/ipc/messenger.py tests/test_ipc_messenger.py
git commit -m "refactor: add canonical ipc protocol module"
```

### Task 3: Introduce socket transport adapters in `src/ipc/client.py`

**Files:**
- Create: `src/ipc/client.py`
- Modify: `src/audio/ear.py`
- Test: `tests/test_ear_hold_state.py`

- [ ] **Step 1: Write failing tests for transport extraction**

Add targeted tests for transport helpers such as:

```python
def test_send_message_to_brain_returns_false_when_message_is_empty():
    from src.ipc.client import send_message_to_brain
    assert send_message_to_brain(b"") is False
```

Also add a focused test for raw stream lifecycle helpers if those become public functions.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ipc_messenger.py tests/test_ear_hold_state.py -q
```

Expected:
- FAIL because `src.ipc.client` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `src/ipc/client.py` with:
- one-shot send helper
- raw stream open helper
- raw stream send helper
- raw stream close helper

Move logic out of these old `Ear` methods:
- `_send_session_event_to_brain`
- `_send_audio_chunk_to_brain`
- `_commit_recording_session`
- `_open_brain_stream`
- `_stream_chunk_to_brain`
- `_close_brain_stream`

Implementation rule:
- do not remove the old `Ear` methods yet
- rewrite them as thin wrappers that call `src.ipc.client`
- preserve test patch targets in `src.audio.ear`

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run pytest tests/test_ipc_messenger.py tests/test_ear_hold_state.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/ipc/client.py src/audio/ear.py tests/test_ear_hold_state.py tests/test_ipc_messenger.py
git commit -m "refactor: extract ear ipc transport client"
```

### Task 4: Extract audio-local helpers into `src/audio/ear_runtime/analysis.py`, `devices.py`, and `platform.py`

**Files:**
- Create: `src/audio/ear_runtime/__init__.py`
- Create: `src/audio/ear_runtime/analysis.py`
- Create: `src/audio/ear_runtime/devices.py`
- Create: `src/audio/ear_runtime/platform.py`
- Modify: `src/audio/ear.py`
- Modify: `src/utils/wizard_tui.py`
- Test: `tests/test_ear_fft.py`
- Test: `tests/test_ear_recording_mode.py`
- Test: `tests/test_ear_hold_state.py`

- [ ] **Step 1: Write failing tests for moved pure helpers**

Add tests that exercise:
- `get_rms` from `src.audio.ear_runtime.analysis`
- frequency-band helper from `src.audio.ear_runtime.analysis`
- `select_mic` from `src.audio.ear_runtime.devices`

Keep existing compatibility imports through `src.audio.ear`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_ear_fft.py tests/test_ear_recording_mode.py tests/test_ear_hold_state.py -q
```

Expected:
- FAIL because the new modules do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Move these helpers:
- `get_rms` -> `analysis.py`
- `_analyze_frequency_bands` support logic -> `analysis.py`
- gain boost helper -> `analysis.py`
- `select_mic` -> `devices.py`
- start sound helpers -> `platform.py`
- macOS voice isolation helper -> `platform.py`

Implementation rules:
- Keep detailed docs.
- Keep the old symbols re-exported or wrapped from `src.audio.ear`.
- Update `wizard_tui.py` only if it can benefit from the new `devices.py` or `models` extraction without causing churn. Do not force unrelated UI redesign.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_ear_fft.py tests/test_ear_recording_mode.py tests/test_ear_hold_state.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear_runtime/__init__.py src/audio/ear_runtime/analysis.py src/audio/ear_runtime/devices.py src/audio/ear_runtime/platform.py src/audio/ear.py src/utils/wizard_tui.py tests/test_ear_fft.py tests/test_ear_recording_mode.py tests/test_ear_hold_state.py
git commit -m "refactor: extract audio runtime helper modules"
```

### Task 5: Extract terminal menu and self-test into `src/audio/ear_runtime/menu.py`

**Files:**
- Create: `src/audio/ear_runtime/menu.py`
- Modify: `src/audio/ear.py`
- Test: `tests/test_ear_recording_mode.py`

- [ ] **Step 1: Write the failing test**

Add import coverage for the new module while keeping compatibility:

```python
def test_terminal_menu_remains_available_from_audio_ear_module():
    import src.audio.ear as ear_module
    assert hasattr(ear_module, "TerminalMenu")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_recording_mode.py -q
```

Expected:
- FAIL once you point tests at the new module before creating it.

- [ ] **Step 3: Write minimal implementation**

Move:
- `send_switch_command`
- `run_self_test`
- `TerminalMenu`

Rules:
- `menu.py` may import `src.ipc.client` and `src.ipc.protocol`.
- `src.audio.ear` must still re-export `TerminalMenu` and `send_switch_command` during migration.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_ear_recording_mode.py tests/test_ipc_messenger.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear_runtime/menu.py src/audio/ear.py tests/test_ear_recording_mode.py
git commit -m "refactor: extract ear menu and self-test helpers"
```

### Task 6: Add `src/ui/hud_client.py` and move Ear-facing HUD client code there

**Files:**
- Create: `src/ui/hud_client.py`
- Modify: `src/audio/ear.py`
- Test: `tests/test_ear_hold_state.py`

- [ ] **Step 1: Write the failing test**

Add focused tests for:
- sending HUD commands
- starting/stopping the volume sender thread in a controlled way

Prefer mock-driven tests that assert the UDP/TCP send calls rather than sleeping for long periods.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py -q
```

Expected:
- FAIL because `src.ui.hud_client` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Move:
- `_send_hud`
- `_start_volume_sender`

Rules:
- Keep `Ear` wrappers if current tests patch them.
- Do not create a circular import with `src/ui/hud.py`.
- Keep ports and command vocabulary unchanged.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/ui/hud_client.py src/audio/ear.py tests/test_ear_hold_state.py
git commit -m "refactor: extract hud client helpers for ear"
```

### Task 7: Introduce `src/streaming/capture_session.py` as the ear-side state module

**Files:**
- Create: `src/streaming/capture_session.py`
- Modify: `src/audio/ear.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_nemotron.py`

- [ ] **Step 1: Write the failing test**

Add tests around the state module directly. Cover:
- session id persists across recordings
- recording index increments on commit only
- chunk sequence increments per sent chunk
- chunk sequence resets after final stop
- overlap is prepended on non-final chunk
- overlap is not prepended on final stop

Example:

```python
def test_capture_session_resets_chunk_sequence_after_final_stop():
    session = CaptureSession(sample_rate=16000, overlap_seconds=0.002)
    session.begin_recording()
    session.mark_chunk_sent()
    session.mark_recording_committed()
    assert session.current_chunk_sequence_number == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_nemotron.py -q
```

Expected:
- FAIL because `CaptureSession` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `CaptureSession` to own:
- session id generation
- recording index tracking
- chunk sequence tracking
- chunk start time
- overlap tail bytes
- begin recording
- prepare chunk for send
- mark chunk sent
- mark recording committed

Rules:
- keep `should_split_chunk_after_silence` in `streaming_shared_logic.py`
- keep Nemotron special-case logic in `Ear` orchestration unless it is moved explicitly with tests
- wire `Ear` to call into `CaptureSession` instead of mutating those fields directly
- do not remove the old fields from `Ear` until wrappers or delegated properties preserve compatibility

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_nemotron.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/streaming/capture_session.py src/audio/ear.py tests/test_ear_hold_state.py tests/test_ear_nemotron.py
git commit -m "refactor: extract ear capture session state"
```

### Task 8: Extract the thin runtime into `src/audio/ear_runtime/controller.py` and `runtime.py`

**Files:**
- Create: `src/audio/ear_runtime/controller.py`
- Create: `src/audio/ear_runtime/runtime.py`
- Modify: `src/audio/ear.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_recording_mode.py`
- Test: `tests/test_ear_fft.py`
- Test: `tests/test_ear_nemotron.py`

- [ ] **Step 1: Write the failing test**

Add compatibility tests that prove:
- `from src.audio.ear import Ear, start_ear, get_active_models` still works
- `src.audio.ear.TerminalMenu` is still importable
- patch targets used by existing tests still resolve

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py tests/test_ear_fft.py tests/test_ear_nemotron.py -q
```

Expected:
- FAIL once imports are pointed at the new package internals before wrappers exist.

- [ ] **Step 3: Write minimal implementation**

Move:
- `Ear` runtime class -> `controller.py`
- `start_ear()` -> `runtime.py`

Then make `src/audio/ear.py` a compatibility façade like:

```python
from src.audio.ear_runtime.controller import Ear
from src.audio.ear_runtime.runtime import start_ear
from src.audio.ear_runtime.devices import select_mic
from src.audio.ear_runtime.analysis import get_rms
from src.audio.ear_runtime.menu import TerminalMenu, send_switch_command, run_self_test
```

Rules:
- Keep `get_active_models` either in `src.audio.ear.py` or extract it to a tiny model helper module, but re-export it from `src.audio.ear`.
- Keep test patch targets stable where practical.
- Do not delete wrappers until the full suite is green.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py tests/test_ear_fft.py tests/test_ear_nemotron.py tests/test_input_hotkeys.py tests/test_ipc_messenger.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear_runtime/controller.py src/audio/ear_runtime/runtime.py src/audio/ear.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py tests/test_ear_fft.py tests/test_ear_nemotron.py tests/test_input_hotkeys.py tests/test_ipc_messenger.py
git commit -m "refactor: split ear runtime into package modules"
```

### Task 9: Remove duplicate old input ownership from `Ear` only after compatibility is verified

**Files:**
- Modify: `src/audio/ear.py`
- Modify: `src/audio/ear_runtime/controller.py`
- Test: `tests/test_input_hotkeys.py`
- Test: `tests/test_ear_hold_state.py`

- [ ] **Step 1: Write the failing test**

Add or tighten tests that prove startup wiring uses `InputTrigger` callbacks instead of relying on `Ear.on_press/on_release/on_mouse_click`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_input_hotkeys.py tests/test_ear_hold_state.py -q
```

Expected:
- FAIL if runtime wiring still depends on the old direct handlers.

- [ ] **Step 3: Write minimal implementation**

Implementation rules:
- Remove runtime reliance on `Ear.on_press`, `Ear.on_release`, and `Ear.on_mouse_click`.
- If external compatibility is still needed, keep them as thin adapters or deprecated wrappers for one pass.
- Make `InputTrigger` the single owner of trigger semantics.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_input_hotkeys.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear.py src/audio/ear_runtime/controller.py tests/test_input_hotkeys.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py
git commit -m "refactor: remove duplicate ear input ownership"
```

### Task 10: Run focused regression suite and clean obvious dead code

**Files:**
- Modify: any touched files only if needed for dead import cleanup
- Test: `tests/test_ipc_messenger.py`
- Test: `tests/test_input_hotkeys.py`
- Test: `tests/test_ear_recording_mode.py`
- Test: `tests/test_ear_nemotron.py`
- Test: `tests/test_ear_fft.py`
- Test: `tests/test_ear_hold_state.py`

- [ ] **Step 1: Run the full targeted regression suite**

Run:

```bash
uv run pytest tests/test_ipc_messenger.py tests/test_input_hotkeys.py tests/test_ear_recording_mode.py tests/test_ear_nemotron.py tests/test_ear_fft.py tests/test_ear_hold_state.py -q
```

Expected:
- PASS

- [ ] **Step 2: Run a broader suite if time allows**

Run:

```bash
uv run pytest tests/test_brain.py tests/test_integration.py tests/test_streaming_session.py tests/test_streaming_shared_logic.py -q
```

Expected:
- PASS, or if there is failure, inspect whether the refactor broke a cross-module contract.

- [ ] **Step 3: Remove dead imports and dead wrappers only if tests stay green**

Rules:
- remove only code that is truly unused after the refactor
- do not delete compatibility re-exports that current tests or runtime still import

- [ ] **Step 4: Final commit**

```bash
git add src tests
git commit -m "refactor: finalize ear package extraction"
```

## Exact implementation commands summary

```bash
mkdir -p docs/superpowers/plans
uv run pytest tests/test_ipc_messenger.py -q
uv run pytest tests/test_ear_hold_state.py -q
uv run pytest tests/test_ear_recording_mode.py -q
uv run pytest tests/test_ear_nemotron.py -q
uv run pytest tests/test_ear_fft.py -q
uv run pytest tests/test_input_hotkeys.py -q
uv run pytest tests/test_brain.py tests/test_integration.py tests/test_streaming_session.py tests/test_streaming_shared_logic.py -q
```

## Notes for the executing agent

- Do not attempt a big-bang rename of `src/audio/ear.py` into a package in one step.
- First create deeper modules and delegate existing behavior into them.
- Keep `src.audio.ear` as the stable import seam until all targeted tests are green.
- Prefer delegation wrappers over patch-target breakage.
- Preserve the exact IPC bytes already asserted in tests.
- Preserve the Nemotron heartbeat behavior exactly.
- Preserve user-visible behavior before trying to delete wrappers.

## Self-Review

- [x] Reviewed current repo seams before writing this plan.
- [x] Used sub-agent review to validate file placement and migration risks.
- [x] Tests listed here cover real edge cases, not only happy paths.
- [x] Plan is incremental, testable, and avoids a risky big-bang rewrite.

