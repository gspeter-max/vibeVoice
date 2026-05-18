# Ear Runtime Controller Responsibility Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

You = frashAgent
**Goal information for freshAgent** :
- The user wants a deeper cleanup of `src/audio/ear_runtime/controller.py`.
- The problem is not one bug. The problem is one file doing too many jobs.
- We must move logic only into files that already fit that responsibility.
- We must not create duplicate helpers when a matching helper already exists.
- We already cleaned obvious wrapper noise, duplicate logic, dead code, and session mirroring.
- We already made `CaptureSession` the single owner of session state.
- This plan now focuses on reducing controller responsibilities further without changing runtime behavior.
- The user wants a clear mapping of which function belongs in which file before implementation.
- Read these files first to understand the current split:
  - `src/audio/ear_runtime/controller.py` - Ear runtime orchestration, still too broad.
  - `src/audio/ear_runtime/runtime.py` - process bootstrap and input-trigger wiring.
  - `src/audio/ear_runtime/analysis.py` - stateless audio math helpers.
  - `src/audio/ear_runtime/devices.py` - microphone selection and input-device helpers.
  - `src/audio/ear_runtime/menu.py` - terminal menu, model switching, self-test helpers.
  - `src/ipc/client.py` - low-level Ear-to-Brain socket transport.
  - `src/streaming/capture_session.py` - session/chunk state owner.
  - `src/ui/hud_client.py` - HUD command send and live volume sender.
  - `tests/test_ear_hold_state.py` - main controller behavior tests.
  - `tests/test_ear_recording_mode.py` - runtime/menu/controller integration seams.
  - `tests/test_ear_nemotron.py` - chunk heartbeat logic.
  - `tests/test_input_hotkeys.py` - input-trigger behavior.

**Architecture:**  
`controller.py` should remain the thin orchestration layer.  
It should coordinate these existing modules instead of re-owning their logic.

```text
InputTrigger / keyboard
        |
        v
   controller.Ear
   - recording flow
   - callback flow
   - stop/split decisions
        |
        +--> analysis.py         (RMS / gain / frequency analysis)
        +--> devices.py          (mic index resolution)
        +--> hud_client.py       (HUD command + volume sender)
        +--> ipc/client.py       (socket stream transport)
        +--> capture_session.py  (session/chunk state transitions)
```

**Important Rule to follow :**
- **CRITICAL:** add  docs in functions and explain the code and logic in comments.
- **CRITICAL:** make the code function name and variable name literal.
- Keep code DRY.
- Do not create new files for responsibilities that already have a home.
- Do not preserve compatibility wrappers if tests can be updated to the real module boundary.

## Issue Coverage Check

This plan must cover every issue already identified:

1. Same state stored in 2 places
   - covered by Task 5 and Task 6
   - result: `CaptureSession` is the single owner

2. Functions that only call another function
   - covered by Task 2, Task 3, Task 4, and Task 6
   - result: controller-only wrapper noise removed

3. Same job written in more than one way
   - partly already fixed in code for gain/frequency flow
   - still needs explicit cleanup of remaining local wrapper helpers
   - covered by new Task 5A below

4. One line did nothing
   - already fixed in code
   - must be protected against regression by final verification in Task 8

5. Fake exception flow
   - already fixed in code
   - must be protected against regression by final verification in Task 8

6. Controller did too many jobs
   - covered by Task 2 through Task 6
   - result: device, HUD, IPC, and session responsibilities move to existing modules

7. VAD flow lived inside controller
   - deliberate decision: keep VAD split policy in `controller.py`
   - reason: `src/audio/vad_segmenter.py` already owns VAD detection primitives, but recording orchestration and chunk split decisions still belong to controller
   - this is not duplicated logic today, so do not move it unless a later pass finds a real reusable unit

---

### Task 1 : Read instructions and freeze boundaries

**Files:**
- Modify: none
- Read: `/Users/apple/.gemini/GEMINI.md`
- Read: `src/audio/ear_runtime/controller.py`
- Read: `src/audio/ear_runtime/runtime.py`
- Read: `src/audio/ear_runtime/devices.py`
- Read: `src/ui/hud_client.py`
- Read: `src/ipc/client.py`
- Read: `src/streaming/capture_session.py`
- Test: none

- [ ] **Step 1: Re-read the boundary map before touching code**

Boundary decisions:
- Keep in `controller.py`:
  - `Ear`
  - `_audio_callback`
  - `_start_recording_state`
  - `on_press`
  - `on_release`
  - `_stop_and_send`
  - `record_loop`
  - `_record_loop_tick`
  - `cleanup`
- Move to `devices.py`:
  - `_default_input_device_index`
  - `_resolve_input_device_index`
- Move to `hud_client.py`:
  - `_send_hud` behavior
  - `_start_volume_sender` behavior
  - shared thread launch helper for HUD actions only if it stays useful
- Move to `ipc/client.py`:
  - `_open_brain_stream`
  - `_stream_chunk_to_brain`
  - `_close_brain_stream`
- Push further into `capture_session.py`:
  - chunk-finalization state transitions that are purely session bookkeeping
- Keep in `controller.py` for now:
  - VAD split policy and recording orchestration around `_utterance_gate`
  - key press/release orchestration

- [ ] **Step 2: Confirm no new file is needed**

Expected result:
- No new file for device helpers.
- No new file for HUD helpers.
- No new file for Brain stream transport.
- No new file for session state.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-18-ear-runtime-controller-responsibility-refactor.md
git commit -m "docs: add ear runtime controller refactor plan"
```

### Task 2: Move microphone index resolution into `devices.py`

**Files:**
- Modify: `src/audio/ear_runtime/devices.py`
- Modify: `src/audio/ear_runtime/controller.py`
- Test: `tests/test_ear_hold_state.py`

- [ ] **Step 1: Write the failing test**

Add tests in `tests/test_ear_hold_state.py` for:
- explicit `input_device_index` wins over env/default
- valid `VIBEVOICE_MIC_INDEX` env value is used
- invalid `VIBEVOICE_MIC_INDEX` falls back to default device

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py -q
```

Expected:
- at least one new mic-resolution test fails because helper does not exist in `devices.py`

- [ ] **Step 3: Write minimal implementation**

Implementation shape:
- add `default_input_device_index(pyaudio_instance) -> int`
- add `resolve_input_device_index(pyaudio_instance, input_device_index) -> int`
- make names public and literal in `devices.py`
- remove `_default_input_device_index` and `_resolve_input_device_index` from `controller.py`
- import `resolve_input_device_index` into `controller.py`

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py -q
```

Expected:
- controller tests still pass
- new helper tests pass

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear_runtime/devices.py src/audio/ear_runtime/controller.py tests/test_ear_hold_state.py
git commit -m "refactor: move mic index resolution into devices helpers"
```

### Task 3: Remove HUD wrappers from `controller.py`

**Files:**
- Modify: `src/ui/hud_client.py`
- Modify: `src/audio/ear_runtime/controller.py`
- Modify: `src/audio/ear_runtime/runtime.py`
- Modify: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_recording_mode.py`

- [ ] **Step 1: Write the failing test**

Add or update tests so they patch the real HUD helpers instead of controller wrapper methods:
- patch `src.ui.hud_client.send_hud_command`
- patch `src.ui.hud_client.start_volume_sender_thread`
- verify `on_press` still triggers listen HUD send and volume sender startup
- verify stop paths still trigger process HUD send

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- failures because code still routes through controller wrappers

- [ ] **Step 3: Write minimal implementation**

Implementation shape:
- delete `_send_hud`
- delete `_start_volume_sender`
- in `controller.py`, call `send_hud_command(..., socket_factory=socket.socket)` directly
- in `controller.py`, call `start_volume_sender_thread(...)` directly
- in `runtime.py`, replace `ear._send_hud` and `ear._start_volume_sender` usage with direct helper calls
- decide whether `_start_daemon_thread` stays:
  - keep it only if it still reduces repeated HUD-thread boilerplate
  - delete it if direct calls become cleaner

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- all affected tests pass using real HUD helper seams

- [ ] **Step 5: Commit**

```bash
git add src/ui/hud_client.py src/audio/ear_runtime/controller.py src/audio/ear_runtime/runtime.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py
git commit -m "refactor: remove controller hud wrappers"
```

### Task 4: Move Brain raw-stream lifecycle into `ipc/client.py`

**Files:**
- Modify: `src/ipc/client.py`
- Modify: `src/audio/ear_runtime/controller.py`
- Modify: `src/audio/ear_runtime/runtime.py`
- Modify: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_recording_mode.py`

- [ ] **Step 1: Write the failing test**

Add or update tests for raw no-streaming behavior around the real IPC boundary:
- opening stream when socket path exists
- refusing to open when socket path is missing
- send failure closes stream and clears stream state
- final stop closes raw stream

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- failure because the transport lifecycle still lives in controller

- [ ] **Step 3: Write minimal implementation**

Implementation shape:
- extend `ipc/client.py` with explicit helpers that match current behavior:
  - `open_checked_raw_audio_stream_to_brain(...) -> socket.socket | None`
  - `send_raw_audio_stream_chunk_or_close(raw_stream_socket, chunk_bytes) -> socket.socket | None`
  - `close_raw_audio_stream_and_forget(raw_stream_socket) -> None`
- keep transport rules in `ipc/client.py`
- remove `_open_brain_stream`, `_stream_chunk_to_brain`, `_close_brain_stream` from `controller.py`
- replace controller calls with direct `ipc.client` helpers
- keep only orchestration and logging in controller

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- no-streaming mode still behaves the same
- raw stream state still gets cleared on failure

- [ ] **Step 5: Commit**

```bash
git add src/ipc/client.py src/audio/ear_runtime/controller.py src/audio/ear_runtime/runtime.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py
git commit -m "refactor: move raw brain stream lifecycle into ipc client"
```

### Task 5A: Remove remaining local wrapper helpers and duplicate code paths

**Files:**
- Modify: `src/audio/ear_runtime/controller.py`
- Modify: `src/audio/ear_runtime/analysis.py`
- Modify: `src/audio/ear_runtime/runtime.py`
- Modify: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_recording_mode.py`

- [ ] **Step 1: Write the failing test**

Add or update tests to assert the real helper boundaries:
- audio callback uses `boost_audio_chunk(...)` from `analysis.py` directly or through one clearly justified path
- no test should patch `_boost_audio_chunk` if the wrapper is removed
- runtime/controller thread-start behavior still works if `_start_daemon_thread` is removed

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- failures where tests or code still depend on local wrapper-only seams

- [ ] **Step 3: Write minimal implementation**

Implementation shape:
- decide whether `_boost_audio_chunk` adds real value:
  - if not, delete it and call `boost_audio_chunk(...)` directly
  - if yes, keep exactly one path and update tests to justify it
- decide whether `_start_daemon_thread` adds real value:
  - if not, delete it
  - if yes, keep it only as one shared orchestration helper and do not duplicate thread-start code elsewhere
- remove any remaining duplicate path for the same work

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_recording_mode.py -q
```

Expected:
- controller uses one clear path per job
- no test depends on wrapper-only behavior unless the wrapper has a real reason to exist

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear_runtime/controller.py src/audio/ear_runtime/analysis.py src/audio/ear_runtime/runtime.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py
git commit -m "refactor: remove remaining local controller wrappers"
```

### Task 5: Push more session bookkeeping into `capture_session.py`

**Files:**
- Modify: `src/streaming/capture_session.py`
- Modify: `src/audio/ear_runtime/controller.py`
- Modify: `tests/test_ear_hold_state.py`
- Modify: `tests/test_ear_nemotron.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_nemotron.py`

- [ ] **Step 1: Write the failing test**

Add tests for session bookkeeping behavior at the `CaptureSession` level:
- final stop clears overlap tail and resets chunk-start time
- non-final send updates next chunk start time
- chunk index returned for telemetry matches sequence just sent
- recording commit advances recording index and resets chunk sequence

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_nemotron.py -q
```

Expected:
- new state-transition tests fail because `CaptureSession` does not own enough transition logic yet

- [ ] **Step 3: Write minimal implementation**

Implementation shape:
- add literal methods to `CaptureSession` for controller-needed state transitions, for example:
  - `mark_recording_stopped()`
  - `mark_nonfinal_chunk_sent(now_seconds)`
  - `current_chunk_age_seconds(now_seconds) -> float`
- update controller to ask `CaptureSession` for chunk-age/transition behavior instead of mutating raw session fields
- keep private compatibility properties only if tests still need them
- remove compatibility properties if tests can directly assert against `_capture_session`

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_nemotron.py -q
```

Expected:
- controller uses `CaptureSession` more directly
- less state math remains in controller

- [ ] **Step 5: Commit**

```bash
git add src/streaming/capture_session.py src/audio/ear_runtime/controller.py tests/test_ear_hold_state.py tests/test_ear_nemotron.py
git commit -m "refactor: move session bookkeeping into capture session"
```

### Task 6: Remove remaining compatibility-only private session aliases if no longer needed

**Files:**
- Modify: `src/audio/ear_runtime/controller.py`
- Modify: `tests/test_ear_hold_state.py`
- Modify: `tests/test_ear_nemotron.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_nemotron.py`

- [ ] **Step 1: Write the failing test**

Replace tests that reach through aliases like:
- `_current_session_id`
- `_recording_index`
- `_chunk_seq`
- `_chunk_started_at`
- `_chunk_overlap_audio_bytes`
- `_last_chunk_tail_bytes`

with tests that assert the real owner:
- `ear._capture_session.current_session_id`
- `ear._capture_session.current_recording_index`
- `ear._capture_session.current_chunk_sequence_number`
- `ear._capture_session.chunk_started_at_seconds`
- `ear._capture_session.overlap_audio_byte_count`
- `ear._capture_session.last_chunk_tail_bytes`

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_nemotron.py -q
```

Expected:
- failures where old aliases are still expected

- [ ] **Step 3: Write minimal implementation**

Implementation shape:
- once tests no longer need compatibility aliases, delete those properties from `controller.py`
- update controller call sites to use `self._capture_session...` directly where clearer
- keep code literal and local

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_ear_hold_state.py tests/test_ear_nemotron.py -q
```

Expected:
- no tests depend on compatibility-only aliases
- controller reads cleaner

- [ ] **Step 5: Commit**

```bash
git add src/audio/ear_runtime/controller.py tests/test_ear_hold_state.py tests/test_ear_nemotron.py
git commit -m "refactor: remove legacy controller session aliases"
```

### Task 8: Full targeted verification

**Files:**
- Modify: none
- Test: `tests/test_ear_fft.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_ear_recording_mode.py`
- Test: `tests/test_ear_nemotron.py`
- Test: `tests/test_input_hotkeys.py`

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
uv run pytest tests/test_ear_fft.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py tests/test_ear_nemotron.py tests/test_input_hotkeys.py -q
```

Expected:
- all focused Ear/runtime tests pass

- [ ] **Step 2: Smoke-check the files for duplicate helper reintroduction**

Run:

```bash
rg -n "_send_hud|_start_volume_sender|_open_brain_stream|_stream_chunk_to_brain|_close_brain_stream|_default_input_device_index|_resolve_input_device_index|_boost_audio_chunk" src/audio/ear_runtime src/ipc src/ui
```

Expected:
- old controller-only helpers are gone
- helper ownership matches the intended module

- [ ] **Step 3: Smoke-check that dead code and fake exception flow did not return**

Run:

```bash
rg -n "bool\\(self\\._current_session_id\\)|raise BrokenPipeError|except \\(BrokenPipeError, ConnectionResetError\\)" src/audio/ear_runtime/controller.py src/ipc/client.py
```

Expected:
- no dead no-op line exists
- no fake exception-control-flow pattern exists

- [ ] **Step 4: Commit**

```bash
git add src/audio/ear_runtime/controller.py src/audio/ear_runtime/runtime.py src/audio/ear_runtime/devices.py src/ipc/client.py src/streaming/capture_session.py src/ui/hud_client.py tests/test_ear_fft.py tests/test_ear_hold_state.py tests/test_ear_recording_mode.py tests/test_ear_nemotron.py tests/test_input_hotkeys.py
git commit -m "refactor: split ear controller responsibilities by module"
```

## Edge Cases To Protect During Implementation

1. `VIBEVOICE_MIC_INDEX` exists but is not an integer.
2. Raw no-streaming mode loses its Brain socket mid-recording.
3. Final stop happens with no utterance bytes to flush.
4. Nemotron fixed 1.12-second heartbeat still bypasses silence split logic.
5. Volume sender and HUD commands still stop correctly when recording flips to false.

## Self-Review (before sharing the plan)
- [ ] run sub-agent for reveiw the plan.
- [ ] Tests catch real edge cases, not just happy paths?
- [ ] No duplicate helper was added where an existing module already owns the responsibility?
- [ ] Controller is thinner after each task, not just renamed?
