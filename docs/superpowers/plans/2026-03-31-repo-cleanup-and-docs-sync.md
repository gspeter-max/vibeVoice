# Repo Cleanup and Docs Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the repository docs, tests, and install metadata in sync with the live streaming voice-typing architecture, while removing only clearly obsolete or generated files.

**Architecture:** Treat this as four small cleanup slices: user-facing docs, runtime/install consistency, test alignment, and safe file removal. Keep historical planning/spec documents that still add value, but remove artifacts that describe dead behavior or are generated noise.

**Tech Stack:** Python 3.11, Bash, Markdown, Pytest, `rg`, `apply_patch`

---

### Task 1: Sync the user-facing docs to the live streaming app

**Files:**
- Modify: `README.md`
- Modify: `docs/THEMES.md`
- Modify: `docs/testing/hold-to-record-implementation-summary.md`
- Modify: `docs/testing/hold-to-record-integration-test-plan.md`
- Delete: `MOUSE_CONTROL_EDGE_CASES.md`

- [ ] **Step 1: Audit the text that is now stale**

Run:
```bash
rg -n "4-click|left mouse button|Right CMD|right mouse button|theme_config|hold-to-record|streaming mode|openvino" README.md docs MOUSE_CONTROL_EDGE_CASES.md
```

Expected:
- The search should show the places where the docs still describe older behavior or incomplete architecture details.

- [ ] **Step 2: Update the docs to match the live system**

Make these doc-level changes:
- `README.md`: describe the current streaming flow, Right Cmd keyboard control, Right mouse-button hold-to-record, the live HUD IPC ports, and the actual backend selection model.
- `docs/THEMES.md`: keep the current unified flowing color theme description, but make it clear that the repository currently ships one theme implementation rather than the older multi-theme plan.
- `docs/testing/hold-to-record-implementation-summary.md`: update the summary language so it describes the current Right mouse-button hold-to-record behavior and the streaming socket pipeline, not the removed 4-click system.
- `docs/testing/hold-to-record-integration-test-plan.md`: change the test steps and expected results so they match the current Right mouse-button hold behavior and the current recording lifecycle.
- `MOUSE_CONTROL_EDGE_CASES.md`: delete it so the repo only carries live guidance for the streaming/right-click flow.

- [ ] **Step 3: Verify the docs no longer contradict the code**

Run:
```bash
rg -n "4-click|left mouse button|mouse_click_count|mouse_click_timeout|theme_config" README.md docs MOUSE_CONTROL_EDGE_CASES.md
```

Expected:
- No references remain to the removed 4-click mouse system.
- Any remaining references to historical plans should be clearly labeled as historical and not as live behavior.

- [ ] **Step 4: Commit the doc sync**

Run:
```bash
git add README.md docs/THEMES.md docs/testing/hold-to-record-implementation-summary.md docs/testing/hold-to-record-integration-test-plan.md MOUSE_CONTROL_EDGE_CASES.md
git commit -m "docs: sync repository docs with streaming voice flow"
```

---

### Task 2: Fix install and backend consistency issues

**Files:**
- Modify: `requirements.txt`
- Modify: `src/brain.py`
- Modify: `src/backend_openvino.py`

- [ ] **Step 1: Fix the broken dependency list**

Edit `requirements.txt` so the last dependency line is valid and separate:
```txt
numpy>=1.26.0
sherpa-onnx>=1.10.0
```

Expected:
- The file remains a pip-compatible fallback and no longer contains the concatenated invalid requirement string.

- [ ] **Step 2: Align the OpenVINO backend contract**

Update the OpenVINO path so `brain.load_backend()` and `backend_openvino.load_model()` agree on the function signature.

Minimal safe direction:
- Either accept `model_name` in `backend_openvino.load_model(model_name=None)` and ignore it, or
- change the brain loader so it calls the OpenVINO loader without an argument.

Choose the option that keeps the loader interface parallel with the other backends and avoids future drift.

- [ ] **Step 3: Verify the startup and backend text still matches reality**

Run:
```bash
rg -n "BACKEND=openvino|PARAKEET_THREADS|faster-whisper|OpenVINO|requirements.txt" README.md start.sh src
```

Expected:
- The docs and startup script should describe the same backend behavior.
- The OpenVINO path should no longer advertise an argument mismatch.

- [ ] **Step 4: Commit the consistency fix**

Run:
```bash
git add requirements.txt src/brain.py src/backend_openvino.py start.sh
git commit -m "fix: align backend loaders and dependency fallback"
```

---

### Task 3: Bring the test suite up to date or remove obsolete tests

**Files:**
- Modify: `tests/test_backend.py`
- Modify: `tests/test_brain.py`
- Modify: `tests/test_ear.py`
- Modify: `tests/test_integration.py`
- Keep: `tests/test_ear_fft.py`, `tests/test_ear_hold_state.py`, `tests/test_theme_manager.py` unless they need small wording updates

- [ ] **Step 1: Identify the stale API assumptions**

Run:
```bash
rg -n "audio_queue|worker|frames|_send_to_brain|load_model\\(\\)\\s*$|base.en|mouse_click_count|left mouse button" tests
```

Expected:
- The search should show the tests that still assume the old queue-based ear/brain implementation or an outdated backend default.

- [ ] **Step 2: Rewrite the tests to match the streaming architecture**

Update the tests so they check the current behavior:
- `tests/test_backend.py`: assert the real default model and loader arguments used by `backend_faster_whisper.load_model()`, plus the current `transcribe()` contract.
- `tests/test_brain.py`: test `handle_connection()` or `start_server()` behavior using the socket-streaming model and `CMD_SWITCH_MODEL:` command path, not queue objects that no longer exist.
- `tests/test_ear.py`: test `get_rms()`, the live mouse/keyboard handlers, and the socket streaming helpers that exist now.
- `tests/test_integration.py`: exercise the real Unix-socket flow between Ear and Brain with the current streaming connection behavior.

If a test is no longer worth keeping because it only validates removed behavior and provides no value for the current app, delete the file instead of forcing a fake update.

- [ ] **Step 3: Run the targeted test set**

Run:
```bash
pytest tests/test_backend.py tests/test_brain.py tests/test_ear.py tests/test_integration.py -v
```

Expected:
- The updated tests should pass against the current code path.
- Any obsolete test file chosen for deletion should be removed from the test run.

- [ ] **Step 4: Commit the test cleanup**

Run:
```bash
git add tests/test_backend.py tests/test_brain.py tests/test_ear.py tests/test_integration.py
git commit -m "test: align suite with streaming architecture"
```

---

### Task 4: Remove only clearly obsolete or generated files

**Files:**
- Delete: `src/__pycache__/`
- Delete: `tests/__pycache__/`

- [ ] **Step 1: Confirm the files are safe to remove**

Run:
```bash
rg -n "__pycache__" .gitignore .git/info/exclude . || true
```

Expected:
- Generated bytecode should be treated as disposable.

- [ ] **Step 2: Delete the generated artifacts and any file that is clearly dead**

Delete only the files that are either generated noise or obviously obsolete after review.
Do not delete the docs/superpowers planning/spec files unless they are directly misleading and not useful for future work.

- [ ] **Step 3: Verify the tree is clean**

Run:
```bash
git status --short
```

Expected:
- Only the intended doc/code/test changes should remain.
- No `__pycache__` directories should be left behind in the tracked tree.

- [ ] **Step 4: Commit the cleanup removal**

Run:
```bash
git add -A
git commit -m "chore: remove obsolete artifacts and generated files"
```

---

## Self-Review Checklist

- [ ] The plan keeps historical planning/spec docs that are still useful.
- [ ] The plan updates the live docs to match the current streaming architecture.
- [ ] The plan fixes the install metadata and backend contract mismatch.
- [ ] The plan updates or removes stale tests instead of preserving broken assumptions.
- [ ] The plan only deletes files that are clearly obsolete or generated noise.
