# Telemetry Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent**: 
    - The goal is to separate telemetry logging logic from the core brain execution logic in `src/backend/brain.py`.
    - We will move telemetry-specific functions to `src/backend/data_record/telemetry.py`.
    - To prevent circular imports between `brain.py` and the new telemetry module, the state classes (`RecordingState`, `SessionState`) and the shared globals (`session_store`, `backend_info`, etc.) will be moved to a new shared file `src/backend/state.py`.
    - **CRITICAL:** Do not assume anything, strictly follow the plan, ask questions if you don't understand anything.
    - **TOKEN EFFICIENCY:** To save context, this plan uses direct CUT and PASTE instructions. Do exactly as instructed to move the existing tested code.

**Architecture:**
- **State Management:** `src/backend/state.py` acts as the single source of truth for the session state. It holds the shared globals and dataclasses but imports nothing from the backend.
- **Telemetry:** `src/backend/data_record/telemetry.py` handles telemetry specific functions. It imports state from `src/backend/state.py` to read and update logs. It never imports `brain.py`.
- **Brain:** `src/backend/brain.py` imports state from `state.py` and logging tools from `data_record/telemetry.py`.

**Important Rule to follow :**
- **CRITICAL:** add detailed docs in functions and explain the code and logic in comments.  
- (**CRITICAL**) make the code function name and variable name clear and easily understandable instead of short and confusing names.
    - so a 5 year old child easily understands.
    - do not put any imagination and analogy to understand for 5 year old child.
    - write code function name and docs and code like this **developer get highest speed to read the code**
    - **Explain like a fresher** 
    - **Write docs in your step-by-step simple style.**
    - **Make the docs in function and file headers human-readable and literal.**

---
### Task 1: [ Read Instruction File ]
    - read GEMINI.md file
    - **CRITICAL:** add detailed docs in functions and explain the code and logic in comments.  
    - (**CRITICAL**) make the code function name and variable name clear not easily to understand instead of short and confusing names 
    - avoid surface level ( happy path ) tests use detailed tests.
    - write code function name and docs and code like this **developer get highest speed to read the code**

### Task 2: State Extraction (TDD)

**Files:**
- Create: `tests/test_backend_state.py`
- Create: `src/backend/state.py`

- [ ] **Step 1: Write the failing test**
Create `tests/test_backend_state.py` to ensure `state.py` handles dataclasses independently of brain.

```python
import sys
import pytest

def test_state_independent_of_brain():
    """Verify that state can be imported without brain.py"""
    if "src.backend.brain" in sys.modules:
        del sys.modules["src.backend.brain"]
    
    import src.backend.state as state
    assert "src.backend.brain" not in sys.modules
    
    # Verify global locks and store
    assert hasattr(state, "session_store")
    assert hasattr(state, "session_store_lock")
    assert hasattr(state, "backend_info")
    assert hasattr(state, "backend_lock")

def test_session_state_creation():
    """Verify state initialization edge cases"""
    import src.backend.state as state
    # Create with None engine
    session = state.SessionState(engine=None)
    
    # Test getting recording state handles missing index
    rec = session.get_or_create_recording(1)
    assert rec.received_count == 0
    assert 1 in session.recordings
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_backend_state.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.backend.state'"

- [ ] **Step 3: Write minimal implementation (CUT AND PASTE)**
Create `src/backend/state.py`.
Add these imports at the top of `state.py`:
```python
import threading
from dataclasses import dataclass, field
from src.streaming.streaming_session_telemetry import StreamingSessionTelemetryRecorder
```
Then, **CUT** the following items exactly as they are from `src/backend/brain.py` and **PASTE** them into `src/backend/state.py`:
1. `backend_info` dictionary
2. `backend_lock` definition
3. `session_store` dictionary
4. `session_store_lock` definition
5. `@dataclass class RecordingState:` (and its entire body)
6. `@dataclass class SessionState:` (and its entire body including `get_or_create_recording`)

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_backend_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add tests/test_backend_state.py src/backend/state.py src/backend/brain.py
git commit -m "feat: extract state globals and classes into state.py"
```

### Task 3: Refactor Brain to use State

**Files:**
- Modify: `src/backend/brain.py`
- Modify: `tests/test_brain.py`

- [ ] **Step 1: Update `brain.py` imports**
In `src/backend/brain.py`, add this import at the top to restore access to the state you just removed:
```python
from src.backend.state import (
    backend_info, backend_lock,
    session_store, session_store_lock,
    RecordingState, SessionState
)
```

- [ ] **Step 2: Update `tests/test_brain.py` to use `state.py`**
In `tests/test_brain.py`, add `import src.backend.state as state`.
Search and replace all references to `brain.` state so they point to `state.` instead:
- Replace `brain.session_store` with `state.session_store`
- Replace `brain.backend_info` with `state.backend_info`
- Replace `brain.SessionState` with `state.SessionState`
- Replace `brain.RecordingState` with `state.RecordingState`

- [ ] **Step 3: Run existing tests**
Run: `pytest tests/test_brain.py -v`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add src/backend/brain.py tests/test_brain.py
git commit -m "refactor: update brain.py and its tests to use state.py"
```

### Task 4: Telemetry Extraction (TDD)

**Files:**
- Create: `tests/test_backend_data_record_telemetry.py`
- Create: `src/backend/data_record/__init__.py`
- Create: `src/backend/data_record/telemetry.py`

- [ ] **Step 1: Write the failing test**
Create `tests/test_backend_data_record_telemetry.py` to test isolated telemetry logic edge cases.

```python
import pytest
from unittest.mock import MagicMock

def test_telemetry_independent_of_brain():
    import sys
    if "src.backend.brain" in sys.modules:
        del sys.modules["src.backend.brain"]
    import src.backend.data_record.telemetry as telemetry
    assert "src.backend.brain" not in sys.modules

def test_telemetry_disabled_fast_return(monkeypatch):
    import src.backend.data_record.telemetry as telemetry
    monkeypatch.setattr(telemetry, "STREAMING_TELEMETRY_ENABLED", False)
    # Should return None immediately without errors
    assert telemetry._telemetry_recorder_for_session("test_sess") is None

def test_model_name_for_telemetry_missing_engine():
    import src.backend.data_record.telemetry as telemetry
    import src.backend.state as state
    state.backend_info["engine"] = None
    assert telemetry._model_name_for_telemetry() is None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_backend_data_record_telemetry.py -v`
Expected: FAIL with "No module named 'src.backend.data_record'"

- [ ] **Step 3: Write minimal implementation (CUT AND PASTE)**
Create the folder `src/backend/data_record/`.
Create `src/backend/data_record/__init__.py` (leave it empty).
Create `src/backend/data_record/telemetry.py`.

Add these imports at the top of `telemetry.py`:
```python
import os
from pathlib import Path
from src.utils.env_utils import get_float_from_environment
from src.streaming.streaming_session_telemetry import StreamingSessionTelemetryRecorder
from src.backend.state import (
    backend_info, session_store, session_store_lock, SessionState
)
```

**CUT** these 3 constants from `brain.py` and **PASTE** into `telemetry.py`:
1. `RECORDING_MODE`
2. `STREAMING_TELEMETRY_ENABLED`
3. `STREAMING_TELEMETRY_DIR`

**CUT** these 6 functions from `brain.py` and **PASTE** into `telemetry.py`:
1. `_model_name_for_telemetry`
2. `_telemetry_seed`
3. `_telemetry_recorder_for_session`
4. `_update_chunk_telemetry_summary`
5. `_update_session_telemetry_summary`
6. `_handle_session_telemetry_event`

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_backend_data_record_telemetry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add tests/test_backend_data_record_telemetry.py src/backend/data_record/ src/backend/brain.py
git commit -m "feat: extract telemetry logic to data_record module"
```

### Task 5: Refactor Brain to use new Telemetry

**Files:**
- Modify: `src/backend/brain.py`
- Modify: `tests/test_brain.py`

- [ ] **Step 1: Update `brain.py` imports**
In `src/backend/brain.py`, add this import at the top so it can use the logger functions you just moved:
```python
from src.backend.data_record.telemetry import (
    _telemetry_seed,
    _telemetry_recorder_for_session,
    _update_chunk_telemetry_summary,
    _update_session_telemetry_summary,
    _handle_session_telemetry_event,
    STREAMING_TELEMETRY_ENABLED,
    STREAMING_TELEMETRY_DIR,
    RECORDING_MODE
)
```
*(Also verify that if `_is_no_streaming_mode` in brain.py needs `NO_STREAMING_MODE`, it still has access to it).*

- [ ] **Step 2: Update `tests/test_brain.py` for telemetry mock paths**
In `tests/test_brain.py`, find the telemetry mock strings in `test_handle_session_event_writes_telemetry_file`.
Change `"src.backend.brain.STREAMING_TELEMETRY_ENABLED"` (if mocked that way, or if set via monkeypatch on `brain`) to target the new location: `src.backend.data_record.telemetry.STREAMING_TELEMETRY_ENABLED`. 
If monkeypatch was used directly on the `brain` module, replace it with `monkeypatch.setattr("src.backend.data_record.telemetry.STREAMING_TELEMETRY_ENABLED", True)`.

- [ ] **Step 3: Run existing tests**
Run: `pytest tests/test_brain.py -v`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add src/backend/brain.py tests/test_brain.py
git commit -m "refactor: update brain.py to use data_record telemetry module"
```