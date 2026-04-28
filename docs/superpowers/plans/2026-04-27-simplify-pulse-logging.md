# Logging Simplification — "Variant 2: The Pulse" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **code-change** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---

**Goal information for fresh Agent:**

This plan simplifies the logging output of the VibeVoice application to match **Variant 2: "The Pulse"** design.

The application currently logs dozens of noisy lines every second — one line for every audio chunk decoded, every UDP volume packet received (60+ per second in `hud.py`), every audio chunk sent (60+ per second in `ear.py`), and a debug voice level every second. This was useful during early development but now makes the terminal unreadable and wastes compute.

**The target output after this plan is complete:**
```
[Brain] 🎙️  Recording started...
[HUD]   → Listening
[Brain] 🏁 "How is the weather today?" (0.35s)
[HUD]   → Done
```

**What the user decided in this session:**
- Keep logging model switches
- Log when a new recording starts (first chunk, seq==0) → "Recording started..."
- Log the final transcript text + time taken in the specific "Pulse" format: `[Brain] 🏁 "text" (0.35s)`
- HUD only logs when its visible state CHANGES — NOT every volume packet
- HUD logs state transitions using the arrow format: `[HUD] → Listening` and `[HUD] → Done`
- Remove the 60-per-second `[HUD] 🎤 Received volume` spam
- Remove the 60-per-second `[HUD] 🎵 Frequency bands updated` spam
- Remove the 60-per-second `📤 Audio chunk sent` spam from `ear.py`
- Remove the per-tick DEBUG voice log in `_tick()` (fires every ~1 second while listening)
- Remove the per-chunk `[Brain] decode took Xs` log in `_handle_audio_chunk`
- Remove dead `pass` statement in `_tick()` (leftover with no real code)
- Replace 7 pre-initialized dedup variables with a single `dedup_analysis = None` in `brain.py`

**Files to read before starting:**
- `src/audio/ear.py` — microphone and audio capture. Changes in `_send_audio_chunk_to_brain()`
- `src/backend/brain.py` — main AI inference loop. Changes in `_handle_audio_chunk()` and `_finalize_recording_if_ready()`
- `src/ui/hud.py` — macOS menu bar HUD. Changes in `_on_volume()`, `_on_frequency_bands()`, `_tick()`, `show_listening()`, `show_done()`
- `tests/test_brain.py` — unit tests for brain.py

**Architecture:**
```
Ear (microphone)
    │ sends audio chunks via Unix socket (SILENT)
    ▼
Brain (_handle_audio_chunk)
    ├── chunk 1 → transcribe → store (LOG: "Recording started...")
    ├── chunk 2 → transcribe → deduplicate → store (SILENT)
    │   ...
    └── finalize → stitch → paste → log final text (LOG: [Brain] 🏁 "text" (Xs))
                                     ▲
                              [Brain] 🏁 "text" (Xs)

HUD (TCP/UDP)
    ├── receives state commands → logs state change once (LOG: [HUD] → State)
    ├── receives volume UDP → silently updates float (SILENT)
    └── receives freq bands UDP → silently updates dict (SILENT)
```

**Important Rules:**
- CRITICAL: Add detailed docs in functions and explain code logic in comments
- CRITICAL: Function names and variable names must be clear and literal — a 5-year-old dev can read them
- Write docs step-by-step in simple English. Explain like a fresher
- Avoid surface-level tests — test edge cases and failure states
- Follow GEMINI.md throughout

---

## Task 1: Read Required Files

- [ ] **Step 1: Read the instructions file**
  Read `/Users/apple/.gemini/GEMINI.md` and follow all rules throughout execution.

- [ ] **Step 2: Read the target source files**
  - `src/audio/ear.py` lines 530–560 (`_send_audio_chunk_to_brain`)
  - `src/backend/brain.py` lines 330–385 (`_finalize_recording_if_ready`) and 403–518 (`_handle_audio_chunk`)
  - `src/ui/hud.py` lines 413–425 (`show_listening`, `show_done`) and 526–565 (`_on_volume`, `_on_frequency_bands`, `_tick`)

---

## Task 2: Simplify and Reformat `brain.py`

**Files:**
- Modify: `src/backend/brain.py`

**What changes and why:**
1. In `_handle_audio_chunk`:
   - Remove 7 pre-initialized variables (`overlap_word_count`, etc.) — replace with single `dedup_analysis = None`
   - Remove the per-chunk log `[Brain] 🎙️ decode took Xs`
   - Add `[Brain] 🎙️ Recording started...` log only when `seq == 0 and rec.received_count == 1`
2. In `_finalize_recording_if_ready`:
   - Replace the verbose multi-line `log.info("🏁 Finalizing recording", ...)` with the single-line Pulse format: `log.info(f"[Brain] 🏁 \"{text}\" ({stt_time:.2f}s)")`
   - Update the "Nothing detected" log to match the style: `log.info(f"[Brain] 🔇 Nothing detected")`

- [ ] **Step 1: Write the failing test for Brain logs**

  Add to `tests/test_brain.py`:

```python
def test_brain_logs_match_pulse_format(caplog):
    """
    Verify that Brain logs follow the 'The Pulse' design.
    - Start: '[Brain] 🎙️ Recording started...'
    - End: '[Brain] 🏁 "text" (Xs)'
    """
    import logging
    from unittest.mock import MagicMock
    import src.backend.brain as brain
    import numpy as np

    mock_engine = MagicMock()
    mock_engine.is_stateful.return_value = False
    mock_engine.transcribe_chunk.return_value = "Hello"

    # Setup session
    session_id = "pulse_test_session"
    brain.backend_info["engine"] = mock_engine
    
    with caplog.at_level(logging.INFO):
        # 1. Send first chunk
        brain._handle_audio_chunk(session_id, 0, 0, b"\x00" * 3200)
        # 2. Commit session (needed for finalization)
        session = brain._get_or_create_session(session_id)
        rec = session.get_or_create_recording(0)
        rec.closed = True
        # 3. Finalize
        brain._finalize_recording_if_ready(session_id, 0)

    logs = [r.message for r in caplog.records if "[Brain]" in r.message]
    assert "[Brain] 🎙️  Recording started..." in logs[0]
    assert '[Brain] 🏁 "Hello" (' in logs[1]
    assert logs[1].endswith("s)")
```

- [ ] **Step 2: Run test to verify it fails**
  ```bash
  pytest tests/test_brain.py::test_brain_logs_match_pulse_format -v
  ```

- [ ] **Step 3: Implement changes in `brain.py`**
  Update `_handle_audio_chunk` and `_finalize_recording_if_ready` as described.

- [ ] **Step 4: Verify Brain tests**
  ```bash
  pytest tests/test_brain.py -v
  ```

---

## Task 3: Remove Noise and Update State Logs in `hud.py`

**Files:**
- Modify: `src/ui/hud.py`

**What changes and why:**
1. Remove `log.info` from `_on_volume()` and `_on_frequency_bands()`.
2. Remove noise from `_on_command()` (the `← listen` noise).
3. Remove the DEBUG log block and dead `pass` from `_tick()`.
4. Update `show_listening()` to log `[HUD] → Listening`.
5. Update `show_done()` to log `[HUD] → Done`.

- [ ] **Step 1: Write the failing test for HUD logs**

  Update/Create `tests/test_hud_logging.py`:

```python
import logging
from unittest.mock import MagicMock, patch
import pytest

def test_hud_logs_follow_pulse_format(caplog):
    """
    Verify HUD logs only state changes in the arrow format.
    """
    # Mock dependencies as in existing plan
    with patch("src.ui.hud.QTimer"), \
         patch("src.ui.hud.IPCServer"), \
         patch("src.ui.hud.VolumeListener"), \
         patch("src.ui.hud.NSStatusBar"), \
         patch("src.ui.hud.MenuBarWaveformView"):
        import src.ui.hud as hud
        controller = hud.MenuBarWaveformController.__new__(hud.MenuBarWaveformController)
        controller._snd_listen = "dummy.mp3"
        controller._snd_done = "dummy.wav"
        
        with caplog.at_level(logging.INFO):
            controller.show_listening()
            controller.show_done()
            controller._on_volume(0.5) # Should be silent
            
        logs = [r.message for r in caplog.records if "[HUD]" in r.message]
        assert "[HUD]   → Listening" in logs
        assert "[HUD]   → Done" in logs
        assert not any("Received volume" in l for l in logs)
```

- [ ] **Step 2: Implement changes in `hud.py`**
  Silence noise and update the state transition logs.

- [ ] **Step 3: Verify HUD tests**
  ```bash
  pytest tests/test_hud_logging.py -v
  ```

---

## Task 4: Silence Capture Noise in `ear.py`

**Files:**
- Modify: `src/audio/ear.py`

**What changes and why:**
1. Remove `log.info("📤 Audio chunk sent", ...)` from `_send_audio_chunk_to_brain()`. This fires 60 times/second and is redundant since the Brain now logs when recording starts.

- [ ] **Step 1: Implement the change**
  Remove line 546 in `src/audio/ear.py`.

- [ ] **Step 2: Verify Ear functionality**
  ```bash
  pytest tests/test_ear_recording_mode.py -v
  ```

---

## Task 5: Final Verification

- [ ] **Step 1: Run the full test suite**
  ```bash
  pytest tests/ -v --timeout=30 --ignore=tests/test_hud_menu_bar.py --ignore=tests/test_theme_manager.py
  ```

- [ ] **Step 2: Run the startup verification script**
  ```bash
  source .venv/bin/activate && python feedback-loop/verify_startup.py
  ```

- [ ] **Step 3: Final commit**
  ```bash
  git add -A
  git commit -m "simplify: implement Variant 2: The Pulse logging across Ear, Brain, and HUD"
  ```
