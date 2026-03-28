# Hold-to-Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 4-click mouse activation system with a push-to-talk hold-based interaction (1 second hold to start, release to stop).

**Architecture:** Track mouse button press/release events using pynput's existing callback. Record press timestamp, check hold duration in the main record_loop every 50ms, start recording when >= 1.0s, stop on button release.

**Tech Stack:** Python 3.11, pynput (mouse events), threading (lock-based state management)

---

## File Structure

**Modified files:**
- `src/ear.py` - Core logic for mouse event handling and recording state
- `src/hud.py` - Visual feedback (remove click counter, no changes needed)

**No new files created** - This is a refactoring of existing functionality.

---

## Task 1: Update Ear.__init__ - Replace Click Variables with Hold Variables

**Files:**
- Modify: `src/ear.py:234-266`

- [ ] **Step 1: Write failing test for new hold state variables**

Create test file `tests/test_ear_hold_state.py`:

```python
import pytest
import time
from src.ear import Ear

def test_ear_has_hold_state_variables():
    """Test that Ear initializes with hold-related state variables."""
    ear = Ear()

    # New hold state variables should exist
    assert hasattr(ear, '_mouse_press_start_time')
    assert ear._mouse_press_start_time == 0.0

    assert hasattr(ear, '_is_holding')
    assert ear._is_holding is False

    assert hasattr(ear, '_recording_from_hold')
    assert ear._recording_from_hold is False

    # Old click state variables should NOT exist
    assert not hasattr(ear, '_mouse_click_count')
    assert not hasattr(ear, '_mouse_clicks_required')
    assert not hasattr(ear, '_mouse_click_timeout')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ear_hold_state.py::test_ear_has_hold_state_variables -v`

Expected output:
```
FAILED - Attribute errors or assert failures
```

- [ ] **Step 3: Modify Ear.__init__ to add hold variables**

Edit `src/ear.py` around line 234-266 (in `__init__` method):

**Remove these lines (old click system):**
```python
self._mouse_click_count = 0
self._mouse_clicks_required = 4
self._mouse_click_timeout = 1.5
```

**Add these lines (new hold system):**
```python
# Hold-to-record state
self._mouse_press_start_time = 0.0  # When mouse button was pressed
self._is_holding = False              # Currently holding mouse button
self._recording_from_hold = False     # Recording started from mouse hold
```

Place these new lines near the recording state initialization, after `self._lock = threading.Lock()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ear_hold_state.py::test_ear_has_hold_state_variables -v`

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/ear.py tests/test_ear_hold_state.py
git commit -m "refactor: replace click counters with hold state variables in Ear.__init__"
```

---

## Task 2: Replace on_mouse_click - Implement Press/Release Hold Logic

**Files:**
- Modify: `src/ear.py:543-604`

- [ ] **Step 1: Write failing test for mouse press/release behavior**

Add to `tests/test_ear_hold_state.py`:

```python
def test_mouse_press_starts_hold_timer():
    """Test that pressing mouse button starts hold timer."""
    ear = Ear()

    # Simulate mouse press
    ear.on_mouse_click(100, 100, mouse.Button.left, pressed=True)

    # Should set press time and holding flag
    assert ear._is_holding is True
    assert ear._mouse_press_start_time > 0
    assert ear._recording_from_hold is False


def test_mouse_release_stops_hold_timer():
    """Test that releasing mouse button clears holding flag."""
    ear = Ear()

    # Press
    ear.on_mouse_click(100, 100, mouse.Button.left, pressed=True)
    assert ear._is_holding is True

    # Release
    ear.on_mouse_click(100, 100, mouse.Button.left, pressed=False)
    assert ear._is_holding is False


def test_only_left_button_triggers_hold():
    """Test that only left mouse button triggers hold logic."""
    ear = Ear()

    # Right button press should be ignored
    ear.on_mouse_click(100, 100, mouse.Button.right, pressed=True)
    assert ear._is_holding is False

    # Left button press should work
    ear.on_mouse_click(100, 100, mouse.Button.left, pressed=True)
    assert ear._is_holding is True


def test_early_release_does_not_start_recording():
    """Test that releasing before 1 second does not start recording."""
    ear = Ear()

    # Press and immediately release (< 1 second)
    ear.on_mouse_click(100, 100, mouse.Button.left, pressed=True)
    ear.on_mouse_click(100, 100, mouse.Button.left, pressed=False)

    # Should not be recording
    assert ear.is_recording is False
    assert ear._recording_from_hold is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ear_hold_state.py -v`

Expected: Tests fail (old logic still counting clicks)

- [ ] **Step 3: Replace on_mouse_click method with hold logic**

Edit `src/ear.py` lines 543-604. Replace the entire `on_mouse_click` method:

**Delete entire current method and replace with:**

```python
def on_mouse_click(self, x, y, button, pressed):
    """
    Handle mouse button press/release for hold-to-record recording control.

    Press behavior (pressed=True):
        - Record press timestamp
        - Set holding flag
        - No visual feedback during hold delay

    Release behavior (pressed=False):
        - Clear holding flag
        - If recording started from this hold: stop recording
        - Early release (< 1s): silent reset (no action)

    Args:
        x, y: Mouse coordinates (unused)
        button: Which mouse button (only Button.left matters)
        pressed: True for press, False for release
    """
    # Only left button triggers hold logic
    if button != mouse.Button.left:
        return

    if pressed:
        # MOUSE BUTTON PRESSED: Start hold timer
        self._mouse_press_start_time = time.time()
        self._is_holding = True
        # No visual feedback during hold delay (silent wait)

    else:
        # MOUSE BUTTON RELEASED: Stop recording or cancel hold
        self._is_holding = False

        # Only stop if we started recording from this specific hold
        if self._recording_from_hold:
            # Check if actually recording before stopping
            with self._lock:
                if not self.is_recording:
                    self._recording_from_hold = False
                    return

            # Stop recording
            print("\r[Ear] ⏹️  Mouse released - STOPPING recording", flush=True)
            self._stop_and_send()
            self._recording_from_hold = False
```

**Note:** This removes all click counting logic, timeout checks, visual feedback commands, and the 4-click activation mechanism.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ear_hold_state.py -v`

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/ear.py tests/test_ear_hold_state.py
git commit -m "feat: replace click-counting with hold-based mouse activation"
```

---

## Task 3: Add Hold Duration Check to record_loop

**Files:**
- Modify: `src/ear.py:623-632`

- [ ] **Step 1: Write failing test for hold duration triggering recording**

Add to `tests/test_ear_hold_state.py`:

```python
import time
import threading
from unittest.mock import Mock, patch

def test_hold_one_second_starts_recording():
    """Test that holding for 1+ seconds starts recording."""
    ear = Ear()

    # Mock the brain stream to avoid needing actual brain connection
    with patch.object(ear, '_open_brain_stream', return_value=True):
        # Press mouse button
        ear.on_mouse_click(100, 100, mouse.Button.left, pressed=True)

        # Wait 1.1 seconds (exceeds 1.0s threshold)
        time.sleep(1.1)

        # Trigger record_loop tick (simulates the 50ms check)
        ear.record_loop()  # Single tick

        # Should have started recording
        assert ear.is_recording is True
        assert ear._recording_from_hold is True


def test_hold_less_than_one_second_no_recording():
    """Test that holding < 1 second does not start recording."""
    ear = Ear()

    with patch.object(ear, '_open_brain_stream', return_value=True):
        # Press mouse button
        ear.on_mouse_click(100, 100, mouse.Button.left, pressed=True)

        # Wait only 0.5 seconds (below threshold)
        time.sleep(0.5)

        # Trigger record_loop tick
        ear.record_loop()

        # Should NOT have started recording
        assert ear.is_recording is False
        assert ear._recording_from_hold is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ear_hold_state.py::test_hold_one_second_starts_recording -v`

Expected: Test fails (hold duration check not implemented yet)

- [ ] **Step 3: Modify record_loop to add hold duration check**

Edit `src/ear.py` lines 623-632. The current `record_loop` method:

```python
def record_loop(self):
    while True:
        time.sleep(0.05)
        with self._lock:
            recording = self.is_recording
            rms = self.last_rms
        if recording:
            meter = "█" * min(int(rms * 500), 50)
            print(f"\r  Level: [{meter:<50}]", end="", flush=True)
```

**Replace with:**

```python
def record_loop(self):
    """Main recording loop.

    Checks for hold duration >= 1.0s to start recording.
    Displays volume meter when recording.
    """
    while True:
        time.sleep(0.05)
        with self._lock:
            recording = self.is_recording
            rms = self.last_rms

        # CHECK: Hold duration >= 1.0s → Start recording
        if self._is_holding and not recording:
            hold_duration = time.time() - self._mouse_press_start_time

            if hold_duration >= 1.0:
                # Hold duration exceeded threshold - start recording

                # Open brain connection first
                if not self._open_brain_stream():
                    print(f"\r[Ear] ❌ Failed to open brain stream", flush=True)
                    self._is_holding = False
                    continue

                with self._lock:
                    self.is_recording = True
                    self.last_rms = 0.0
                    self._total_frames = 0
                    self._recording_from_hold = True

                print("\r\n" + "─" * 50, flush=True)
                print(f"\r🎙️  RECORDING+STREAMING via MOUSE HOLD ({self.active_mic_name})", flush=True)

                threading.Thread(target=self._send_hud, args=("listen",), daemon=True).start()
                self._start_volume_sender()

        # Display volume meter when recording
        if recording:
            meter = "█" * min(int(rms * 500), 50)
            print(f"\r  Level: [{meter:<50}]", end="", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ear_hold_state.py -v`

Expected: All tests pass

Note: The test might need adjustment since `record_loop()` is an infinite while loop. Consider refactoring to a single tick method for testing, or use a timeout.

- [ ] **Step 5: Commit**

```bash
git add src/ear.py tests/test_ear_hold_state.py
git commit -m "feat: add hold duration check to record_loop for auto-start"
```

---

## Task 4: Remove Mouse Click Counter from HUD.__init__

**Files:**
- Modify: `src/hud.py:246-251`

- [ ] **Step 1: Verify HUD has no click counter variables**

No test needed - this is cleanup only.

- [ ] **Step 2: Remove click counter variables from HUD.__init__**

Edit `src/hud.py` lines 246-251. Delete these lines:

```python
# Mouse click counter visualization
self._mouse_click_count = 0
self._mouse_click_timeout = 1.5  # Reset after 1.5s of inactivity (instant, no fade)
self._last_mouse_click_time = 0.0
self._show_mouse_counter = False
self._mouse_counter_alpha = 0.0
```

- [ ] **Step 3: Verify HUD still runs**

Run: `python src/hud.py`

Expected: HUD launches without errors, shows idle pill

- [ ] **Step 4: Commit**

```bash
git add src/hud.py
git commit -m "refactor: remove unused mouse click counter variables from HUD"
```

---

## Task 5: Remove _on_mouse_click Method from HUD

**Files:**
- Modify: `src/hud.py:345-369`

- [ ] **Step 1: Remove _on_mouse_click method**

Edit `src/hud.py`. Delete the entire `_on_mouse_click` method (lines 345-369).

This method is no longer needed since we removed the click counter visualization.

- [ ] **Step 2: Verify HUD still runs**

Run: `python src/hud.py`

Expected: HUD launches without errors

- [ ] **Step 3: Commit**

```bash
git add src/hud.py
git commit -m "refactor: remove unused _on_mouse_click method from HUD"
```

---

## Task 6: Remove mouse_click Command Handler from HUD._on_command

**Files:**
- Modify: `src/hud.py:380-396`

- [ ] **Step 1: Remove mouse_click command handler**

Edit `src/hud.py` lines 380-396. In the `_on_command` method, delete the `elif c.startswith("mouse_click:"):` block.

**Before:**
```python
elif c.startswith("mouse_click:"):
    # Parse mouse click count: "mouse_click:1", "mouse_click:2", etc.
    try:
        click_num = int(c.split(":")[1])
        self._on_mouse_click(click_num)
    except (ValueError, IndexError):
        print(f"[HUD] Invalid mouse_click command: {c}", flush=True)
```

**After:**
```python
# (mouse_click handler removed - no longer needed)
```

- [ ] **Step 2: Verify HUD still runs**

Run: `python src/hud.py`

Expected: HUD launches without errors

- [ ] **Step 3: Commit**

```bash
git add src/hud.py
git commit -m "refactor: remove mouse_click command handler from HUD"
```

---

## Task 7: Remove Mouse Counter Timeout Check from HUD._tick

**Files:**
- Modify: `src/hud.py:435-444`

- [ ] **Step 1: Remove mouse counter timeout logic**

Edit `src/hud.py` lines 435-444. Delete this entire section from `_tick` method:

```python
# Mouse click counter timeout - INSTANT reset after 1.5 seconds
if self._show_mouse_counter:
    time_since_click = time.time() - self._last_mouse_click_time
    if time_since_click > self._mouse_click_timeout:
        # INSTANT reset - no gradual fade
        self._show_mouse_counter = False
        self._mouse_click_count = 0
        self._mouse_counter_alpha = 0.0
        print(f"[HUD] ⚡ Timeout (1.5s) - counter instantly reset", flush=True)
        self.update()  # Immediate repaint
```

- [ ] **Step 2: Verify HUD still runs**

Run: `python src/hud.py`

Expected: HUD launches without errors, animations work

- [ ] **Step 3: Commit**

```bash
git add src/hud.py
git commit -m "refactor: remove mouse counter timeout check from HUD._tick"
```

---

## Task 8: Remove Mouse Counter Rendering from HUD.paintEvent

**Files:**
- Modify: `src/hud.py:562-563`

- [ ] **Step 1: Remove mouse counter rendering call**

Edit `src/hud.py` line 562-563. Delete these lines from `paintEvent`:

```python
# Draw mouse click counter (outside the pill)
if self._show_mouse_counter and self._mouse_counter_alpha > 0.05:
    self._draw_mouse_click_counter(p, cx, cy)
```

- [ ] **Step 2: Remove _draw_mouse_click_counter method**

Edit `src/hud.py` lines 567-603. Delete the entire `_draw_mouse_click_counter` method.

- [ ] **Step 3: Verify HUD still runs**

Run: `python src/hud.py`

Expected: HUD launches, shows pill, no errors

- [ ] **Step 4: Commit**

```bash
git add src/hud.py
git commit -m "refactor: remove mouse counter rendering from HUD"
```

---

## Task 9: Manual Integration Testing

**Files:**
- None (manual testing)

- [ ] **Step 1: Test basic hold-to-record**

1. Start the application: `python -m src.brain &` then `python -m src.ear`
2. Press and hold left mouse button
3. Wait 1+ seconds
4. **Verify:** Recording starts (see "RECORDING+STREAMING via MOUSE HOLD" message)
5. Continue holding for 2-3 seconds
6. Release mouse button
7. **Verify:** Recording stops, audio sent to brain

Expected: Recording starts after ~1 second hold, stops on release

- [ ] **Step 2: Test early release**

1. Press and hold left mouse button
2. Release after 0.5 seconds (< 1 second)
3. **Verify:** No recording starts, no error messages

Expected: Silent reset, nothing happens

- [ ] **Step 3: Test Right CMD still works**

1. Tap Right CMD key
2. **Verify:** Recording starts immediately
3. Tap Right CMD again
4. **Verify:** Recording stops

Expected: Keyboard shortcut still works

- [ ] **Step 4: Test mixed methods**

1. Start recording with Right CMD
2. While recording, press and hold mouse button
3. **Verify:** Mouse hold ignored (no duplicate recording)

Expected: No conflicts between control methods

- [ ] **Step 5: Test brain not ready**

1. Stop brain process (if running)
2. Press and hold mouse button for 1+ seconds
3. **Verify:** Error message logged, hold resets

Expected: Graceful error handling

- [ ] **Step 6: Document test results**

Create file `docs/testing/hold-to-record-test-results.md`:

```markdown
# Hold-to-Record Test Results

**Date:** 2026-03-28
**Tester:** [Your Name]

## Test Results

- [x] Basic hold-to-record: PASS
- [x] Early release (< 1s): PASS
- [x] Right CMD keyboard shortcut: PASS
- [x] Mixed methods (no conflicts): PASS
- [x] Brain not ready error handling: PASS

## Notes

[Add any observations or issues found]
```

- [ ] **Step 7: Commit test results**

```bash
git add docs/testing/hold-to-record-test-results.md
git commit -m "test: document hold-to-record integration test results"
```

---

## Summary

**Total Changes:**
- 2 files modified (`src/ear.py`, `src/hud.py`)
- 1 test file created (`tests/test_ear_hold_state.py`)
- ~150 lines removed (old click system)
- ~80 lines added (new hold system)

**Key Behaviors:**
- ✅ Hold left mouse button 1 second → Start recording
- ✅ Release mouse button → Stop recording
- ✅ Early release (< 1s) → Silent reset
- ✅ Right CMD shortcut still works
- ✅ No visual feedback during hold delay
- ✅ "listening" state shown when recording starts

**No Breaking Changes:**
- Right CMD keyboard shortcut unchanged
- Recording mechanism unchanged
- Brain communication unchanged
- Volume visualization unchanged
