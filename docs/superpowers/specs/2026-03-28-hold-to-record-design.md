# Hold-to-Record Feature Design Document

**Date:** 2026-03-28
**Author:** Claude Sonnet 4.6
**Status:** Design Phase

---

## Overview

Replace the current 4-click mouse activation system with a hold-based push-to-talk interaction model.

**Current Behavior:**
- User clicks 4 times within 1.5 seconds → Toggle recording
- Visual counter shows progress (dots 1/4, 2/4, 3/4, 4/4)

**New Behavior:**
- User presses and holds mouse button for 1 second → Start recording
- User continues holding while speaking
- User releases mouse button → Stop recording
- Early release (< 1 second) → Silent reset (no action)

---

## Requirements

### Functional Requirements
1. **Hold Duration:** 1.0 second delay before recording starts
2. **Mouse Button:** Left mouse button only
3. **Early Release:** If user releases before 1.0s, silently reset (no visual feedback)
4. **Visual Feedback:** No progress indicator during hold delay
5. **Recording Indicator:** Show "listening" state only when recording actually starts
6. **Keyboard Shortcut:** Keep Right CMD tap-to-toggle working alongside mouse hold

### Non-Functional Requirements
1. **Performance:** Hold detection must not block audio processing
2. **Thread Safety:** Must use existing lock patterns to avoid race conditions
3. **Backward Compatibility:** Right CMD shortcut must continue working
4. **Code Quality:** Remove all old click-counting code (no duplicates)

---

## Architecture

### Current State

```
ear.py:
  - on_mouse_click(pressed=True)
    → Count clicks (1/4, 2/4, 3/4, 4/4)
    → If count >= 4: Toggle recording

hud.py:
  - Show click counter dots (1/4, 2/4, 3/4)
  - Hide counter on 4th click
```

### New Design

```
ear.py:
  - on_mouse_click(pressed=True)
    → Record press time: _mouse_press_start_time
    → Set flag: _is_holding = True

  - record_loop() [runs every 50ms]
    → Check: if _is_holding AND (time - press_time >= 1.0s)
    → If NOT recording: Start recording
    → Set flag: _recording_from_hold = True

  - on_mouse_click(pressed=False)
    → Clear flag: _is_holding = False
    → If _recording_from_hold: Stop recording
    → Reset flags

hud.py:
  - Remove all click counter visualization
```

### Component Changes

**Removed Components:**
- Mouse click counting logic (`_mouse_click_count`, `_mouse_clicks_required`)
- Click timeout mechanism (`_mouse_click_timeout`)
- Visual counter dots on HUD
- "mouse_click:N" IPC commands

**New Components:**
- Press timestamp tracking (`_mouse_press_start_time`)
- Hold state flag (`_is_holding`)
- Recording source flag (`_recording_from_hold`)
- Hold duration check in `record_loop()`

**Unchanged Components:**
- Right CMD keyboard shortcut (`on_press()`, `on_release()`)
- Recording mechanism (`is_recording`, `_open_brain_stream()`, `_stop_and_send()`)
- Volume visualization
- HUD states (listening, thinking, processing, done)

---

## Data Flow

### Scenario 1: Successful Hold-to-Record

```
Time 0.0s: Mouse Button Pressed
  OS Event → pynput → on_mouse_click(pressed=True)
    ↓
  Set _mouse_press_start_time = time.time()
  Set _is_holding = True
    ↓
  [No visual feedback - silent wait]

Time 0.0-1.0s: User Holding
  record_loop() runs every 50ms
    ↓
  Check: time.time() - _mouse_press_start_time >= 1.0?
    ↓
  [Not yet - continue waiting]

Time 1.0s: Hold Duration Reached
  record_loop() check: TRUE (>= 1.0s)
    ↓
  Check: is_recording? NO
    ↓
  Action: Start Recording
    - _open_brain_stream()
    - is_recording = True
    - _recording_from_hold = True
    - Send "listen" to HUD
    ↓
  [User sees "listening" state with animated bars]

Time 1.0-10.0s: User Speaking
  Recording continues normally
  Volume bars animate on HUD
    ↓

Time 10.0s: Mouse Button Released
  OS Event → pynput → on_mouse_click(pressed=False)
    ↓
  Check: _recording_from_hold? YES
    ↓
  Action: Stop Recording
    - _stop_and_send()
    - is_recording = False
    - _recording_from_hold = False
    - _is_holding = False
    ↓
  Audio sent to brain for transcription
```

### Scenario 2: Early Release (Cancel)

```
Time 0.0s: Mouse Button Pressed
  on_mouse_click(pressed=True)
    ↓
  Set _mouse_press_start_time = time.time()
  Set _is_holding = True

Time 0.3s: Mouse Button Released (Too Early)
  on_mouse_click(pressed=False)
    ↓
  Check: _recording_from_hold? NO (never started)
    ↓
  Action: Silent Reset
    - _is_holding = False
    - _mouse_press_start_time = 0.0
    ↓
  [No visual feedback, no recording started]
```

### Scenario 3: Right CMD Keyboard Shortcut (Unchanged)

```
User taps Right CMD
  on_press() → Check recording state
    ↓
  If not recording: Start
  If recording: Stop
    ↓
  [Works exactly as before, independent of mouse system]
```

---

## Implementation Details

### File: `src/ear.py`

#### Changes to `__init__` method (lines 234-266)

**Remove:**
```python
self._mouse_click_count = 0
self._mouse_clicks_required = 4
self._mouse_click_timeout = 1.5
```

**Add:**
```python
self._mouse_press_start_time = 0.0
self._is_holding = False
self._recording_from_hold = False
```

#### Replace `on_mouse_click` method (lines 543-604)

**Old behavior:** Count clicks, trigger on 4th click

**New behavior:**
```python
def on_mouse_click(self, x, y, button, pressed):
    """Handle mouse press/release for hold-to-record.

    Press: Start hold timer
    Release: Stop recording (if active) or cancel hold
    """
    # Only left button
    if button != mouse.Button.left:
        return

    if pressed:
        # BUTTON PRESSED: Start hold timer
        self._mouse_press_start_time = time.time()
        self._is_holding = True
        # No visual feedback during hold delay
    else:
        # BUTTON RELEASED: Stop recording or cancel
        self._is_holding = False

        # Only stop if we started recording from this hold
        if self._recording_from_hold:
            with self._lock:
                if not self.is_recording:
                    return
            self._stop_and_send()
            self._recording_from_hold = False
            print("\r[Ear] ⏹️  Mouse released - STOPPING recording", flush=True)
```

#### Modify `record_loop` method (lines 623-632)

**Add hold duration check:**
```python
def record_loop(self):
    while True:
        time.sleep(0.05)
        with self._lock:
            recording = self.is_recording
            rms = self.last_rms

        # CHECK: Hold duration >= 1.0s → Start recording
        if self._is_holding and not recording:
            hold_duration = time.time() - self._mouse_press_start_time
            if hold_duration >= 1.0:
                # Start recording
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

        if recording:
            meter = "█" * min(int(rms * 500), 50)
            print(f"\r  Level: [{meter:<50}]", end="", flush=True)
```

### File: `src/hud.py`

#### Remove from `__init__` (lines 246-251)

**Delete:**
```python
# Mouse click counter visualization
self._mouse_click_count = 0
self._mouse_click_timeout = 1.5
self._last_mouse_click_time = 0.0
self._show_mouse_counter = False
self._mouse_counter_alpha = 0.0
```

#### Remove `_on_mouse_click` method (lines 345-369)

**Delete entire method.**

#### Remove `_draw_mouse_click_counter` method (lines 567-603)

**Delete entire method.**

#### Modify `_on_command` (lines 380-396)

**Remove "mouse_click:" handler:**
```python
def _on_command(self, cmd):
    c = cmd.strip().lower()
    print(f"[HUD] ← {c}", flush=True)
    if   c == "listen":    self.show_listening()
    elif c == "thinking":  self.show_thinking()
    elif c == "process":   self.show_processing()
    elif c == "done":      self.show_done()
    elif c == "hide":      self.hide_hud()
    # REMOVE: mouse_click: handler
    else:
        print(f"[HUD] Unknown command: {c}", flush=True)
```

#### Modify `paintEvent` (lines 499-565)

**Remove mouse counter rendering:**
```python
# Remove this section:
if self._show_mouse_counter and self._mouse_counter_alpha > 0.05:
    self._draw_mouse_click_counter(p, cx, cy)
```

#### Modify `_tick` (lines 418-497)

**Remove mouse counter timeout:**
```python
# Remove this section (lines 435-444):
if self._show_mouse_counter:
    time_since_click = time.time() - self._last_mouse_click_time
    if time_since_click > self._mouse_click_timeout:
        self._show_mouse_counter = False
        self._mouse_click_count = 0
        self._mouse_counter_alpha = 0.0
```

---

## Error Handling

### Edge Cases

1. **Mouse Disconnected Mid-Hold**
   - pynput may crash or stop receiving events
   - Current behavior: No recovery mechanism
   - New behavior: Same limitation (acceptable for v1)

2. **Brain Not Ready**
   - Hold reaches 1.0s, but brain not available
   - Action: Log error, reset `_is_holding = False`, show error message
   - User can retry hold

3. **Recording Active + Mouse Hold**
   - User holds mouse while already recording (e.g., from Right CMD)
   - Action: Ignore hold (don't start duplicate recording)
   - Check: `if self._is_holding and not recording:`

4. **Right CMD + Mouse Hold Together**
   - User presses Right CMD, then holds mouse button
   - Action: Both systems independent, no conflict
   - Right CMD sets `is_recording`, mouse hold checks `not recording`

5. **System Sleep During Hold**
   - User holds mouse, Mac sleeps, wakes up later
   - Action: Hold timer continues, likely exceeds 1.0s
   - Release after wake: Stops recording normally
   - Acceptable behavior

---

## Thread Safety

### Lock Usage

**Existing lock:** `self._lock` (already used for `is_recording`)

**New shared variables:**
- `_mouse_press_start_time` - Only written by mouse thread, read by main thread
- `_is_holding` - Written by mouse thread, read by main thread
- `_recording_from_hold` - Written by both, needs lock protection

**Lock strategy:**
```python
# Mouse thread (on_mouse_click):
with self._lock:
    self._recording_from_hold = False  # Protect with lock

# Main thread (record_loop):
with self._lock:
    if self._is_holding and not self.is_recording:
        # Check and modify protected state
```

**Race condition prevention:**
- `_mouse_press_start_time` and `_is_holding` are read-only in main thread
- Only `_recording_from_hold` needs lock protection (already have lock)
- No new locks needed

---

## Testing Strategy

### Manual Test Cases

1. **Basic Hold-to-Record**
   - Press and hold mouse button for 1.5 seconds
   - Verify: Recording starts after ~1 second
   - Release button
   - Verify: Recording stops, audio sent to brain

2. **Early Release**
   - Press and hold mouse button for 0.5 seconds
   - Release button
   - Verify: No recording starts, no error messages

3. **Right CMD Still Works**
   - Tap Right CMD
   - Verify: Recording starts
   - Tap Right CMD again
   - Verify: Recording stops

4. **Mixed Methods**
   - Start recording with Right CMD
   - Try mouse hold while recording
   - Verify: Mouse hold ignored (no duplicate recording)

5. **Brain Not Ready**
   - Hold mouse button for 1 second (brain not running)
   - Verify: Error message logged, hold resets

### Integration Test Points

- Mouse event detection (press/release)
- Timer accuracy (1.0 second hold duration)
- Recording state transitions
- IPC communication (ear → HUD)
- Audio streaming to brain

---

## Performance Considerations

### Timer Resolution

- `record_loop()` runs every 50ms (0.05s)
- Hold duration check accuracy: ±50ms
- **Actual delay:** 1.0-1.05 seconds
- **Human perception:** Negligible difference

### CPU Impact

- Additional check in `record_loop()`: One `if` statement
- Time complexity: O(1) per tick
- **Performance impact:** Negligible (< 1μs per tick)

### Memory Impact

- 3 new float/bool variables in `Ear` class
- **Memory increase:** ~24 bytes
- **Impact:** Negligible

---

## Rollback Plan

If issues arise:

1. **Git revert:** `git revert <commit-hash>`
2. **Restore old files:**
   - `src/ear.py` from backup
   - `src/hud.py` from backup
3. **No database migration needed** (in-memory state only)

---

## Future Enhancements (Out of Scope)

1. Configurable hold duration (user preference)
2. Visual progress indicator during hold delay
3. Mouse button selection (left/middle/right)
4. Hold-to-record sensitivity adjustment
5. Haptic feedback on recording start/stop

---

## Approval Checklist

- [ ] Architecture reviewed
- [ ] Data flow understood
- [ ] Thread safety verified
- [ ] Error handling covered
- [ ] Testing strategy defined
- [ ] Performance impact acceptable
- [ ] Rollback plan clear

**Sign-off:** User approval required before implementation
