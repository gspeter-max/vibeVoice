# Input Hotkeys Extraction Implementation Plan (Phase 1: Construction Only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent** : 
- We are building a new dedicated module `src/input/hotkeys.py` to handle all keyboard and mouse OS-level listening.
- **CRITICAL ARCHITECTURE RULE:** You are operating in a parallel worktree. **DO NOT MODIFY `src/audio/ear.py` OR `src/backend/brain.py`.** Your job is ONLY to create the new module and its tests.
- Currently, `ear.py` imports `pynput` and creates a `keyboard.Listener` and `mouse.Listener`. It hardcodes the `Right CMD` and `Right Mouse Button` logic.
- We want to build a clean `InputTrigger` class that wraps `pynput` and emits simple callbacks (e.g. `on_start_recording`, `on_stop_recording`, `on_toggle_recording`).
- **CRITICAL:** Do not assume anything, strictly follow the plan, and ask questions if you don't understand anything.

**Architecture:**
- Create `src/input/hotkeys.py`.
- Define an `InputTrigger` class.
- It will accept callback functions in its `__init__`.
- It will encapsulate the `pynput` listeners and the logic for short-press vs long-hold (using the `0.4` second threshold currently in `ear.py`).

**Important Rule to follow :**
- **CRITICAL: ** add detailed docs in functions and explain the code and logic in comments.  
- **CRITICAL:** make the code function name and variable name clear and easily to understand instead of short and confusing names.
- write code function name and docs and code like this: **developer gets highest speed to read the code**
- **Explain like a fresher** 

---
## Task Structure

### Task 1 : Read out instruction file
- [ ] read `/Users/apple/.gemini/GEMINI.md` file

### Task 2: The Hotkeys Module

**Files:**
- Create: `src/input/__init__.py`
- Create: `src/input/hotkeys.py`
- Create: `tests/test_input_hotkeys.py`

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_input_hotkeys.py
import pytest
import time
from unittest.mock import MagicMock
from src.input.hotkeys import InputTrigger, _is_right_cmd

def test_is_right_cmd_detects_various_formats():
    # Test dictionary-like or object-like pynput events
    class MockKey:
        name = 'cmd_r'
    
    assert _is_right_cmd(MockKey()) is True
    assert _is_right_cmd("wrong_key") is False

def test_input_trigger_handles_short_press_toggle():
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()
    
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=mock_toggle,
        hold_threshold_seconds=0.4
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # Simulate press
    trigger._handle_key_press(MockKey(), current_time=1.0)
    mock_start.assert_called_once_with(from_hold=False)
    
    # Simulate fast release (toggle mode)
    trigger._handle_key_release(MockKey(), current_time=1.1)
    mock_toggle.assert_called_once()
    mock_stop.assert_not_called()

def test_input_trigger_handles_long_hold_stop():
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()
    
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=mock_toggle,
        hold_threshold_seconds=0.4
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # Simulate press
    trigger._handle_key_press(MockKey(), current_time=1.0)
    mock_start.assert_called_once_with(from_hold=False)
    
    # Simulate slow release (hold mode)
    trigger._handle_key_release(MockKey(), current_time=2.0) # 1.0s later
    mock_stop.assert_called_once_with(stop_session=True)
    mock_toggle.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_input_hotkeys.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation for __init__.py**
```python
# src/input/__init__.py
"""
The Input package.
Handles listening for keyboard and mouse events across the operating system.
"""
```

- [ ] **Step 4: Write implementation for hotkeys.py**
```python
# src/input/hotkeys.py
import time
from typing import Callable

try:
    from pynput import keyboard, mouse
except Exception:
    class _FallbackKey:
        cmd_r = "cmd_r"
    class _FallbackKeyboardModule:
        Key = _FallbackKey()
        Listener = lambda *args, **kwargs: None
    keyboard = _FallbackKeyboardModule()
    
    class _FallbackMouseButton:
        right = "right"
    class _FallbackMouseModule:
        Button = _FallbackMouseButton()
        Listener = lambda *args, **kwargs: None
    mouse = _FallbackMouseModule()

_RCMD_VK = 54

def _is_right_cmd(key) -> bool:
    """
    Checks if a keyboard event corresponds to the Right Command key.
    Works across different operating systems.
    """
    if key == getattr(keyboard.Key, 'cmd_r', None):
        return True
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == _RCMD_VK:
        return True
    return False

class InputTrigger:
    """
    Listens for Right Command and Right Mouse Button presses.
    It tells the rest of the application when to start or stop recording.
    """
    def __init__(
        self, 
        on_start_recording: Callable[[bool], None], 
        on_stop_recording: Callable[[bool], None],
        on_toggle_recording: Callable[[], None],
        hold_threshold_seconds: float = 0.4
    ):
        self._on_start_recording = on_start_recording
        self._on_stop_recording = on_stop_recording
        self._on_toggle_recording = on_toggle_recording
        self._hold_threshold = hold_threshold_seconds
        
        self._cmd_press_time = 0.0
        self._toggle_active = False
        
        self._mouse_press_start_time = 0.0
        self.is_mouse_holding = False
        self._recording_from_mouse_hold = False
        
        self._keyboard_listener = None
        self._mouse_listener = None

    def start_listening(self):
        """Starts the background threads that listen for keys."""
        self._keyboard_listener = keyboard.Listener(
            on_press=lambda k: self._handle_key_press(k, time.time()), 
            on_release=lambda k: self._handle_key_release(k, time.time())
        )
        self._keyboard_listener.start()
        
        self._mouse_listener = mouse.Listener(
            on_click=self._handle_mouse_click
        )
        self._mouse_listener.start()

    def stop_listening(self):
        """Stops the background threads."""
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        if self._mouse_listener:
            self._mouse_listener.stop()

    def _handle_key_press(self, key, current_time: float):
        if not _is_right_cmd(key):
            return

        if self._toggle_active:
            self._toggle_active = False
            self._on_stop_recording(stop_session=True)
            return

        self._cmd_press_time = current_time
        self._on_start_recording(from_hold=False)

    def _handle_key_release(self, key, current_time: float):
        if not _is_right_cmd(key):
            return
            
        if self._toggle_active: 
            return

        time_held = current_time - self._cmd_press_time
        if time_held >= self._hold_threshold:
            # Long press: stop recording immediately
            self._on_stop_recording(stop_session=True)
        else:
            # Short tap: enter toggle mode (keep recording until next tap)
            self._toggle_active = True
            self._on_toggle_recording()

    def _handle_mouse_click(self, x, y, button, pressed):
        if button != getattr(mouse.Button, 'right', None):
            return
            
        if pressed:
            self._mouse_press_start_time = time.time()
            self.is_mouse_holding = True
        else:
            self.is_mouse_holding = False
            if self._recording_from_mouse_hold:
                self._on_stop_recording(stop_session=True)
                self._recording_from_mouse_hold = False
                
    def check_mouse_hold_threshold(self) -> bool:
        """
        Called continuously by the main loop. 
        Returns True if the mouse has been held long enough to start recording.
        """
        if self.is_mouse_holding and not self._recording_from_mouse_hold:
            if time.time() - self._mouse_press_start_time >= 1.0:
                self._recording_from_mouse_hold = True
                self._on_start_recording(from_hold=True)
                return True
        return False
```

- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/test_input_hotkeys.py -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/input/ tests/test_input_hotkeys.py
git commit -m "feat: add input hotkeys module for OS-level listening"
```
