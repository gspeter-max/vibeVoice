# tests/test_input_hotkeys.py
import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from src.input.hotkeys import InputTrigger, _is_right_cmd

def test_is_right_cmd_detects_various_formats():
    """
    Verifies that the _is_right_cmd function correctly identifies 
    different ways pynput represents the Right Command key.
    """
    class MockKey:
        name = 'cmd_r'
    
    assert _is_right_cmd(MockKey()) is True
    assert _is_right_cmd("wrong_key") is False

def test_input_trigger_handles_double_tap_toggle():
    """
    Checks that tapping the hotkey twice quickly triggers the 'toggle' mode.
    """
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()
    
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=mock_toggle,
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # First Tap: Press at 1.0, Release at 1.1
    trigger._handle_key_press(MockKey(), current_time=1.0)
    # Timer should be started but not fired.
    assert trigger._delayed_start_timer is not None
    mock_start.assert_not_called()
    
    trigger._handle_key_release(MockKey(), current_time=1.1)
    # Timer should be cancelled.
    assert trigger._delayed_start_timer is None
    mock_toggle.assert_not_called()
    
    # Second Tap: Press at 1.2 (difference from last release is 0.1s <= 0.3s)
    trigger._handle_key_press(MockKey(), current_time=1.2)
    mock_toggle.assert_called_once()
    assert trigger._is_toggle_mode_active is True

def test_input_trigger_handles_long_hold_stop():
    """
    Checks that holding the hotkey triggers recording after the threshold,
    and stops it on release.
    """
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()
    
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=mock_toggle,
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # Simulate press
    trigger._handle_key_press(MockKey(), current_time=1.0)
    
    # Manually fire the timer to simulate 0.3s passing
    trigger._trigger_hold_recording()
    
    mock_start.assert_called_once_with(from_hold=True)
    assert trigger._is_recording_due_to_hold is True
    
    # Simulate slow release
    trigger._handle_key_release(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)
    assert trigger._is_recording_due_to_hold is False

def test_input_trigger_ignores_auto_repeat():
    """
    Verifies that multiple press events (auto-repeat) do not trigger multiple toggles or timers.
    """
    mock_start = MagicMock()
    mock_stop = MagicMock()
    
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=MagicMock(),
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # 1. First Press
    trigger._handle_key_press(MockKey(), current_time=1.0)
    assert trigger._delayed_start_timer is not None
    first_timer = trigger._delayed_start_timer
    
    # 2. Noise from OS (Auto-repeat events)
    trigger._handle_key_press(MockKey(), current_time=1.2)
    
    # The timer should NOT have changed (ignored repeat)
    assert trigger._delayed_start_timer is first_timer

def test_input_trigger_toggle_off_behavior():
    """
    Checks that pressing the key while in Toggle Mode correctly stops the recording.
    """
    mock_stop = MagicMock()
    trigger = InputTrigger(
        on_start_recording=MagicMock(),
        on_stop_recording=mock_stop,
        on_toggle_recording=MagicMock()
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # Manually enter toggle mode
    trigger._is_toggle_mode_active = True
    
    # Step 2: Press again to stop
    trigger._handle_key_press(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)
    assert trigger._is_toggle_mode_active is False

def test_input_trigger_mouse_hold_logic():
    """
    Verifies that holding the right mouse button for 1 second triggers recording.
    """
    mock_start = MagicMock()
    mock_stop = MagicMock()
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=MagicMock()
    )
    
    # Simulate Right Mouse Press
    with patch('time.time', return_value=10.0):
        trigger._handle_mouse_click(0, 0, 'right', pressed=True)
        
    # Check threshold at 10.5s (not long enough)
    with patch('time.time', return_value=10.5):
        assert trigger.check_mouse_hold_threshold() is False
        mock_start.assert_not_called()
        
    # Check threshold at 11.1s (long enough!)
    with patch('time.time', return_value=11.1):
        assert trigger.check_mouse_hold_threshold() is True
        mock_start.assert_called_once_with(from_hold=True)
        
    # Release mouse
    trigger._handle_mouse_click(0, 0, 'right', pressed=False)
    mock_stop.assert_called_once_with(stop_session=True)

def test_input_trigger_start_listening_tolerates_missing_listener_methods(monkeypatch):
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()

    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=mock_toggle,
    )

    monkeypatch.setattr("src.input.hotkeys.keyboard.Listener", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr("src.input.hotkeys.mouse.Listener", lambda *args, **kwargs: None, raising=False)

    trigger.start_listening()
    trigger.stop_listening()
