# tests/test_input_hotkeys.py
import pytest
import time
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

def test_input_trigger_handles_short_press_toggle():
    """
    Checks that a quick tap of the hotkey (shorter than the threshold)
    triggers the 'toggle' mode.
    """
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
    
    # Simulate fast release
    trigger._handle_key_release(MockKey(), current_time=1.1)
    mock_toggle.assert_called_once()
    mock_stop.assert_not_called()

def test_input_trigger_handles_long_hold_stop():
    """
    Checks that holding the hotkey for a long time causes it to stop 
    recording immediately when released.
    """
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
    
    # Simulate slow release (1.0s later)
    trigger._handle_key_release(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)

def test_input_trigger_ignores_auto_repeat():
    """
    INDUSTRY STANDARD TEST: Verifies that multiple press events 
    (auto-repeat) do not reset the start timer or trigger multiple starts.
    """
    mock_start = MagicMock()
    mock_stop = MagicMock()
    
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=MagicMock(),
        hold_threshold_seconds=0.4
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # 1. First Press
    trigger._handle_key_press(MockKey(), current_time=1.0)
    mock_start.assert_called_once()
    
    # 2. Noise from OS (Auto-repeat events)
    trigger._handle_key_press(MockKey(), current_time=1.2)
    trigger._handle_key_press(MockKey(), current_time=1.4)
    
    # 3. Release after a long time (from the FIRST press)
    trigger._handle_key_release(MockKey(), current_time=2.0)
    
    # If it works, mock_stop should be called because 2.0 - 1.0 = 1.0s (> 0.4s).
    # If it fails (timer reset), it would think duration is 2.0 - 1.4 = 0.6s, 
    # but wait, that's still > 0.4s. Let's make a tighter test.
    
    # Reset mocks for a tighter duration test
    mock_start.reset_mock()
    mock_stop.reset_mock()
    
    # Press at 1.0
    trigger._handle_key_press(MockKey(), current_time=1.0)
    # Auto-repeat at 1.5 (This would reset the timer to 1.5 if buggy)
    trigger._handle_key_press(MockKey(), current_time=1.5)
    # Release at 1.7
    trigger._handle_key_release(MockKey(), current_time=1.7)
    
    # Total physical time: 1.7 - 1.0 = 0.7s (Long Press)
    # Buggy time (if reset): 1.7 - 1.5 = 0.2s (Short Tap)
    
    # It SHOULD be a long press stop call
    mock_stop.assert_called_once_with(stop_session=True)

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
        
    # Step 1: Tap to enter toggle mode
    trigger._handle_key_press(MockKey(), current_time=1.0)
    trigger._handle_key_release(MockKey(), current_time=1.1)
    assert trigger._is_toggle_mode_active is True
    
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
