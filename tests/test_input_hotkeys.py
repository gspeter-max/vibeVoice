# tests/test_input_hotkeys.py
import pytest
import time
from unittest.mock import MagicMock
from src.input.hotkeys import InputTrigger, _is_right_cmd

def test_is_right_cmd_detects_various_formats():
    """
    Verifies that the _is_right_cmd function correctly identifies 
    different ways pynput represents the Right Command key.
    """
    # Create a mock object that mimics a pynput key with a 'name' attribute
    class MockKey:
        name = 'cmd_r'
    
    # We check if it returns True for 'cmd_r' and False for anything else
    assert _is_right_cmd(MockKey()) is True
    assert _is_right_cmd("wrong_key") is False

def test_input_trigger_handles_short_press_toggle():
    """
    Checks that a quick tap of the hotkey (shorter than the threshold)
    triggers the 'toggle' mode instead of a simple stop.
    """
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()
    
    # Initialize the trigger with our mocks and a 0.4s threshold
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=mock_toggle,
        hold_threshold_seconds=0.4
    )
    
    class MockKey:
        name = 'cmd_r'
        
    # Simulate a key press event at time 1.0
    trigger._handle_key_press(MockKey(), current_time=1.0)
    # The start recording callback should be called immediately
    mock_start.assert_called_once_with(from_hold=False)
    
    # Simulate a fast release at time 1.1 (only 0.1s held)
    trigger._handle_key_release(MockKey(), current_time=1.1)
    # Since it was short, it should trigger the toggle callback
    mock_toggle.assert_called_once()
    # It should NOT trigger the stop callback yet
    mock_stop.assert_not_called()

def test_input_trigger_handles_long_hold_stop():
    """
    Checks that holding the hotkey for a long time (longer than threshold)
    causes it to stop recording immediately when released.
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
        
    # Simulate a key press event at time 1.0
    trigger._handle_key_press(MockKey(), current_time=1.0)
    mock_start.assert_called_once_with(from_hold=False)
    
    # Simulate a slow release at time 2.0 (1.0s held, which is > 0.4s)
    trigger._handle_key_release(MockKey(), current_time=2.0)
    # Since it was long, it should trigger the stop callback immediately
    mock_stop.assert_called_once_with(stop_session=True)
    # It should NOT trigger the toggle callback
    mock_toggle.assert_not_called()
