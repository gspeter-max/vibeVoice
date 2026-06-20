# tests/test_input_hotkeys.py
from unittest.mock import MagicMock, patch
from src.input.hotkeys import InputTrigger, _is_rcmd

def test_is_right_cmd_detects_various_formats():
    """
    Verifies that the _is_rcmd function correctly identifies 
    different ways pynput represents the Right Command key.
    """
    class MockKey:
        name = 'cmd_r'
    
    assert _is_rcmd(MockKey()) is True
    assert _is_rcmd("wrong_key") is False

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
    trigger._key_press(MockKey(), current_time=1.0)
    # Timer should be started but not fired.
    assert trigger._timer is not None
    mock_start.assert_not_called()
    
    trigger._key_release(MockKey(), current_time=1.1)
    # Timer should be cancelled.
    assert trigger._timer is None
    mock_toggle.assert_not_called()
    
    # Second Tap: Press at 1.2 (difference from last release is 0.1s <= 0.3s)
    trigger._key_press(MockKey(), current_time=1.2)
    mock_toggle.assert_called_once()
    assert trigger._toggle_active is True

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
    trigger._key_press(MockKey(), current_time=1.0)
    
    # Manually fire the timer to simulate 0.3s passing
    trigger._trigger_hold()
    
    mock_start.assert_called_once_with(from_hold=True)
    assert trigger._rec_hold is True
    
    # Simulate slow release
    trigger._key_release(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)
    assert trigger._rec_hold is False

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
    trigger._key_press(MockKey(), current_time=1.0)
    assert trigger._timer is not None
    first_timer = trigger._timer
    
    # 2. Noise from OS (Auto-repeat events)
    trigger._key_press(MockKey(), current_time=1.2)
    
    # The timer should NOT have changed (ignored repeat)
    assert trigger._timer is first_timer

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
    trigger._toggle_active = True
    
    # Step 2: Press again to stop
    trigger._key_press(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)
    assert trigger._toggle_active is False

def test_input_trigger_mouse_hold_logic():
    """
    Verifies that holding the right mouse button for 1 second triggers recording.
    """
    import src.input.hotkeys as hotkeys_module
    # Use the actual button value from the hotkeys module so the comparison
    # inside _handle_mouse_click works regardless of pynput version.
    right_button = getattr(hotkeys_module.mouse.Button, 'right', 'right')

    mock_start = MagicMock()
    mock_stop = MagicMock()
    trigger = InputTrigger(
        on_start_recording=mock_start,
        on_stop_recording=mock_stop,
        on_toggle_recording=MagicMock()
    )
    
    with patch('src.input.hotkeys.time.time', return_value=10.0):
        trigger._mouse_click(0, 0, right_button, pressed=True)
        
    with patch('src.input.hotkeys.time.time', return_value=10.5):
        assert trigger.check_mouse_hold() is False
        mock_start.assert_not_called()
        
    with patch('src.input.hotkeys.time.time', return_value=11.1):
        assert trigger.check_mouse_hold() is True
        mock_start.assert_called_once_with(from_hold=True)
        
    trigger._mouse_click(0, 0, right_button, pressed=False)
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

    trigger.start()
    trigger.stop()


def test_new_short_names_exist():
    from src.input.hotkeys import _is_rcmd, InputTrigger
    assert _is_rcmd is not None
    trigger = InputTrigger(None, None, None)
    assert hasattr(trigger, "start")
    assert hasattr(trigger, "check_mouse_hold")
