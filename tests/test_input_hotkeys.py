# tests/test_input_hotkeys.py
from unittest.mock import MagicMock, patch

from typer.cli import callback

from src.input.hotkeys import InputTrigger, RecordingCallbacks, is_rcmd


def test_is_right_cmd_detects_various_formats():
    """
    Verifies that the _is_rcmd function correctly identifies
    different ways pynput represents the Right Command key.
    """

    class MockKey:
        name = "cmd_r"

    assert is_rcmd(MockKey()) is True
    assert is_rcmd("wrong_key") is False


def _test_helper():
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_toggle = MagicMock()

    callbacks = RecordingCallbacks(on_start=mock_start, on_stop=mock_stop, on_toggle=mock_toggle)

    trigger = InputTrigger(callbacks)

    class MockKey:
        name = "cmd_r"

    return trigger, MockKey, mock_start, mock_stop, mock_toggle


def test_input_trigger_handles_double_tap_toggle():
    """
    Checks that tapping the hotkey twice quickly triggers the 'toggle' mode.
    """

    trigger, MockKey, mock_start, _, mock_toggle = _test_helper()
    # First Tap: Press at 1.0, Release at 1.1
    trigger.key_press(MockKey(), current_time=1.0)
    # Timer should be started but not fired.
    assert trigger.key.timer is not None
    mock_start.assert_not_called()

    trigger.key_release(MockKey(), current_time=1.1)
    # Timer should be cancelled.
    assert trigger.key.timer is None
    mock_toggle.assert_not_called()

    # Second Tap: Press at 1.2 (difference from last release is 0.1s <= 0.3s)
    trigger.key_press(MockKey(), current_time=1.2)
    mock_toggle.assert_called_once()
    assert trigger.key.toggle_active is True


def test_input_trigger_handles_long_hold_stop():
    """
    Checks that holding the hotkey triggers recording after the threshold,
    and stops it on release.
    """
    trigger, MockKey, mock_start, mock_stop, _ = _test_helper()
    # Simulate press
    trigger.key_press(MockKey(), current_time=1.0)

    # Manually fire the timer to simulate 0.3s passing
    trigger.trigger_hold()

    mock_start.assert_called_once_with()
    assert trigger.key.rec_hold is True

    # Simulate slow release
    trigger.key_release(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)
    assert trigger.key.rec_hold is False


def test_input_trigger_ignores_auto_repeat():
    """
    Verifies that multiple press events (auto-repeat) do not trigger multiple toggles or timers.
    """
    trigger, MockKey, _, _, _ = _test_helper()
    # 1. First Press
    trigger.key_press(MockKey(), current_time=1.0)
    assert trigger.key.timer is not None
    first_timer = trigger.key.timer

    # 2. Noise from OS (Auto-repeat events)
    trigger.key_press(MockKey(), current_time=1.2)

    # The timer should NOT have changed (ignored repeat)
    assert trigger.key.timer is first_timer


def test_input_trigger_toggle_off_behavior():
    """
    Checks that pressing the key while in Toggle Mode correctly stops the recording.
    """
    trigger, MockKey, _, mock_stop, _ = _test_helper()
    # Manually enter toggle mode
    trigger.key.toggle_active = True

    # Step 2: Press again to stop
    trigger.key_press(MockKey(), current_time=2.0)
    mock_stop.assert_called_once_with(stop_session=True)
    assert trigger.key.toggle_active is False


def test_input_trigger_mouse_hold_logic():
    """
    Verifies that holding the right mouse button for 1 second triggers recording.
    """
    import src.input.hotkeys as hotkeys_module

    # Use the actual button value from the hotkeys module so the comparison
    # inside _handle_mouse_click works regardless of pynput version.
    right_button = getattr(hotkeys_module.mouse.Button, "right", "right")
    trigger, _, mock_start, mock_stop, _ = _test_helper()
    with patch("src.input.hotkeys.time.time", return_value=10.0):
        trigger.mouse_click(0, 0, right_button, pressed=True)

    with patch("src.input.hotkeys.time.time", return_value=10.5):
        assert trigger.check_mouse_hold() is False
        mock_start.assert_not_called()

    with patch("src.input.hotkeys.time.time", return_value=11.1):
        assert trigger.check_mouse_hold() is True
        mock_start.assert_called_once_with()

    trigger.mouse_click(0, 0, right_button, pressed=False)
    mock_stop.assert_called_once_with(stop_session=True)


def test_input_trigger_start_listening_tolerates_missing_listener_methods(monkeypatch):
    trigger, _, _, _, _ = _test_helper()
    monkeypatch.setattr(
        "src.input.hotkeys.keyboard.Listener", lambda *args, **kwargs: None, raising=False
    )
    monkeypatch.setattr(
        "src.input.hotkeys.mouse.Listener", lambda *args, **kwargs: None, raising=False
    )

    trigger.start()
    trigger.stop()


def test_new_short_names_exist():

    assert is_rcmd is not None
    callbacks = RecordingCallbacks(on_start=None, on_stop=None, on_toggle=None)
    trigger = InputTrigger(callbacks)
    assert hasattr(trigger, "start")
    assert hasattr(trigger, "check_mouse_hold")
