import pytest
import sys
import os
from unittest.mock import Mock, patch

# Mock pynput before importing Ear
sys.modules['pynput'] = type(sys)('pynput')
sys.modules['pynput.keyboard'] = type(sys)('pynput.keyboard')
sys.modules['pynput.mouse'] = type(sys)('pynput.mouse')

# Add Button mock
class MockButton:
    left = 'left'
    right = 'right'

sys.modules['pynput.mouse'].Button = MockButton()

# Mock pyaudio
class MockPyAudio:
    paInt16 = 16
    class PyAudio:
        def get_default_input_device_info(self):
            return {"index": 0}
        def get_device_info_by_index(self, index):
            return {"name": "Test Device"}

sys.modules['pyaudio'] = MockPyAudio()

from src.ear import Ear

# Patch the _open_mic_stream method to avoid audio setup issues
@pytest.fixture(autouse=True)
def patch_open_mic_stream(monkeypatch):
    """Patch _open_mic_stream to avoid actual audio setup."""
    def dummy_open_stream(self):
        pass
    monkeypatch.setattr("src.ear.Ear._open_mic_stream", dummy_open_stream)

def test_ear_has_hold_state_variables():
    """Test that Ear initializes with hold-related state variables."""
    ear = Ear()

    # New hold state variables should exist
    assert hasattr(ear, '_mouse_press_start_time'), "Ear should have _mouse_press_start_time attribute"
    assert ear._mouse_press_start_time == 0.0, "_mouse_press_start_time should initialize to 0.0"

    assert hasattr(ear, '_is_holding'), "Ear should have _is_holding attribute"
    assert ear._is_holding is False, "_is_holding should initialize to False"

    assert hasattr(ear, '_recording_from_hold'), "Ear should have _recording_from_hold attribute"
    assert ear._recording_from_hold is False, "_recording_from_hold should initialize to False"

    # Old click state variables should NOT exist
    assert not hasattr(ear, '_mouse_click_count'), "Ear should NOT have _mouse_click_count attribute (removed)"
    assert not hasattr(ear, '_mouse_clicks_required'), "Ear should NOT have _mouse_clicks_required attribute (removed)"
    assert not hasattr(ear, '_mouse_click_timeout'), "Ear should NOT have _mouse_click_timeout attribute (removed)"

def test_mouse_press_starts_hold_timer():
    """Test that pressing mouse button starts hold timer."""
    ear = Ear()

    # Simulate mouse press
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=True)

    # Should set press time and holding flag
    assert ear._is_holding is True
    assert ear._mouse_press_start_time > 0
    assert ear._recording_from_hold is False


def test_mouse_release_stops_hold_timer():
    """Test that releasing mouse button clears holding flag."""
    ear = Ear()

    # Press
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=True)
    assert ear._is_holding is True

    # Release
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=False)
    assert ear._is_holding is False


def test_only_left_button_triggers_hold():
    """Test that only left mouse button triggers hold logic."""
    ear = Ear()

    # Right button press should be ignored
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)
    assert ear._is_holding is False

    # Left button press should work
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=True)
    assert ear._is_holding is True


def test_early_release_does_not_start_recording():
    """Test that releasing before 1 second does not start recording."""
    ear = Ear()

    # Press and immediately release (< 1 second)
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=True)
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=False)

    # Should not be recording
    assert ear.is_recording is False
    assert ear._recording_from_hold is False
