import pytest
import sys
import os
from unittest.mock import Mock, patch

# Mock pynput before importing Ear
sys.modules['pynput'] = type(sys)('pynput')
sys.modules['pynput.keyboard'] = type(sys)('pynput.keyboard')
sys.modules['pynput.mouse'] = type(sys)('pynput.mouse')

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
