import pytest
import numpy as np

@pytest.fixture
def sample_audio_bytes():
    """Returns 1 second of 16kHz Mono 16-bit PCM audio (sine wave)."""
    rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    # Generate a sine wave and scale to 16-bit range
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    return audio_data.tobytes()

@pytest.fixture
def mock_socket_path(tmp_path):
    """Returns a temporary socket path for testing."""
    return str(tmp_path / "test_parakeet.sock")

import sys
sys.modules['pynput'] = type(sys)('pynput')
sys.modules['pynput.keyboard'] = type(sys)('pynput.keyboard')
sys.modules['pynput.mouse'] = type(sys)('pynput.mouse')

class MockButton:
    left = 'left'
    right = 'right'

sys.modules['pynput.mouse'].Button = MockButton()

class MockPyAudio:
    paInt16 = 16
    paContinue = 0
    class PyAudio:
        def get_default_input_device_info(self):
            return {"index": 0}
        def get_device_info_by_index(self, index):
            return {"name": "Test Device"}

sys.modules['pyaudio'] = MockPyAudio()

class MockConn:
    """
    A fake socket connection that returns pre-defined audio chunks
    and then returns an empty bytes object to signal the connection is closed.
    """
    def __init__(self, *chunks):
        self._chunks = list(chunks) + [b""]

    def settimeout(self, _timeout):
        return None

    def recv(self, _size):
        return self._chunks.pop(0)

    def close(self):
        return None
