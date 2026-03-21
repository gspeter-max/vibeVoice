import pytest
import numpy as np
import os

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
