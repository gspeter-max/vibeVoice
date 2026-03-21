import math
import struct
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from ear import get_rms, Ear

def test_get_rms_silence():
    """Test get_rms with absolute silence."""
    silence = b'\x00\x00' * 1024
    assert get_rms(silence) == 0.0

def test_get_rms_full_scale():
    """Test get_rms with full scale samples."""
    # All samples at max positive 16-bit value (32767)
    full_scale = struct.pack('<h', 32767) * 1024
    rms = get_rms(full_scale)
    # 32767/32768 is ~1.0, so RMS should be ~1.0
    assert 0.99 < rms <= 1.0

def test_get_rms_sine(sample_audio_bytes):
    """Test get_rms with a 440Hz sine wave."""
    rms = get_rms(sample_audio_bytes)
    # RMS of a sine wave is peak / sqrt(2)
    # Here peak is 32767/32768 ~ 1.0, so RMS ~ 0.707
    assert 0.70 < rms < 0.71

def test_get_rms_empty():
    """Test get_rms with empty input."""
    assert get_rms(b'') == 0.0

@patch('pyaudio.PyAudio')
def test_ear_init(mock_pyaudio):
    """Test Ear class initialization."""
    ear = Ear()
    assert mock_pyaudio.called
    assert not ear.is_recording
    assert len(ear.frames) == 0

@patch('pyaudio.PyAudio')
@patch('socket.socket')
def test_ear_recording_flow(mock_socket, mock_pyaudio, sample_audio_bytes):
    """Test the recording start/stop and sending to brain flow."""
    ear = Ear()
    mock_stream = MagicMock()
    ear.p.open.return_value = mock_stream
    
    # Simulate pressing the hotkey
    from pynput.keyboard import Key
    ear.on_press(Key.cmd_r)
    
    assert ear.is_recording
    assert ear.p.open.called
    
    # Simulate some audio frames coming in via the callback
    # The callback is called by PyAudio (mocked here)
    ear._audio_callback(sample_audio_bytes, 1024, None, None)
    assert len(ear.frames) == 1
    
    # Simulate releasing the hotkey
    with patch('threading.Thread') as mock_thread:
        ear.on_release(Key.cmd_r)
        assert not ear.is_recording
        assert mock_stream.stop_stream.called
        assert mock_stream.close.called
        # Verify it attempts to send to brain
        mock_thread.assert_called_once()
        # The target should be ear._send_to_brain
        assert mock_thread.call_args[1]['target'] == ear._send_to_brain
