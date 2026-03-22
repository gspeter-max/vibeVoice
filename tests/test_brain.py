import numpy as np
import pytest
import os
import socket
from unittest.mock import MagicMock, patch, call
from brain import start_server, worker, audio_queue, backend_info

# Mock data and classes
class MockConn:
    def __init__(self, data):
        self.data = data
        self.sent = False

    def recv(self, size):
        if not self.sent:
            self.sent = True
            return self.data
        return b""

    def close(self):
        pass

@patch('brain.load_backend')
@patch('brain.keyboard')
@patch('socket.socket')
@patch('os.path.exists')
@patch('os.remove')
def test_brain_server_logic(mock_remove, mock_exists, mock_socket_class, mock_keyboard, mock_load_backend, sample_audio_bytes):
    """Test the brain's main server loop logic."""
    # Setup mocks
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_load_backend.return_value = (mock_backend, mock_model)
    
    # Initialize backend_info manually for the worker
    backend_info["backend"] = mock_backend
    backend_info["model"] = mock_model
    
    mock_backend.transcribe.return_value = "hello world"
    
    # Put item in queue and a stop signal
    audio_queue.put((sample_audio_bytes, False))
    audio_queue.put(None)
    
    # Run worker directly
    with patch('sys.stdout', new=MagicMock()), patch('brain.send_hud'):
        worker()
        
    # Verify transcription was called with converted audio
    mock_backend.transcribe.assert_called_once()
    args, _ = mock_backend.transcribe.call_args
    assert args[0] == mock_model
    assert isinstance(args[1], np.ndarray)
    assert args[1].dtype == np.float32
    
    # Verify keyboard typed the result
    mock_keyboard.type.assert_called_once_with("hello world ")

@patch('brain.load_backend')
@patch('brain.keyboard')
@patch('socket.socket')
def test_brain_switch_model_command(mock_socket_class, mock_keyboard, mock_load_backend):
    """Test the command to switch models."""
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_load_backend.return_value = (mock_backend, mock_model)
    
    backend_info["backend"] = mock_backend
    backend_info["model"] = mock_model
    
    # Send a command instead of audio
    command = "CMD_SWITCH_MODEL:tiny.en"
    audio_queue.put((command.encode('utf-8'), True))
    audio_queue.put(None)
    
    with patch('sys.stdout', new=MagicMock()), patch('brain.send_hud'):
        worker()
        
    # Verify load_backend was called again for the switch
    mock_load_backend.assert_called_once_with("tiny.en")

@patch('brain.load_backend')
@patch('brain.keyboard')
@patch('socket.socket')
def test_brain_too_short_audio(mock_socket_class, mock_keyboard, mock_load_backend):
    """Test that brain skips too short audio clips."""
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_load_backend.return_value = (mock_backend, mock_model)
    
    backend_info["backend"] = mock_backend
    backend_info["model"] = mock_model
    
    # 0.1s of audio is too short (less than 4800 samples)
    short_audio = b'\x00\x00' * 1600
    audio_queue.put((short_audio, False))
    audio_queue.put(None)
    
    with patch('sys.stdout', new=MagicMock()), patch('brain.send_hud'):
        worker()
        
    # Transcribe should NOT be called
    mock_backend.transcribe.assert_not_called()
    mock_keyboard.type.assert_not_called()
