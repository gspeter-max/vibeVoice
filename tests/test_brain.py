import numpy as np
import pytest
import os
import socket
from unittest.mock import MagicMock, patch, call
from brain import start_server

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
    
    mock_server_socket = MagicMock()
    mock_socket_class.return_value = mock_server_socket
    
    # Simulate receiving audio data once and then raising KeyboardInterrupt to exit loop
    mock_conn = MockConn(sample_audio_bytes)
    mock_server_socket.accept.side_effect = [(mock_conn, None), KeyboardInterrupt]
    
    mock_backend.transcribe.return_value = "hello world"
    
    # Run server (should raise KeyboardInterrupt eventually)
    with patch('sys.stdout', new=MagicMock()): # suppress output
        start_server()
        
    # Verify load_backend was called
    mock_load_backend.assert_called_once()
    
    # Verify transcription was called with converted audio
    # audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
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
    
    mock_server_socket = MagicMock()
    mock_socket_class.return_value = mock_server_socket
    
    # Send a command instead of audio
    command = "CMD_SWITCH_MODEL:tiny.en"
    mock_conn = MockConn(command.encode('utf-8'))
    mock_server_socket.accept.side_effect = [(mock_conn, None), KeyboardInterrupt]
    
    with patch('sys.stdout', new=MagicMock()):
        start_server()
        
    # Verify load_model was called again for the switch
    mock_backend.load_model.assert_called_once_with("tiny.en")

@patch('brain.load_backend')
@patch('brain.keyboard')
@patch('socket.socket')
def test_brain_too_short_audio(mock_socket_class, mock_keyboard, mock_load_backend):
    """Test that brain skips too short audio clips."""
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_load_backend.return_value = (mock_backend, mock_model)
    
    mock_server_socket = MagicMock()
    mock_socket_class.return_value = mock_server_socket
    
    # 0.1s of audio is too short (less than 4800 samples)
    short_audio = b'\x00\x00' * 1600
    mock_conn = MockConn(short_audio)
    mock_server_socket.accept.side_effect = [(mock_conn, None), KeyboardInterrupt]
    
    with patch('sys.stdout', new=MagicMock()):
        start_server()
        
    # Transcribe should NOT be called
    mock_backend.transcribe.assert_not_called()
    mock_keyboard.type.assert_not_called()
