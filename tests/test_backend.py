import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import backend_faster_whisper

def test_backend_load_model():
    """Test loading the faster-whisper model (mocked)."""
    with patch('backend_faster_whisper.WhisperModel') as mock_whisper_class:
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        
        # Test default load
        model = backend_faster_whisper.load_model()
        
        assert mock_whisper_class.called
        assert model == mock_model
        # Check some default args
        args, kwargs = mock_whisper_class.call_args
        assert args[0] == "base.en"
        assert kwargs['device'] == "cpu"

def test_backend_transcribe():
    """Test the transcribe function (mocked)."""
    mock_model = MagicMock()
    # Mock return value of model.transcribe which returns (segments, info)
    mock_segment = MagicMock()
    mock_segment.text = "hello"
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())
    
    audio_data = np.zeros(16000, dtype=np.float32)
    text = backend_faster_whisper.transcribe(mock_model, audio_data)
    
    assert text == "hello"
    mock_model.transcribe.assert_called_once()
    args, kwargs = mock_model.transcribe.call_args
    assert np.array_equal(args[0], audio_data)
