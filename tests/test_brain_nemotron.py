import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import src.brain as brain
from src.streaming.nemotron import NemotronStreamingBackend

def test_brain_routes_to_nemotron():
    """Verify that brain can load the nemotron backend."""
    # Mock the backend class where it is imported in brain.py
    # Since brain.py is in src/, and it does 'from streaming.nemotron import ...'
    with patch('streaming.nemotron.NemotronStreamingBackend') as MockBackend:
        mock_instance = MockBackend.return_value
        backend, model = brain.load_backend("nemotron-streaming-0.6b")
        assert backend == mock_instance
        assert model == mock_instance

@patch('src.brain._get_or_create_session')
@patch('src.brain._normalize_audio')
def test_handle_audio_chunk_uses_transcribe_chunk(mock_norm, mock_get_session):
    """Verify that _handle_audio_chunk uses transcribe_chunk for stateful backends."""
    # Ensure audio is not None
    mock_norm.return_value = np.zeros(1600)
    
    # Create a mock backend with the transcribe_chunk method
    class MockStatefulBackend:
        def transcribe_chunk(self, audio):
            return "cumulative text"
    
    mock_backend = MockStatefulBackend()
    mock_backend.transcribe_chunk = MagicMock(return_value="cumulative text")
    
    mock_session = MagicMock()
    mock_session.backend = mock_backend
    mock_session.model = mock_backend
    mock_get_session.return_value = mock_session
    
    # Mock recording state
    mock_rec = MagicMock()
    mock_rec.transcript_parts = {}
    mock_session.get_or_create_recording.return_value = mock_rec
    
    # Call the function
    brain._handle_audio_chunk("session1", 0, 0, b'\x00\x00' * 1600)
    
    # Check if transcribe_chunk was called
    mock_backend.transcribe_chunk.assert_called_once()
    
    # Check if transcript_parts was updated correctly (index 0)
    assert mock_rec.transcript_parts[0] == "cumulative text"

@patch('src.brain._get_or_create_session')
def test_mark_session_closed_resets_backend(mock_get_session):
    """Verify that _mark_session_closed resets stateful backends."""
    mock_backend = MagicMock()
    mock_backend.reset = MagicMock()
    
    mock_session = MagicMock()
    mock_session.backend = mock_backend
    mock_get_session.return_value = mock_session
    
    # Call the function
    brain._mark_session_closed("session1", 0)
    
    # Check if reset was called
    mock_backend.reset.assert_called_once()
