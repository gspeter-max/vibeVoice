import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import src.brain as brain
from src.streaming.nemotron import NemotronStreamingEngine

def test_brain_routes_to_nemotron():
    """Verify that brain can load the nemotron engine."""
    with patch('streaming.nemotron.NemotronStreamingEngine') as MockEngine:
        mock_instance = MockEngine.return_value
        backend, model = brain.load_transcription_engine("nemotron-streaming-0.6b")
        assert backend == mock_instance
        assert model == mock_instance

@patch('src.brain._get_or_create_session')
@patch('src.brain._normalize_audio')
def test_handle_audio_chunk_uses_add_audio_method(mock_norm, mock_get_session):
    """Verify that _handle_audio_chunk uses add_audio_chunk_and_get_text for stateful engines."""
    mock_norm.return_value = np.zeros(1600)
    
    class MockStatefulEngine:
        def add_audio_chunk_and_get_text(self, audio):
            return "cumulative text"
    
    mock_engine = MockStatefulEngine()
    mock_engine.add_audio_chunk_and_get_text = MagicMock(return_value="cumulative text")
    
    mock_session = MagicMock()
    mock_session.backend = mock_engine
    mock_session.model = mock_engine
    mock_get_session.return_value = mock_session
    
    mock_rec = MagicMock()
    mock_rec.transcript_parts = {}
    mock_session.get_or_create_recording.return_value = mock_rec
    
    brain._handle_audio_chunk("session1", 0, 0, b'\x00\x00' * 1600)
    
    mock_engine.add_audio_chunk_and_get_text.assert_called_once()
    assert mock_rec.transcript_parts[0] == "cumulative text"

@patch('src.brain._get_or_create_session')
@patch('src.brain.send_hud')
@patch('src.brain.paste_instantly')
def test_finalize_recording_if_ready_clears_memory(mock_paste, mock_hud, mock_get_session):
    """Verify that _finalize_recording_if_ready clears stateful engine memory."""
    mock_engine = MagicMock()
    mock_engine.clear_internal_memory = MagicMock()
    
    mock_session = MagicMock()
    mock_session.backend = mock_engine
    
    # We must properly mock session_store since _finalize_recording_if_ready uses it directly
    with patch.dict('src.brain.session_store', {"session1": mock_session}):
        mock_rec = MagicMock()
        mock_rec.finalized = False
        mock_rec.closed = True
        mock_rec.done_count = 1
        mock_rec.received_count = 1
        mock_rec.stt_time = 0.5
        mock_rec.transcript_parts = {0: "final text"}
        mock_session.recordings = {0: mock_rec}
        
        brain._finalize_recording_if_ready("session1", 0)
        
    mock_engine.clear_internal_memory.assert_called_once()
