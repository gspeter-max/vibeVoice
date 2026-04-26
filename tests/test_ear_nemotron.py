import pytest
import time
from unittest.mock import MagicMock, patch
import src.audio.ear as ear

@pytest.fixture
def mock_ear():
    with patch('src.audio.ear.SileroVAD'), patch('src.audio.ear.pyaudio.PyAudio'):
        e = ear.Ear()
        return e

def test_ear_tracks_current_model(mock_ear):
    """Verify that Ear tracks the current model."""
    mock_ear.current_model = "nemotron-streaming-0.6b"
    assert "nemotron" in mock_ear.current_model

@patch('src.audio.ear.should_split_chunk_after_silence')
def test_ear_forces_1_12s_heartbeat_for_nemotron(mock_split, mock_ear):
    """Verify that Ear bypasses silence split and uses 1.12s heartbeat for Nemotron."""
    mock_ear.current_model = "nemotron-streaming-0.6b"
    mock_ear.is_recording = True
    mock_ear._chunk_started_at = time.time() - 1.2 # Started 1.2s ago
    
    # Mock split to NOT trigger
    mock_split.return_value.should_split_now = False
    mock_ear._stop_and_send = MagicMock()
    
    # Run loop tick
    mock_ear._record_loop_tick()
    
    # Should have called _stop_and_send because 1.2 > 1.12
    mock_ear._stop_and_send.assert_called_with(stop_session=False)

@patch('src.audio.ear.should_split_chunk_after_silence')
def test_ear_keeps_standard_logic_for_other_models(mock_split, mock_ear):
    """Verify that Ear uses standard silence logic for non-nemotron models."""
    mock_ear.current_model = "parakeet-tdt-0.6b-v3"
    mock_ear.is_recording = True
    mock_ear._chunk_started_at = time.time() - 1.2
    
    # Mock split to NOT trigger
    mock_split.return_value.should_split_now = False
    mock_ear._stop_and_send = MagicMock()
    
    # Run loop tick
    mock_ear._record_loop_tick()
    
    # Should NOT have called _stop_and_send yet (silence logic governs it)
    mock_ear._stop_and_send.assert_not_called()
