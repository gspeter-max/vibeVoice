"""
test_brain_nemotron.py — Tests for the Nemotron (stateful) engine path in brain.py

After Phase 2 wiring, brain.py uses the TranscriptionEngine interface.
These tests verify that:
  - load_transcription_engine returns a single NemotronEngine for nemotron models
  - _handle_audio_chunk correctly calls engine.transcribe_chunk() for stateful engines
  - _finalize_recording_if_ready always calls engine.clear_internal_memory()
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import src.backend.brain as brain


def test_brain_routes_to_nemotron():
    """
    Verify that load_transcription_engine returns a single NemotronEngine object
    (not a tuple) when the model name contains 'nemotron'.
    """
    with patch("src.engines.nemotron.LegacyNemotron"):
        engine = brain.load_transcription_engine("nemotron-streaming-0.6b")

    # The result must be a single object with is_stateful() == True
    assert engine.is_stateful() is True


@patch.object(brain, "_get_or_create_session")
@patch.object(brain, "_normalize_audio")
def test_handle_audio_chunk_uses_transcribe_chunk_for_stateful_engine(mock_norm, mock_get_session):
    """
    Verify that _handle_audio_chunk calls engine.transcribe_chunk() for a stateful engine,
    and stores the full cumulative text in transcript_parts[0].
    """
    mock_norm.return_value = np.zeros(1600)

    # Create a mock engine that acts stateful
    mock_engine = MagicMock()
    mock_engine.is_stateful.return_value = True
    mock_engine.transcribe_chunk.return_value = "cumulative text"

    # Wire the engine into the mock session
    mock_session = MagicMock()
    mock_session.engine = mock_engine
    mock_get_session.return_value = mock_session

    mock_rec = MagicMock()
    mock_rec.transcript_parts = {}
    mock_session.get_or_create_recording.return_value = mock_rec

    brain._handle_audio_chunk("session1", 0, 0, b'\x00\x00' * 1600)

    # The engine must have been asked to transcribe once
    mock_engine.transcribe_chunk.assert_called_once()
    # The full cumulative text must be stored under key 0
    assert mock_rec.transcript_parts[0] == "cumulative text"


@patch.object(brain, "send_hud")
@patch.object(brain, "paste_instantly")
def test_finalize_recording_if_ready_always_clears_engine_memory(mock_paste, mock_hud):
    """
    Verify that _finalize_recording_if_ready always calls engine.clear_internal_memory()
    after finalizing — regardless of whether the engine is stateful or stateless.
    (Stateless engines safely do nothing when clear_internal_memory is called.)
    """
    session_id = "session1"

    # Create a mock engine with a trackable clear_internal_memory method
    mock_engine = MagicMock()

    # Build a real SessionState with the engine
    session = brain.SessionState(engine=mock_engine)
    mock_rec = brain.RecordingState()
    mock_rec.finalized = False
    mock_rec.closed = True
    mock_rec.done_count = 1
    mock_rec.received_count = 1
    mock_rec.stt_time = 0.5
    mock_rec.transcript_parts = {0: "final text"}
    session.recordings = {0: mock_rec}

    brain.session_store[session_id] = session

    brain._finalize_recording_if_ready(session_id, 0)

    # clear_internal_memory must have been called exactly once after finalization
    mock_engine.clear_internal_memory.assert_called_once()
