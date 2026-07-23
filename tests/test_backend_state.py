import concurrent.futures
from email.errors import MultipartInvariantViolationDefect
from unittest.mock import MagicMock, patch

import pytest

from src.backend.state import BackendState, SessionState
from src.models.nemotron_engine import NemotronEngine
from src.models.parakeet_engine import ParakeetEngine


def test_backend_state_lazy_loading_success():
    """
    Test that calling get_engine() on a BackendState with no active engine
    but a valid model_name successfully triggers load_tts(), initializes
    the engine, and returns it under lock.
    """
    mock_engine = MagicMock()
    with patch(
        "src.models.parakeet_engine.ParakeetEngine", return_value=mock_engine
    ) as mock_parakeet_engine:
        backend_state = BackendState(model_name="parakeet")
        engine = backend_state.get_engine()

        assert engine is mock_engine
        mock_parakeet_engine.assert_called_once_with("parakeet")


def test_backend_state_lazy_loading_missing_model(caplog):
    """
    Test that calling get_engine() on a BackendState when both engine
    and model_name are None returns None and writes an error log.
    """
    backend_state = BackendState()
    engine = backend_state.get_engine()

    assert engine is None
    assert caplog.records[0].filename == "state.py"
    assert caplog.records[0].levelname == "ERROR"
    assert any(
        r.message == "Failed to load the transcription engine. model_name is None"
        and r.levelname == "ERROR"
        for r in caplog.records
    )


def test_backend_state_thread_safety():
    """
    Test that BackendState's engine retrieval is thread-safe by spawning
    multiple threads attempting to access get_engine() concurrently,
    verifying no race conditions occur.
    """
    mock_engine = MagicMock()
    with patch("src.models.parakeet_engine.ParakeetEngine", return_value=mock_engine) as mock_parakeet:
        backend_state = BackendState(model_name="parakeet")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            task = []
            task.append(pool.submit(backend_state.get_engine))
            task.append(pool.submit(backend_state.get_engine))

            for t in task:
                t.result()
        mock_parakeet.assert_called_once()


def test_backend_state_engine_selection():
    """
    Test that load_tts() correctly resolves any model_name containing
    "nemotron" (case-insensitive) to NemotronEngine, and defaults to
    ParakeetEngine for all other model names.
    """
    mock_engine_parakeet = MagicMock()
    mock_engine_nemotron = MagicMock()
    with (
        patch("src.models.parakeet_engine.ParakeetEngine", return_value=mock_engine_parakeet),
        patch("src.models.nemotron_engine.NemotronEngine", return_value=mock_engine_nemotron),
    ):
        backend_state = BackendState()
        backend_state.load_tts("Nemotron")
        assert backend_state.engine is mock_engine_nemotron
        backend_state.load_tts("moonshine_backend")
        assert backend_state.engine is mock_engine_parakeet


def test_session_state_recording_isolation():
    """
    Test that SessionState.get_or_create_recording() correctly generates
    independent RecordingState instances for new indexes, verifying that
    modifying one recording index does not alter the others.
    """
    session_state = SessionState()
    rec_1 = session_state.get_or_create_recording(rec_idx=1)
    rec_2 = session_state.get_or_create_recording(rec_idx=2)

    rec_2.received_count = 5

    assert rec_1.received_count == 0
    assert session_state.recordings.get(1, None) is not None
