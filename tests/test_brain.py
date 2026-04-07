import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import brain
from streaming_shared_logic import remove_duplicate_chunk_prefix


class MockConn:
    def __init__(self, *chunks):
        self._chunks = list(chunks) + [b""]

    def settimeout(self, _timeout):
        return None

    def recv(self, _size):
        return self._chunks.pop(0)

    def close(self):
        return None


@pytest.fixture(autouse=True)
def clear_session_store():
    brain.session_store.clear()
    yield
    brain.session_store.clear()


def test_handle_connection_transcribes_audio(sample_audio_bytes):
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_backend.transcribe.return_value = "hello world"

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    conn = MockConn(sample_audio_bytes)

    with patch("brain.send_hud") as mock_hud, patch("brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_called_once()
    args, _ = mock_backend.transcribe.call_args
    assert args[0] == mock_model
    assert isinstance(args[1], np.ndarray)
    assert args[1].dtype == np.float32
    mock_paste.assert_called_once_with("hello world ")
    assert any(call.args[0] == "done" for call in mock_hud.call_args_list)


def test_handle_connection_switch_model_command():
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_new_backend = MagicMock()
    mock_new_model = MagicMock()

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    conn = MockConn(b"CMD_SWITCH_MODEL:tiny.en")

    with patch("brain.load_backend", return_value=(mock_new_backend, mock_new_model)) as mock_load:
        brain.handle_connection(conn)

    mock_load.assert_called_once_with("tiny.en")
    assert brain.backend_info["backend"] == mock_new_backend
    assert brain.backend_info["model"] == mock_new_model


def test_handle_connection_skips_too_short_audio():
    mock_backend = MagicMock()
    mock_model = MagicMock()

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    short_audio = b"\x00\x00" * 1600
    conn = MockConn(short_audio)

    with patch("brain.send_hud"), patch("brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_not_called()
    mock_paste.assert_not_called()


def test_load_backend_uses_openvino_when_selected(monkeypatch):
    fake_backend = types.SimpleNamespace(load_model=MagicMock(return_value="ov-model"))
    monkeypatch.setattr(brain, "BACKEND", "openvino")

    with patch.dict(sys.modules, {"backend_openvino": fake_backend}):
        backend_module, model = brain.load_backend("base.en")

    assert backend_module is fake_backend
    assert model == "ov-model"
    fake_backend.load_model.assert_called_once_with("base.en")


def test_load_backend_falls_back_to_faster_whisper_when_openvino_unavailable(monkeypatch):
    failing_openvino = types.SimpleNamespace(load_model=MagicMock(side_effect=RuntimeError("missing openvino")))
    fallback_backend = types.SimpleNamespace(load_model=MagicMock(return_value="fw-model"))
    monkeypatch.setattr(brain, "BACKEND", "openvino")

    with patch.dict(
        sys.modules,
        {
            "backend_openvino": failing_openvino,
            "backend_faster_whisper": fallback_backend,
        },
    ):
        backend_module, model = brain.load_backend("base.en")

    assert backend_module is fallback_backend
    assert model == "fw-model"
    failing_openvino.load_model.assert_called_once_with("base.en")
    fallback_backend.load_model.assert_called_once_with("base.en")


def test_dedupe_with_previous_chunk_removes_repeated_prefix():
    cleaned = brain._remove_duplicate_chunk_prefix(
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    )

    assert cleaned == "and doing H3 grid"


def test_brain_uses_shared_remove_duplicate_chunk_prefix():
    assert brain.remove_duplicate_chunk_prefix is remove_duplicate_chunk_prefix


def test_dedupe_with_previous_chunk_keeps_non_overlapping_text():
    cleaned = brain._remove_duplicate_chunk_prefix(
        "Now I want to see that things are happening",
        "Window",
    )

    assert cleaned == "Window"


def test_handle_audio_chunk_dedupes_against_previous_chunk_text():
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_backend.transcribe.side_effect = [
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    ]

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model
    session_id = "session123"
    audio_bytes = b"\x00\x10" * 32000

    brain._handle_audio_chunk(session_id, 0, audio_bytes)
    brain._handle_audio_chunk(session_id, 1, audio_bytes)

    session = brain.session_store[session_id]
    assert session.transcript_parts[0] == "I want to see that things are happening fine"
    assert session.transcript_parts[1] == "and doing H3 grid"
