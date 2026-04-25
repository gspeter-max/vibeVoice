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

    with (
        patch("brain.send_hud") as mock_hud,
        patch("brain.paste_instantly") as mock_paste,
    ):
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_called_once()
    args, _ = mock_backend.transcribe.call_args
    assert args[0] == mock_model
    assert isinstance(args[1], np.ndarray)
    assert args[1].dtype == np.float32
    mock_paste.assert_called_once_with("hello world ")
    assert any(call.args[0] == "done" for call in mock_hud.call_args_list)


def test_handle_connection_no_streaming_buffers_raw_audio_until_socket_close(
    sample_audio_bytes, monkeypatch
):
    monkeypatch.setattr(brain, "RECORDING_MODE", "no_streaming")

    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_backend.transcribe.return_value = "hello world"

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    conn = MockConn(sample_audio_bytes)

    with patch("brain.send_hud"), patch("brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_called_once()
    mock_paste.assert_called_once_with("hello world ")


def test_handle_connection_switch_model_command():
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_new_backend = MagicMock()
    mock_new_model = MagicMock()

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    conn = MockConn(b"CMD_SWITCH_MODEL:tiny.en")

    with patch(
        "brain.load_transcription_engine", return_value=(mock_new_backend, mock_new_model)
    ) as mock_load:
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


def test_dedupe_with_previous_chunk_removes_repeated_prefix():
    cleaned = remove_duplicate_chunk_prefix(
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    )

    assert cleaned == "doing H3 grid"


def test_handle_audio_chunk_dedupes_against_previous_chunk_text():
    mock_backend = MagicMock()
    del mock_backend.add_audio_chunk_and_get_text
    mock_model = MagicMock()
    mock_backend.transcribe.side_effect = [
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    ]

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model
    session_id = "session123"
    audio_bytes = b"\x00\x10" * 32000

    brain._handle_audio_chunk(session_id, 0, 0, audio_bytes)
    brain._handle_audio_chunk(session_id, 0, 1, audio_bytes)

    session = brain.session_store[session_id]
    assert (
        session.recordings[0].transcript_parts[0]
        == "I want to see that things are happening fine"
    )
    assert session.recordings[0].transcript_parts[1] == "doing H3 grid"


def test_finalize_session_pastes_stitched_text_directly():
    session_id = "session123"
    session = brain.SessionState(backend=object(), model=object())
    # Set up a completed recording at index 0
    rec = brain.RecordingState()
    rec.received_count = 2
    rec.done_count = 2
    rec.closed = True
    rec.transcript_parts[0] = "hello"
    rec.transcript_parts[1] = "world"
    session.recordings[0] = rec
    brain.session_store[session_id] = session

    with (
        patch("brain.log.info") as mock_log,
        patch("brain.send_hud"),
        patch("brain.paste_instantly") as mock_paste,
    ):
        brain._finalize_recording_if_ready(session_id, 0)

    mock_paste.assert_called_once_with("hello world ")
    logged_messages = [call.args[0] for call in mock_log.call_args_list]
    assert any("Finalizing recording" in message for message in logged_messages)


def test_handle_session_event_writes_telemetry_file(tmp_path, monkeypatch):
    monkeypatch.setattr(brain, "STREAMING_TELEMETRY_ENABLED", True)
    monkeypatch.setattr(brain, "STREAMING_TELEMETRY_DIR", tmp_path)
    brain.backend_info["backend"] = MagicMock(CURRENT_MODEL_NAME="base.en")
    brain.backend_info["model"] = MagicMock()

    blob = (
        b"CMD_SESSION_EVENT:session123:0\n\n"
        b'{"type":"vad_no_speech_warning","chunk_index":0,"max_score":0.017,"threshold":0.5,"last_energy":0.0074,"energy_threshold":0.05}'
    )

    brain._handle_session_event(blob)

    json_files = list(tmp_path.glob("*session123*.json"))
    assert len(json_files) == 1
    payload = json_files[0].read_text(encoding="utf-8")
    assert '"vad_no_speech_warning_seen": true' in payload
