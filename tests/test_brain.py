from unittest.mock import MagicMock, patch

import numpy as np

import brain


class MockConn:
    def __init__(self, *chunks):
        self._chunks = list(chunks) + [b""]

    def settimeout(self, _timeout):
        return None

    def recv(self, _size):
        return self._chunks.pop(0)

    def close(self):
        return None


def test_handle_connection_transcribes_audio(sample_audio_bytes):
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_backend.transcribe.return_value = "hello world"

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model
    brain.vad_engine = None

    conn = MockConn(sample_audio_bytes)

    with patch("brain.send_hud"), patch("brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_called_once()
    args, _ = mock_backend.transcribe.call_args
    assert args[0] == mock_model
    assert isinstance(args[1], np.ndarray)
    assert args[1].dtype == np.float32
    mock_paste.assert_called_once_with("hello world ")


def test_handle_connection_switch_model_command():
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_new_backend = MagicMock()
    mock_new_model = MagicMock()

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model
    brain.vad_engine = None

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
    brain.vad_engine = None

    short_audio = b"\x00\x00" * 1600
    conn = MockConn(short_audio)

    with patch("brain.send_hud"), patch("brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_backend.transcribe.assert_not_called()
    mock_paste.assert_not_called()
