from unittest.mock import MagicMock, patch

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


def test_socket_communication(sample_audio_bytes):
    """
    Test that a chunk sent by Ear is accumulated by Brain and only pasted on commit.
    """
    mock_backend = MagicMock()
    mock_model = MagicMock()

    brain.backend_info["backend"] = mock_backend
    brain.backend_info["model"] = mock_model

    # New 4-part header: CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ
    chunk = b"CMD_AUDIO_CHUNK:session123:0:0\n\n" + sample_audio_bytes
    # New 3-part header: CMD_SESSION_COMMIT:SESSION_ID:RECORDING_INDEX
    commit = b"CMD_SESSION_COMMIT:session123:0"

    mock_backend.transcribe.return_value = "integrated test result"

    with patch("brain.keyboard", MagicMock()), \
         patch("brain.paste_instantly") as mock_paste, \
         patch("brain.send_hud") as mock_hud:
        brain.handle_connection(MockConn(chunk))
        mock_backend.transcribe.assert_called_once()
        mock_paste.assert_not_called()

        brain.handle_connection(MockConn(commit))

    mock_paste.assert_called_once_with("integrated test result ")
    assert any(call.args[0] == "done" for call in mock_hud.call_args_list)
