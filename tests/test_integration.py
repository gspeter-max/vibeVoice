"""
test_integration.py — Integration test for the brain socket communication flow.

Tests that the full CMD_AUDIO_CHUNK → CMD_SESSION_COMMIT pipeline works:
  - Ear sends a chunk → Brain transcribes it (but does NOT paste yet)
  - Ear sends a commit → Brain stitches and pastes the final text

After Phase 2 wiring, brain.py uses a single TranscriptionEngine engine object.
We mock it with is_stateful()=False and transcribe_chunk() returning a fixed string.
"""

from unittest.mock import MagicMock, patch

import src.backend.brain as brain


class MockConn:
    """
    A fake socket connection that returns pre-defined audio chunks
    and then returns an empty bytes object to signal the connection is closed.
    """
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

    Flow:
      1. Ear sends CMD_AUDIO_CHUNK → Brain transcribes it, does NOT paste yet
      2. Ear sends CMD_SESSION_COMMIT → Brain stitches and calls paste_instantly
    """
    # Create a mock engine that is stateless (deduplication path)
    mock_engine = MagicMock()
    mock_engine.is_stateful.return_value = False
    mock_engine.transcribe_chunk.return_value = "integrated test result"

    # Set the engine in global brain state
    brain.backend_info["engine"] = mock_engine
    brain.session_store.clear()

    # New 4-part header: CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ
    chunk = b"CMD_AUDIO_CHUNK:session123:0:0\n\n" + sample_audio_bytes
    # New 3-part header: CMD_SESSION_COMMIT:SESSION_ID:RECORDING_INDEX
    commit = b"CMD_SESSION_COMMIT:session123:0"

    with patch("src.backend.brain.keyboard", MagicMock()), \
         patch("src.backend.brain.paste_instantly") as mock_paste, \
         patch("src.backend.brain.send_hud") as mock_hud:

        # Step 1: Send the audio chunk — should transcribe but NOT paste yet
        brain.handle_connection(MockConn(chunk))
        mock_engine.transcribe_chunk.assert_called_once()
        mock_paste.assert_not_called()

        # Step 2: Send the commit signal — should stitch and paste now
        brain.handle_connection(MockConn(commit))

    mock_paste.assert_called_once_with("integrated test result ")
    assert any(call.args[0] == "done" for call in mock_hud.call_args_list)
