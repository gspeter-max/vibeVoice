import threading
import socket
from unittest.mock import MagicMock, patch

import brain
import ear


def test_socket_communication(sample_audio_bytes):
    """
    Test that data sent from Ear's streaming socket reaches Brain's handler.
    """
    mock_backend = MagicMock()
    mock_model = MagicMock()
    transcription_event = threading.Event()

    def mock_transcribe(*_args, **_kwargs):
        if _args and hasattr(_args[1], "__len__") and len(_args[1]) > 10000:
            transcription_event.set()
        return "integrated test result"

    mock_backend.transcribe.side_effect = mock_transcribe

    with patch("brain.keyboard", MagicMock()), \
         patch("brain.paste_instantly"), \
         patch("brain.send_hud"), \
         patch("ear.Ear._open_mic_stream", lambda self: None), \
         patch("ear.pyaudio.PyAudio") as mock_pyaudio:

        mock_pyaudio.return_value.get_default_input_device_info.return_value = {"index": 0}
        mock_pyaudio.return_value.get_device_info_by_index.return_value = {"name": "Test Device"}
        mock_pyaudio.return_value.terminate.return_value = None

        brain.backend_info["backend"] = mock_backend
        brain.backend_info["model"] = mock_model
        brain.vad_engine = None

        parent_sock, child_sock = socket.socketpair()

        def run_brain_handler():
            brain.handle_connection(child_sock)

        brain_thread = threading.Thread(target=run_brain_handler, daemon=True)
        brain_thread.start()

        e = ear.Ear(input_device_index=0)
        e._brain_sock = parent_sock
        e._stream_chunk_to_brain(sample_audio_bytes)
        e._close_brain_stream()

        transcription_event.wait(timeout=5.0)
        assert transcription_event.is_set()
        mock_backend.transcribe.assert_called()
