import socket
import threading
import time
import os
import pytest
from unittest.mock import MagicMock, patch
import brain
import ear

def test_socket_communication(sample_audio_bytes):
    """
    Test that data sent from Ear's _send_to_brain reaches Brain's start_server.
    We use a real Unix socket but mock the transcription and keyboard.
    """
    # Use a shorter temporary socket file to avoid "AF_UNIX path too long"
    mock_socket_path = "/tmp/test_para.sock"
    if os.path.exists(mock_socket_path):
        os.remove(mock_socket_path)

    # Setup mocks
    mock_backend = MagicMock()
    mock_model = MagicMock()
    mock_keyboard = MagicMock()
    
    transcription_event = threading.Event()
    def mock_transcribe(*args, **kwargs):
        transcription_event.set()
        return "integrated test result"
    mock_backend.transcribe.side_effect = mock_transcribe

    # Start patching globally for all threads
    patch_socket_brain = patch('brain.SOCKET_PATH', mock_socket_path)
    patch_socket_ear = patch('ear.SOCKET_PATH', mock_socket_path)
    patch_load_backend = patch('brain.load_backend', return_value=(mock_backend, mock_model))
    patch_keyboard = patch('brain.keyboard', mock_keyboard)
    patch_perf = patch('brain.time.perf_counter', side_effect=range(100))

    patch_socket_brain.start()
    patch_socket_ear.start()
    patch_load_backend.start()
    patch_keyboard.start()
    patch_perf.start()

    try:
        def run_brain():
            try:
                brain.start_server()
            except Exception as e:
                print(f"Brain thread error: {e}")

        brain_thread = threading.Thread(target=run_brain, daemon=True)
        brain_thread.start()

        # Wait for socket to be created
        max_wait = 3.0
        start_wait = time.time()
        while not os.path.exists(mock_socket_path):
            if time.time() - start_wait > max_wait:
                pytest.fail("Brain socket was not created in time")
            time.sleep(0.1)

        # 2. Setup Ear and send data
        e = ear.Ear()
        e.frames = [sample_audio_bytes]
        e._send_to_brain()

        # 3. Verify Brain received and "typed" the result
        transcription_event.wait(timeout=5.0)
        assert transcription_event.is_set()
        
        # Wait a tiny bit for the type() call to complete in the other thread
        for _ in range(20):
            if mock_keyboard.type.called:
                break
            time.sleep(0.1)
            
        mock_keyboard.type.assert_called_with("integrated test result ")

    finally:
        # Stop patches
        patch_perf.stop()
        patch_keyboard.stop()
        patch_load_backend.stop()
        patch_socket_ear.stop()
        patch_socket_brain.stop()
        
        # Cleanup socket
        if os.path.exists(mock_socket_path):
            os.remove(mock_socket_path)
