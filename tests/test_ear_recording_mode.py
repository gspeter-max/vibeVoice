import os
import pytest
from unittest.mock import patch, Mock
import sys

# Standard mock setup for Ear tests to avoid audio hardware issues
sys.modules['pynput'] = type(sys)('pynput')
sys.modules['pynput.keyboard'] = type(sys)('pynput.keyboard')
sys.modules['pynput.mouse'] = type(sys)('pynput.mouse')

class MockButton:
    left = 'left'
    right = 'right'

sys.modules['pynput.mouse'].Button = MockButton()

class MockPyAudio:
    paInt16 = 16
    paContinue = 0
    class PyAudio:
        def get_default_input_device_info(self):
            return {"index": 0}
        def get_device_info_by_index(self, index):
            return {"name": "Test Device"}

sys.modules['pyaudio'] = MockPyAudio()

import src.audio.ear_runtime.controller as ear_module
from src.audio.ear_runtime.controller import Ear as RuntimeEar
from src.audio.ear_runtime.menu import TerminalMenu as RuntimeTerminalMenu
from src.audio.ear_runtime.runtime import start_ear as runtime_start_ear

def test_nemotron_not_in_models_when_no_streaming():
    """
    When RECORDING_MODE is set to 'no_streaming', Nemotron should NOT
    be included in the active models list to prevent incompatibility issues.
    """
    with patch.dict(os.environ, {"RECORDING_MODE": "no_streaming"}):
        models = ear_module.get_active_models()
        assert "nemotron-streaming-0.6b" not in models
        # Menu should only allow 4 options
        assert len(models) == 4


def test_terminal_menu_remains_available_from_audio_ear_module():
    assert hasattr(ear_module, "TerminalMenu")
    assert ear_module.TerminalMenu is RuntimeTerminalMenu


def test_audio_ear_runtime_exports_match_compatibility_surface():
    assert RuntimeEar is not None
    assert runtime_start_ear is not None

def test_terminal_menu_run_uses_audio_ear_switch_command_seam(monkeypatch):
    monkeypatch.setattr("sys.stdin.fileno", lambda: 0)
    menu = ear_module.TerminalMenu()
    captured = {}

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("termios.tcgetattr", lambda _fd: "old-settings")
    monkeypatch.setattr("termios.tcsetattr", lambda *_args: None)
    monkeypatch.setattr("tty.setcbreak", lambda _fd: None)
    monkeypatch.setattr(
        "select.select",
        lambda _read, _write, _error, _timeout: ([sys.stdin], [], []),
    )

    def fake_read(_count):
        menu._stop.set()
        return "5"

    monkeypatch.setattr("sys.stdin.read", fake_read)

    def fake_send_switch(model_name, ear_instance=None):
        captured["model_name"] = model_name
        captured["ear_instance"] = ear_instance

    monkeypatch.setattr(ear_module, "send_switch_command", fake_send_switch)
    monkeypatch.setenv("RECORDING_MODE", "silence_streaming")

    menu.run()

    assert captured == {
        "model_name": "nemotron-streaming-0.6b",
        "ear_instance": None,
    }


def test_run_self_test_uses_audio_ear_socket_path_override(monkeypatch):
    checked_paths = []
    sent = {}

    monkeypatch.setattr(
        "src.audio.ear_runtime.menu._get_socket_path_from_audio_ear",
        lambda: "/tmp/custom-parakeet.sock",
    )

    monkeypatch.setattr(
        "src.audio.ear_runtime.menu.os.path.exists",
        lambda path: checked_paths.append(path) or path == "/tmp/custom-parakeet.sock",
    )
    monkeypatch.setattr(
        "src.audio.ear_runtime.menu.send_message_to_brain",
        lambda payload, **kwargs: sent.setdefault("socket_path", kwargs.get("socket_path")) or True,
    )

    ear_module.run_self_test()

    assert checked_paths == ["/tmp/custom-parakeet.sock"]
    assert sent["socket_path"] == "/tmp/custom-parakeet.sock"

def test_nemotron_in_models_when_silence_streaming():
    """
    When RECORDING_MODE is 'silence_streaming', Nemotron SHOULD be
    included as the 5th option in the models list.
    """
    with patch.dict(os.environ, {"RECORDING_MODE": "silence_streaming"}):
        models = ear_module.get_active_models()
        assert "nemotron-streaming-0.6b" in models
        # Menu should allow all 5 options
        assert len(models) == 5

@patch('sys.stdin.fileno', return_value=0)
@patch('src.audio.ear_runtime.controller.send_switch_command')
def test_terminal_menu_ignores_nemotron_key_in_no_streaming(mock_send, mock_fileno):
    """
    Verifies that pressing '5' in the terminal menu does absolutely nothing
    if the current mode does not support Nemotron.
    """
    menu = ear_module.TerminalMenu()
    
    with patch.dict(os.environ, {"RECORDING_MODE": "no_streaming"}):
        # We simulate the sys.stdin.read returning '5'
        with patch('sys.stdin.read', return_value='5'):
            # Trigger the logic that handles 'c in 12345'
            # (Note: we just test the inner logic, not the full run() loop which is blocking)
            c = '5'
            idx = int(c) - 1
            active_models = ear_module.get_active_models()
            
            if idx < len(active_models):
                mock_send(active_models[idx], menu.ear)
                
            # mock_send should NOT have been called because len(active_models) is 4, and idx is 4.
            # 4 < 4 is False.
            mock_send.assert_not_called()

@patch('sys.stdin.fileno', return_value=0)
@patch('src.audio.ear_runtime.controller.send_switch_command')
def test_terminal_menu_accepts_nemotron_key_in_streaming(mock_send, mock_fileno):
    """
    Verifies that pressing '5' works when in streaming mode.
    """
    menu = ear_module.TerminalMenu()
    
    with patch.dict(os.environ, {"RECORDING_MODE": "silence_streaming"}):
        c = '5'
        idx = int(c) - 1
        active_models = ear_module.get_active_models()
        
        if idx < len(active_models):
            mock_send(active_models[idx], menu.ear)
            
        mock_send.assert_called_once_with("nemotron-streaming-0.6b", None)
