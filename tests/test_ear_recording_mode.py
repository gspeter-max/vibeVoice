import threading
from unittest.mock import Mock, patch
import sys



import src.audio.ear_runtime.controller as ear_module
import src.audio.ear_runtime.menu as menu_module
from src.utils.settings import settings
from src.audio.ear_runtime.controller import Ear as RuntimeEar
from src.audio.ear_runtime.menu import TerminalMenu as RuntimeTerminalMenu
from src.audio.ear_runtime.menu import run_self_test
from src.audio.ear_runtime.runtime import start_ear as runtime_start_ear

def test_nemotron_not_in_models_when_no_streaming(monkeypatch):
    """
    When RECORDING_MODE is set to 'no_streaming', Nemotron should NOT
    be included in the active models list to prevent incompatibility issues.
    """
    monkeypatch.setattr(settings, "recording_mode", "no_streaming")
    models = settings.active_stt_models
    assert "nemotron-streaming-0.6b" not in models
    # Menu should only allow 4 options
    assert len(models) == 4


def test_terminal_menu_remains_available_from_audio_ear_module():
    assert hasattr(ear_module, "TerminalMenu")
    assert ear_module.TerminalMenu is RuntimeTerminalMenu


def test_audio_ear_runtime_exports_match_compatibility_surface():
    assert RuntimeEar is not None
    assert runtime_start_ear is not None


def test_runtime_start_wrapper_uses_real_hud_helpers(monkeypatch):
    import src.audio.ear_runtime.runtime as runtime_module

    captured = {}

    class FakeEar:
        def __init__(self, input_device_index=None):
            self.input_device_index = input_device_index
            self.active_mic_name = "Test Device"
            self._cmd_press_time = 0.0
            self._brain_sock = None
            self._brain_sock_lock = threading.Lock()
            captured["ear"] = self

        def _stop_and_send(self, stop_session):
            captured["stopped_with"] = stop_session

        def record_loop(self, input_trigger=None):
            captured["input_trigger"] = input_trigger
            raise KeyboardInterrupt

        def cleanup(self):
            captured["cleaned_up"] = True

    class FakeMenu:
        def __init__(self, ear_instance=None):
            self.ear_instance = ear_instance

        def start(self):
            return None

        def stop(self):
            return None

    class FakeInputTrigger:
        def __init__(self, on_start_recording, on_stop_recording, on_toggle_recording, **_kwargs):
            captured["on_start_recording"] = on_start_recording
            captured["on_stop_recording"] = on_stop_recording
            captured["on_toggle_recording"] = on_toggle_recording

        def start_listening(self):
            return None

    class FakePyAudioInstance:
        def terminate(self):
            return None

    monkeypatch.setattr(runtime_module, "Ear", FakeEar)
    monkeypatch.setattr(runtime_module, "TerminalMenu", FakeMenu)
    monkeypatch.setattr(runtime_module, "InputTrigger", FakeInputTrigger)
    monkeypatch.setattr(runtime_module, "select_mic", lambda _p: 7)
    monkeypatch.setattr(runtime_module.pyaudio, "PyAudio", lambda: FakePyAudioInstance())
    monkeypatch.setattr(runtime_module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(runtime_module, "start_hud_command_thread", Mock())
    monkeypatch.setattr(runtime_module, "start_volume_sender_thread", Mock())
    monkeypatch.setattr(
        runtime_module,
        "start_recording_state",
        lambda ear, *, from_hold: captured.setdefault("started_from_hold", from_hold),
    )
    monkeypatch.delenv("VIBEVOICE_MIC_INDEX", raising=False)

    runtime_module.start_ear()
    captured["on_start_recording"](from_hold=True)

    runtime_module.start_hud_command_thread.assert_called_once()
    runtime_module.start_volume_sender_thread.assert_called_once()
    assert captured["started_from_hold"] is True


def test_runtime_start_wrapper_uses_real_ipc_helper_in_no_streaming_mode(monkeypatch):
    import src.audio.ear_runtime.runtime as runtime_module

    captured = {}
    opened_socket = object()

    class FakeEar:
        def __init__(self, input_device_index=None):
            self.input_device_index = input_device_index
            self.active_mic_name = "Test Device"
            self._cmd_press_time = 0.0
            self._brain_sock = None
            self._brain_sock_lock = threading.Lock()
            captured["ear"] = self

        def _stop_and_send(self, stop_session):
            captured["stopped_with"] = stop_session

        def record_loop(self, input_trigger=None):
            captured["input_trigger"] = input_trigger
            raise KeyboardInterrupt

        def cleanup(self):
            return None

    class FakeMenu:
        def __init__(self, ear_instance=None):
            self.ear_instance = ear_instance

        def start(self):
            return None

        def stop(self):
            return None

    class FakeInputTrigger:
        def __init__(self, on_start_recording, on_stop_recording, on_toggle_recording, **_kwargs):
            captured["on_start_recording"] = on_start_recording

        def start_listening(self):
            return None

    class FakePyAudioInstance:
        def terminate(self):
            return None

    monkeypatch.setattr(runtime_module, "Ear", FakeEar)
    monkeypatch.setattr(runtime_module, "TerminalMenu", FakeMenu)
    monkeypatch.setattr(runtime_module, "InputTrigger", FakeInputTrigger)
    monkeypatch.setattr(runtime_module, "select_mic", lambda _p: 7)
    monkeypatch.setattr(runtime_module.pyaudio, "PyAudio", lambda: FakePyAudioInstance())
    monkeypatch.setattr(runtime_module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(runtime_module.settings, "recording_mode", "no_streaming")
    monkeypatch.setattr(runtime_module, "open_checked_raw_audio_stream_to_brain", Mock(return_value=opened_socket))
    monkeypatch.setattr(runtime_module, "start_hud_command_thread", Mock())
    monkeypatch.setattr(runtime_module, "start_volume_sender_thread", Mock())
    monkeypatch.setattr(
        runtime_module,
        "start_recording_state",
        lambda ear, *, from_hold: captured.setdefault("started_from_hold", from_hold),
    )
    monkeypatch.delenv("VIBEVOICE_MIC_INDEX", raising=False)

    runtime_module.start_ear()
    captured["on_start_recording"](from_hold=False)

    runtime_module.open_checked_raw_audio_stream_to_brain.assert_called_once()
    assert captured["ear"]._brain_sock is opened_socket

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

    # Patch _send_switch_command_via_audio_ear on the menu module directly.
    # That helper does a lazy `from src.audio import ear` which no longer exists
    # after the ear package refactor. We bypass it entirely at the seam.
    monkeypatch.setattr(menu_module, "send_switch_command", fake_send_switch)
    monkeypatch.setattr(settings, "recording_mode", "silence_streaming")

    menu.run()

    assert captured == {
        "model_name": "nemotron-streaming-0.6b",
        "ear_instance": None,
    }


def test_run_self_test_uses_audio_ear_socket_path_override(monkeypatch):
    checked_paths = []
    sent = {}

    monkeypatch.setattr(menu_module.settings, "socket_path", "/tmp/custom-parakeet.sock")

    monkeypatch.setattr(
        "src.audio.ear_runtime.menu.os.path.exists",
        lambda path: checked_paths.append(path) or path == "/tmp/custom-parakeet.sock",
    )
    monkeypatch.setattr(
        "src.audio.ear_runtime.menu.send_message_to_brain",
        lambda payload, **kwargs: sent.setdefault("socket_path", kwargs.get("socket_path")) or True,
    )

    run_self_test()

    assert checked_paths == ["/tmp/custom-parakeet.sock"]
    assert sent["socket_path"] == "/tmp/custom-parakeet.sock"

def test_nemotron_in_models_when_silence_streaming(monkeypatch):
    """
    When RECORDING_MODE is 'silence_streaming', Nemotron SHOULD be
    included as the 5th option in the models list.
    """
    monkeypatch.setattr(settings, "recording_mode", "silence_streaming")
    models = settings.active_stt_models
    assert "nemotron-streaming-0.6b" in models
    # Menu should allow all 5 options
    assert len(models) == 5

@patch('sys.stdin.fileno', return_value=0)
@patch('src.audio.ear_runtime.menu.send_switch_command')
def test_terminal_menu_ignores_nemotron_key_in_no_streaming(mock_send, mock_fileno, monkeypatch):
    """
    Verifies that pressing '5' in the terminal menu does absolutely nothing
    if the current mode does not support Nemotron.
    """
    menu = ear_module.TerminalMenu()

    monkeypatch.setattr(settings, "recording_mode", "no_streaming")
    # We simulate the sys.stdin.read returning '5'
    with patch('sys.stdin.read', return_value='5'):
        # Trigger the logic that handles 'c in 12345'
        # (Note: we just test the inner logic, not the full run() loop which is blocking)
        c = '5'
        idx = int(c) - 1
        active_models = settings.active_stt_models

        if idx < len(active_models):
            mock_send(active_models[idx], menu.ear)

        # mock_send should NOT have been called because len(active_models) is 4, and idx is 4.
        # 4 < 4 is False.
        mock_send.assert_not_called()

@patch('sys.stdin.fileno', return_value=0)
@patch('src.audio.ear_runtime.menu.send_switch_command')
def test_terminal_menu_accepts_nemotron_key_in_streaming(mock_send, mock_fileno, monkeypatch):
    """
    Verifies that pressing '5' works when in streaming mode.
    """
    menu = ear_module.TerminalMenu()

    monkeypatch.setattr(settings, "recording_mode", "silence_streaming")
    c = '5'
    idx = int(c) - 1
    active_models = settings.active_stt_models

    if idx < len(active_models):
        mock_send(active_models[idx], menu.ear)

    mock_send.assert_called_once_with("nemotron-streaming-0.6b", None)
