import pytest
import time
import sys
import json
import threading
from unittest.mock import ANY, Mock, patch

from src.streaming.session import should_split_chunk_after_silence

from src.audio.ear_runtime.controller import Ear
from src.audio.ear_runtime.devices import resolve_input_device_index, select_mic
from src.utils.settings import settings

class FakePyAudio:
    def get_default_input_device_info(self):
        return {"index": 0}
    def get_device_info_by_index(self, index):
        return {"name": "Test Device"}
    def terminate(self):
        pass

# Patch the _open_mic_stream method to avoid audio setup issues
@pytest.fixture(autouse=True)
def patch_open_mic_stream(monkeypatch):
    """Patch `open_mic_stream` to avoid actual audio setup."""
    def dummy_open_stream(_self):
        pass
    monkeypatch.setattr("src.audio.ear_runtime.controller.open_mic_stream", dummy_open_stream)


@pytest.fixture(autouse=True)
def patch_silero_vad(monkeypatch):
    """Patch SileroVAD so Ear tests do not load the real ONNX model."""
    def dummy_vad(*_args, **_kwargs):
        return Mock(
            reset=lambda: None,
            is_speech=lambda *_a, **_k: 1.0,
        )
    monkeypatch.setattr("src.audio.ear_runtime.controller.SileroVAD", dummy_vad)

def test_ear_does_not_own_mouse_hold_state():
    """
    Mouse hold state (press time, holding flag, recording-from-hold flag) is
    managed exclusively by InputTrigger. Ear must NOT have those fields.
    InputTrigger.check_mouse_hold_threshold() is the single source of truth.
    """
    ear = Ear(pyaudio_lib=FakePyAudio())

    # These fields lived on Ear before the refactor — they are now gone.
    assert not hasattr(ear, '_is_holding'), "_is_holding was moved to InputTrigger"
    assert not hasattr(ear, '_mouse_press_start_time'), "_mouse_press_start_time was moved to InputTrigger"
    assert not hasattr(ear, '_recording_from_hold'), "_recording_from_hold was moved to InputTrigger"
    assert not hasattr(ear, 'on_mouse_click'), "on_mouse_click was removed; InputTrigger handles mouse events"


def test_capture_session_keeps_session_id_across_recordings_and_increments_on_commit_only():
    from src.streaming.capture_session import CaptureSession

    session = CaptureSession(sample_rate=16000, overlap_seconds=0.002)
    first_session_id = session.current_session_id

    session.begin_recording(now_seconds=1.0)
    assert session.current_recording_index == 0

    session.mark_recording_committed()
    assert session.current_session_id == first_session_id
    assert session.current_recording_index == 1


def test_capture_session_resets_chunk_sequence_after_final_stop():
    from src.streaming.capture_session import CaptureSession

    session = CaptureSession(sample_rate=16000, overlap_seconds=0.002)
    session.begin_recording(now_seconds=1.0)
    assert session.current_chunk_sequence_number == 0

    session.mark_chunk_sent()
    assert session.current_chunk_sequence_number == 1

    session.mark_recording_committed()
    assert session.current_chunk_sequence_number == 0


def test_capture_session_mark_recording_stopped_clears_overlap_tail_and_chunk_start():
    from src.streaming.capture_session import CaptureSession

    session = CaptureSession(sample_rate=16000, overlap_seconds=0.002)
    session.chunk_started_at_seconds = 12.5
    session.last_chunk_tail_bytes = b"tail"

    session.mark_recording_stopped()

    assert session.chunk_started_at_seconds == 0.0
    assert session.last_chunk_tail_bytes == b""


def test_capture_session_mark_nonfinal_chunk_sent_updates_next_chunk_start_time():
    from src.streaming.capture_session import CaptureSession

    session = CaptureSession(sample_rate=16000, overlap_seconds=0.002)
    session.mark_nonfinal_chunk_sent(now_seconds=7.25)

    assert session.chunk_started_at_seconds == 7.25


def test_capture_session_current_chunk_age_seconds_uses_session_owned_start_time():
    from src.streaming.capture_session import CaptureSession

    session = CaptureSession(sample_rate=16000, overlap_seconds=0.002)
    session.chunk_started_at_seconds = 3.0

    assert session.current_chunk_age_seconds(now_seconds=5.5) == 2.5


def test_resolve_input_device_index_prefers_explicit_index(monkeypatch):
    monkeypatch.setenv("VIBEVOICE_MIC_INDEX", "9")

    class FakePyAudioForDeviceResolution:
        def get_default_input_device_info(self):
            return {"index": 4}

    assert resolve_input_device_index(FakePyAudioForDeviceResolution(), 2) == 2


def test_resolve_input_device_index_uses_valid_environment_value(monkeypatch):
    monkeypatch.setenv("VIBEVOICE_MIC_INDEX", "7")

    class FakePyAudioForDeviceResolution:
        def get_default_input_device_info(self):
            return {"index": 4}

    assert resolve_input_device_index(FakePyAudioForDeviceResolution(), None) == 7


def test_resolve_input_device_index_falls_back_to_default_when_environment_is_invalid(monkeypatch):
    monkeypatch.setenv("VIBEVOICE_MIC_INDEX", "not-a-number")

    class FakePyAudioForDeviceResolution:
        def get_default_input_device_info(self):
            return {"index": 4}

    assert resolve_input_device_index(FakePyAudioForDeviceResolution(), None) == 4


def test_send_message_to_brain_returns_false_when_message_is_empty():
    from src.ipc.client import send_message_to_brain

    assert send_message_to_brain(b"") is False


def test_hud_client_sends_command_over_tcp_socket():
    from src.ui.hud_client import send_hud_command

    captured = {"connected": None, "payload": None, "closed": False}

    class FakeSocket:
        def settimeout(self, _timeout):
            return None

        def connect(self, address):
            captured["connected"] = address

        def sendall(self, payload):
            captured["payload"] = payload

        def close(self):
            captured["closed"] = True

    with patch("src.utils.socket_utils.socket.socket", return_value=FakeSocket()):
        assert send_hud_command("listen") is True

    assert captured["connected"] == ("127.0.0.1", 57234)
    assert captured["payload"] == b"listen"
    assert captured["closed"] is True


def test_hud_client_volume_sender_stops_after_recording_ends():
    from src.ui.hud_client import start_volume_sender_thread

    sent_packets = []

    class FakeUdpSocket:
        def sendto(self, payload, address):
            sent_packets.append((payload, address))

        def close(self):
            sent_packets.append(("closed", None))

    class FakeEarState:
        def __init__(self):
            self._lock = threading.Lock()
            self.is_recording = True
            self.last_rms = 0.5
            self.last_frequency_bands = {"bass": 0.1, "mid": 0.7, "treble": 0.2}

    ear_state = FakeEarState()

    def fake_sleep(_seconds):
        with ear_state._lock:
            ear_state.is_recording = False

    with patch("src.utils.socket_utils.socket.socket", return_value=FakeUdpSocket()), \
         patch("src.ui.hud_client.time.sleep", side_effect=fake_sleep):
        sender_thread = start_volume_sender_thread(ear_state, volume_port=57235)
        sender_thread.join(timeout=1.0)

    assert sent_packets[0][1] == ("127.0.0.1", 57235)
    assert sent_packets[0][0].startswith(b"vol:0.5000,bass:0.100,mid:0.700,treble:0.200")
    assert sent_packets[-1] == ("closed", None)


def test_raw_stream_helpers_open_send_and_close_socket():
    from src.ipc.client import (
        close_raw_audio_stream_and_forget,
        open_checked_raw_audio_stream_to_brain,
        open_raw_audio_stream_to_brain,
        send_raw_audio_stream_chunk_or_close,
        send_raw_audio_stream_chunk,
        close_raw_audio_stream_to_brain,
    )

    captured = {"connected": False, "payloads": [], "shutdown_called": False, "closed": False}

    class FakeSocket:
        def settimeout(self, _timeout):
            return None

        def connect(self, address):
            captured["connected"] = True
            captured["address"] = address

        def sendall(self, payload):
            captured["payloads"].append(payload)

        def shutdown(self, _how):
            captured["shutdown_called"] = True

        def close(self):
            captured["closed"] = True

    with patch("src.utils.socket_utils.socket.socket", return_value=FakeSocket()), \
         patch("src.ipc.client.os.path.exists", return_value=True):
        socket_handle = open_raw_audio_stream_to_brain()
        assert socket_handle is not None
        assert send_raw_audio_stream_chunk(socket_handle, b"chunk-bytes") is True
        close_raw_audio_stream_to_brain(socket_handle)
        checked_socket_handle = open_checked_raw_audio_stream_to_brain()
        assert checked_socket_handle is not None
        assert send_raw_audio_stream_chunk_or_close(checked_socket_handle, b"chunk-two") is checked_socket_handle
        close_raw_audio_stream_and_forget(checked_socket_handle)

    assert captured["connected"] is True
    assert captured["payloads"] == [b"chunk-bytes", b"chunk-two"]
    assert captured["shutdown_called"] is True
    assert captured["closed"] is True


def test_send_raw_audio_stream_chunk_or_close_returns_none_after_send_failure():
    from src.ipc.client import send_raw_audio_stream_chunk_or_close

    captured = {"shutdown_called": False, "closed": False}

    class BrokenSocket:
        def sendall(self, _payload):
            raise OSError("disconnect")

        def shutdown(self, _how):
            captured["shutdown_called"] = True

        def close(self):
            captured["closed"] = True

    assert send_raw_audio_stream_chunk_or_close(BrokenSocket(), b"chunk") is None
    assert captured["shutdown_called"] is True
    assert captured["closed"] is True

def test_record_loop_tick_delegates_mouse_hold_to_input_trigger():
    """
    _record_loop_tick must call input_trigger.check_mouse_hold_threshold() on
    every tick when an InputTrigger is provided. This is how the 1-second
    right-mouse-button hold-to-record activates — InputTrigger owns all mouse
    state; Ear just polls it each cycle.
    """
    ear = Ear(pyaudio_lib=FakePyAudio())

    mock_trigger = Mock()
    mock_trigger.check_mouse_hold_threshold.return_value = False

    ear._record_loop_tick(input_trigger=mock_trigger)

    mock_trigger.check_mouse_hold_threshold.assert_called_once()


def test_record_loop_tick_does_not_error_without_input_trigger():
    """
    _record_loop_tick is safe to call with no input_trigger argument.
    This keeps unit tests for other tick behaviour simple — they don't
    need to supply a trigger if they're not testing mouse hold.
    """
    ear = Ear(pyaudio_lib=FakePyAudio())
    # Must not raise even with no trigger passed
    ear._record_loop_tick(input_trigger=None)


def test_mouse_release_finalizes_immediately_when_recording():
    """Releasing the right mouse button stops recording via InputTrigger callbacks."""
    import src.input.hotkeys as hotkeys_module
    from src.input.hotkeys import InputTrigger

    # Use the actual button value that hotkeys.py compares against internally
    right_button = getattr(hotkeys_module.mouse.Button, 'right', 'right')

    mock_stop = Mock()
    trigger = InputTrigger(
        on_start_recording=Mock(),
        on_stop_recording=mock_stop,
        on_toggle_recording=Mock(),
    )

    with patch('src.input.hotkeys.time.time', return_value=0.0):
        trigger._handle_mouse_click(0, 0, right_button, pressed=True)

    with patch('src.input.hotkeys.time.time', return_value=1.1):
        trigger.check_mouse_hold_threshold()  # starts recording

    trigger._handle_mouse_click(0, 0, right_button, pressed=False)
    mock_stop.assert_called_once_with(stop_session=True)


def test_key_release_finalizes_immediately_when_recording():
    """Right CMD release should finalize now (no extra silence wait)."""
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._cmd_press_time = time.time() - 1.0

    with patch("src.audio.ear_runtime.controller._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_release(object())

    mock_stop.assert_called_once_with(stop_session=True)


def test_quick_cmd_tap_enters_toggle_mode_without_stopping():
    """Short Right CMD tap should keep recording active and arm toggle mode."""
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._cmd_press_time = time.time()

    with patch("src.audio.ear_runtime.controller._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_release(object())

    assert ear._toggle_active is True
    mock_stop.assert_not_called()


def test_second_cmd_press_stops_recording_when_toggle_active():
    """Second Right CMD press should stop an active toggle recording."""
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._toggle_active = True

    with patch("src.audio.ear_runtime.controller._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_press(object())

    assert ear._toggle_active is False
    mock_stop.assert_called_once_with(stop_session=True)


def test_key_release_does_nothing_when_toggle_already_active():
    """Releasing Right CMD should be ignored once toggle mode is active."""
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._toggle_active = True

    with patch("src.audio.ear_runtime.controller._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_release(object())

    mock_stop.assert_not_called()


def test_stop_and_send_uses_no_streaming_path(monkeypatch):
    monkeypatch.setattr(settings, "recording_mode", "no_streaming")
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True

    with patch("src.audio.ear_runtime.controller.stop_no_streaming") as mock_stop_no_streaming, \
         patch("src.audio.ear_runtime.controller.flush_current_chunk") as mock_flush:
        ear._stop_and_send(stop_session=True)

    mock_stop_no_streaming.assert_called_once_with(ear)
    mock_flush.assert_not_called()


def test_stop_and_send_uses_silence_streaming_path(monkeypatch):
    monkeypatch.setattr(settings, "recording_mode", "silence_streaming")
    ear = Ear(pyaudio_lib=FakePyAudio())

    with patch("src.audio.ear_runtime.controller.stop_no_streaming") as mock_stop_no_streaming, \
         patch("src.audio.ear_runtime.controller.flush_current_chunk") as mock_flush:
        ear._stop_and_send(stop_session=False)

    mock_stop_no_streaming.assert_not_called()
    mock_flush.assert_called_once_with(ear, stop_session=False)


def test_silence_boundary_splits_chunk_while_recording_continues():
    """Test that a silence boundary sends a chunk but keeps recording active."""
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._total_frames = 8

    mock_gate = Mock()
    mock_gate.has_speech_started.return_value = True
    mock_gate.should_finalize.return_value = True
    mock_gate.silence_elapsed.return_value = 0.2
    mock_gate.flush.return_value = b"\x01\x00" * 8
    ear._utterance_gate = mock_gate
    ear._capture_session.current_session_id = "session123"

    with patch("src.audio.ear_runtime.recording.send_audio_chunk_to_brain", return_value=True) as mock_send, \
         patch("src.audio.ear_runtime.recording.commit_recording_session") as mock_commit:
        ear._record_loop_tick()

    mock_send.assert_called_once()
    mock_commit.assert_not_called()
    assert ear.is_recording is True
    assert ear._total_frames == 0


def test_select_mic_shows_contiguous_choice_indexes_and_returns_selected_device_index(monkeypatch):
    from src.audio.ear_runtime.devices import select_mic as runtime_select_mic

    class FakePyAudio:
        def get_default_input_device_info(self):
            return {"index": 2}

        def get_device_count(self):
            return 5

        def get_device_info_by_index(self, index):
            devices = {
                0: {"name": "ZEB-THUNDER PRO", "maxInputChannels": 1},
                1: {"name": "Speaker", "maxInputChannels": 0},
                2: {"name": "External Microphone", "maxInputChannels": 1},
                3: {"name": "Monitor", "maxInputChannels": 0},
                4: {"name": "MacBook Pro Microphone", "maxInputChannels": 1},
            }
            return devices[index]

    captured_prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: captured_prompts.append(prompt) or "2",
    )

    assert runtime_select_mic(FakePyAudio()) == 4
    assert select_mic(FakePyAudio()) == 4
    assert captured_prompts == [
        "Select Mic Index [default 1]: ",
        "Select Mic Index [default 1]: ",
    ]


def test_select_mic_returns_default_device_index_when_choice_is_blank(monkeypatch):
    class FakePyAudio:
        def get_default_input_device_info(self):
            return {"index": 2}

        def get_device_count(self):
            return 3

        def get_device_info_by_index(self, index):
            return {
                "name": f"Mic {index}",
                "maxInputChannels": 1,
            }

    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert select_mic(FakePyAudio()) == 2


def test_flush_current_chunk_prepends_last_chunk_overlap_for_nonfinal_chunk():
    from src.audio.ear_runtime.recording import flush_current_chunk

    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._capture_session.current_session_id = "sess"
    ear._capture_session.overlap_audio_byte_count_override = 4
    ear._capture_session.last_chunk_tail_bytes = b"\x01\x00\x02\x00"
    ear._total_frames = 4
    ear.gain_multiplier = 1.0

    with patch.object(ear._utterance_gate, "silence_elapsed", return_value=0.0), \
         patch.object(ear._utterance_gate, "flush", return_value=b"\x03\x00\x04\x00\x05\x00\x06\x00"), \
         patch("src.audio.ear_runtime.recording.send_audio_chunk_to_brain", return_value=True) as mock_send:
        flush_current_chunk(ear, stop_session=False)

    mock_send.assert_called_once_with(ear, b"\x02\x00\x05\x00\x03\x00\x04\x00\x05\x00\x06\x00")
    assert ear._capture_session.last_chunk_tail_bytes == b"\x03\x00\x04\x00\x05\x00\x06\x00"[-4:]


def test_flush_current_chunk_does_not_prepend_overlap_on_final_stop():
    from src.audio.ear_runtime.recording import flush_current_chunk

    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._capture_session.current_session_id = "sess"
    ear._capture_session.overlap_audio_byte_count_override = 4
    ear._capture_session.last_chunk_tail_bytes = b"\x01\x00\x02\x00"
    ear._total_frames = 4
    ear.gain_multiplier = 1.0

    with patch.object(ear._utterance_gate, "silence_elapsed", return_value=0.0), \
         patch.object(ear._utterance_gate, "flush", return_value=b"\x03\x00\x04\x00"), \
         patch("src.audio.ear_runtime.recording.send_audio_chunk_to_brain", return_value=True) as mock_send, \
         patch("src.audio.ear_runtime.recording.commit_recording_session", return_value=True), \
         patch("src.audio.ear_runtime.recording.start_hud_command_thread"):
        flush_current_chunk(ear, stop_session=True)

    mock_send.assert_called_once_with(ear, b"\x03\x00\x04\x00")
    assert ear._capture_session.last_chunk_tail_bytes == b""
    assert ear._capture_session.chunk_started_at_seconds == 0.0


def test_audio_callback_streams_chunks_in_no_streaming_mode(monkeypatch):
    monkeypatch.setattr(settings, "recording_mode", "no_streaming")
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._brain_sock = object()

    with patch("src.audio.ear_runtime.recording.send_raw_audio_stream_chunk_or_close", return_value=ear._brain_sock) as mock_stream:
        ear._audio_callback(b"\x01\x00" * 8, frame_count=8, time_info=None, status=None)

    mock_stream.assert_called_once_with(ear._brain_sock, b"\x01\x00" * 8)


def test_audio_callback_uses_boosted_audio_for_vad():
    """VAD should receive the boosted signal that's used for transcription."""
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear.gain_multiplier = 4.0

    raw = (b"\x01\x00" * 4)
    gate = Mock()
    gate.push.return_value = False
    ear._utterance_gate = gate

    ear._audio_callback(raw, frame_count=4, time_info=None, status=None)

    gate.push.assert_called_once()
    assert gate.push.call_args.args == ()
    assert gate.push.call_args.kwargs["audio_chunk"] == raw
    # analysis_chunk should be the boosted version (1 * 4 = 4)
    assert gate.push.call_args.kwargs["analysis_chunk"] == (b"\x04\x00" * 4)


def test_on_press_opens_brain_stream_in_no_streaming_mode(monkeypatch):
    monkeypatch.setattr(settings, "recording_mode", "no_streaming")
    ear = Ear(pyaudio_lib=FakePyAudio())
    opened_socket = object()

    with patch("src.audio.ear_runtime.controller._is_right_cmd", return_value=True), \
         patch("src.audio.ear_runtime.controller.open_checked_raw_audio_stream_to_brain", return_value=opened_socket) as mock_open, \
         patch("src.audio.ear_runtime.controller.start_volume_sender_thread") as mock_start_volume_sender, \
         patch("src.audio.ear_runtime.controller.start_hud_command_thread") as mock_start_hud_thread:
        ear.on_press(object())

    mock_open.assert_called_once()
    mock_start_volume_sender.assert_called_once()
    mock_start_hud_thread.assert_called_once_with("listen", socket_factory=ANY)
    assert ear.is_recording is True
    assert ear._brain_sock is opened_socket


def test_flush_current_chunk_boosts_before_sending_to_brain():
    """Test that audio sent to Brain is boosted while VAD stays raw."""
    from src.audio.ear_runtime.recording import flush_current_chunk

    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._total_frames = 4
    ear.gain_multiplier = 2.0

    raw = b"\x01\x00" * 4
    gate = Mock()
    gate.has_speech_started.return_value = True
    gate.silence_elapsed.return_value = 0.2
    gate.flush.return_value = raw
    ear._utterance_gate = gate
    ear._capture_session.current_session_id = "session123"

    with patch("src.audio.ear_runtime.recording.send_audio_chunk_to_brain") as mock_send, \
         patch("src.audio.ear_runtime.recording.start_hud_command_thread") as mock_start_hud_thread:
        flush_current_chunk(ear, stop_session=True)

    mock_start_hud_thread.assert_called_once_with("process", socket_factory=ANY)
    sent_bytes = mock_send.call_args.args[1]
    assert sent_bytes == b"\x02\x00" * 4


def test_flush_current_chunk_sends_commit_even_if_last_chunk_empty():
    """If final flush is empty, Ear must still commit already-sent chunks."""
    from src.audio.ear_runtime.recording import flush_current_chunk

    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._capture_session.current_session_id = "session123"
    gate = Mock()
    gate.silence_elapsed.return_value = 0.0
    gate.flush.return_value = b""
    ear._utterance_gate = gate

    with patch("src.audio.ear_runtime.recording.commit_recording_session", return_value=True) as mock_commit:
        sent = flush_current_chunk(ear, stop_session=True)

    assert sent is False
    mock_commit.assert_called_once()
    # After stop, session ID persists (it's created once at app launch)
    assert ear._capture_session.current_session_id is not None
    assert ear._capture_session.current_chunk_sequence_number == 0


def test_audio_chunk_send_uses_session_header_and_sequence():
    """Test that Ear formats chunk payloads with session id and chunk sequence."""
    from src.audio.ear_runtime.recording import send_audio_chunk_to_brain

    ear = Ear(pyaudio_lib=FakePyAudio())
    ear._capture_session.current_session_id = "session123"
    ear._capture_session.current_chunk_sequence_number = 7

    captured = []

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            pass

        def settimeout(self, _timeout):
            return None

        def connect(self, _addr):
            return None

        def sendall(self, data):
            captured.append(data)

        def shutdown(self, _how):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    with patch("src.audio.ear_runtime.recording.socket.socket", return_value=FakeSocket()):
        sent = send_audio_chunk_to_brain(ear, b"\x01\x00" * 2)

    assert sent is True
    # New 4-part header: CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ
    assert captured[0].startswith(b"CMD_AUDIO_CHUNK:session123:0:7\n\n")
    assert captured[0].endswith(b"\x01\x00" * 2)


def test_session_event_send_uses_json_payload_and_session_header():
    """Test that Ear formats telemetry events with session id and JSON payload."""
    from src.audio.ear_runtime.recording import send_session_event_to_brain

    ear = Ear(pyaudio_lib=FakePyAudio())
    ear._telemetry_enabled = True
    ear._capture_session.current_session_id = "session123"

    captured = {}

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            pass

        def settimeout(self, _timeout):
            return None

        def connect(self, _addr):
            return None

        def sendall(self, data):
            captured["data"] = data

        def shutdown(self, _how):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    with patch("src.audio.ear_runtime.recording.socket.socket", return_value=FakeSocket()):
        sent = send_session_event_to_brain(
            ear,
            "chunk_sent_to_brain",
            {"chunk_index": 7, "audio_bytes": 42},
        )

    assert sent is True
    # New 3-part header: CMD_SESSION_EVENT:SESSION_ID:RECORDING_INDEX
    assert captured["data"].startswith(b"CMD_SESSION_EVENT:session123:0\n\n")
    payload = json.loads(captured["data"].split(b"\n\n", 1)[1].decode("utf-8"))
    assert payload == {
        "type": "chunk_sent_to_brain",
        "chunk_index": 7,
        "audio_bytes": 42,
    }



def test_start_ear_wires_input_trigger_callbacks_without_using_direct_ear_handlers(monkeypatch):
    import src.audio.ear_runtime.runtime as runtime_module

    captured = {}

    class FakeEar:
        def __init__(self, input_device_index=None):
            self.active_mic_name = "Test Device"
            self.input_device_index = input_device_index
            self._toggle_active = False
            self._brain_sock = None
            self._brain_sock_lock = threading.Lock()

        def _stop_and_send(self, stop_session):
            captured["stopped_with"] = stop_session

        def record_loop(self, input_trigger=None):
            # record_loop now accepts input_trigger and would call
            # input_trigger.check_mouse_hold_threshold() on each tick.
            raise KeyboardInterrupt

        def cleanup(self):
            captured["cleaned_up"] = True

        def on_press(self, _key):
            raise AssertionError("start_ear should not wire Ear.on_press")

        def on_release(self, _key):
            raise AssertionError("start_ear should not wire Ear.on_release")

    class FakeMenu:
        def __init__(self, ear_instance=None):
            captured["menu_ear_instance"] = ear_instance

        def start(self):
            captured["menu_started"] = True

        def stop(self):
            captured["menu_stopped"] = True

    class FakeInputTrigger:
        def __init__(self, on_start_recording, on_stop_recording, on_toggle_recording, **_kwargs):
            captured["on_start_recording"] = on_start_recording
            captured["on_stop_recording"] = on_stop_recording
            captured["on_toggle_recording"] = on_toggle_recording

        def start_listening(self):
            captured["input_trigger_started"] = True

    class FakePyAudioInstance:
        def terminate(self):
            return None

    monkeypatch.setattr(runtime_module, "Ear", FakeEar)
    monkeypatch.setattr(runtime_module, "TerminalMenu", FakeMenu)
    monkeypatch.setattr(runtime_module, "InputTrigger", FakeInputTrigger)
    monkeypatch.setattr(runtime_module, "select_mic", lambda _p: 7)
    monkeypatch.setattr(runtime_module.pyaudio, "PyAudio", lambda: FakePyAudioInstance())
    monkeypatch.setattr(runtime_module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        runtime_module,
        "start_recording_state",
        lambda ear, *, from_hold: captured.setdefault("started_from_hold", from_hold),
    )
    monkeypatch.delenv("VIBEVOICE_MIC_INDEX", raising=False)

    runtime_module.start_ear()

    assert captured["input_trigger_started"] is True
    assert captured["menu_started"] is True
    assert captured["menu_stopped"] is True
    assert captured["cleaned_up"] is True
    assert captured["on_start_recording"].__name__ == "_start_recording_wrapper"
    assert captured["on_stop_recording"].__name__ == "_stop_recording_wrapper"
    assert captured["on_toggle_recording"].__name__ == "_toggle_recording_wrapper"


def test_record_loop_tick_skips_silence_finalize_in_no_streaming_mode(monkeypatch):
    monkeypatch.setattr(settings, "recording_mode", "no_streaming")
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._capture_session.chunk_started_at_seconds = time.time() - 2.0
    ear._utterance_gate = Mock()
    ear._utterance_gate.has_speech_started.return_value = True
    ear._utterance_gate.should_finalize.return_value = True

    with patch.object(ear, "_stop_and_send") as mock_stop:
        ear._record_loop_tick()

    mock_stop.assert_not_called()


def test_record_loop_tick_finalizes_on_silence_in_silence_streaming_mode(monkeypatch):
    monkeypatch.setattr(settings, "recording_mode", "silence_streaming")
    monkeypatch.setattr(settings, "minimum_chunk_age_before_silence_split_seconds", 1.0)
    ear = Ear(pyaudio_lib=FakePyAudio())
    ear.is_recording = True
    ear._capture_session.chunk_started_at_seconds = time.time() - 2.0
    ear._utterance_gate = Mock()
    ear._utterance_gate.has_speech_started.return_value = True
    ear._utterance_gate.should_finalize.return_value = True
    ear._utterance_gate.silence_elapsed.return_value = 1.2

    with patch.object(ear, "_stop_and_send") as mock_stop:
        ear._record_loop_tick()

    mock_stop.assert_called_once_with(stop_session=False)
