import pytest
import time
import sys
import os
import json
from unittest.mock import Mock, patch

from src.streaming.streaming_shared_logic import should_split_chunk_after_silence

# Mock pynput before importing Ear
sys.modules['pynput'] = type(sys)('pynput')
sys.modules['pynput.keyboard'] = type(sys)('pynput.keyboard')
sys.modules['pynput.mouse'] = type(sys)('pynput.mouse')

# Add Button mock
class MockButton:
    left = 'left'
    right = 'right'

sys.modules['pynput.mouse'].Button = MockButton()

# Mock pyaudio
class MockPyAudio:
    paInt16 = 16
    paContinue = 0
    class PyAudio:
        def get_default_input_device_info(self):
            return {"index": 0}
        def get_device_info_by_index(self, index):
            return {"name": "Test Device"}

sys.modules['pyaudio'] = MockPyAudio()

from src.audio.ear import Ear, select_mic

# Patch the _open_mic_stream method to avoid audio setup issues
@pytest.fixture(autouse=True)
def patch_open_mic_stream(monkeypatch):
    """Patch _open_mic_stream to avoid actual audio setup."""
    def dummy_open_stream(self):
        pass
    monkeypatch.setattr("src.audio.ear.Ear._open_mic_stream", dummy_open_stream)


@pytest.fixture(autouse=True)
def patch_silero_vad(monkeypatch):
    """Patch SileroVAD so Ear tests do not load the real ONNX model."""
    def dummy_vad(*_args, **_kwargs):
        return Mock(
            reset=lambda: None,
            is_speech=lambda *_a, **_k: 1.0,
        )
    monkeypatch.setattr("src.audio.ear.SileroVAD", dummy_vad)

def test_ear_has_hold_state_variables():
    """Test that Ear initializes with hold-related state variables."""
    ear = Ear()

    # New hold state variables should exist
    assert hasattr(ear, '_mouse_press_start_time'), "Ear should have _mouse_press_start_time attribute"
    assert ear._mouse_press_start_time == 0.0, "_mouse_press_start_time should initialize to 0.0"

    assert hasattr(ear, '_is_holding'), "Ear should have _is_holding attribute"
    assert ear._is_holding is False, "_is_holding should initialize to False"

    assert hasattr(ear, '_recording_from_hold'), "Ear should have _recording_from_hold attribute"
    assert ear._recording_from_hold is False, "_recording_from_hold should initialize to False"

    # Old click state variables should NOT exist
    assert not hasattr(ear, '_mouse_click_count'), "Ear should NOT have _mouse_click_count attribute (removed)"
    assert not hasattr(ear, '_mouse_clicks_required'), "Ear should NOT have _mouse_clicks_required attribute (removed)"
    assert not hasattr(ear, '_mouse_click_timeout'), "Ear should NOT have _mouse_click_timeout attribute (removed)"

def test_mouse_press_starts_hold_timer():
    """Test that pressing mouse button starts hold timer."""
    ear = Ear()

    # Simulate mouse press
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)

    # Should set press time and holding flag
    assert ear._is_holding is True
    assert ear._mouse_press_start_time > 0
    assert ear._recording_from_hold is False


def test_mouse_release_stops_hold_timer():
    """Test that releasing mouse button clears holding flag."""
    ear = Ear()

    # Press
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)
    assert ear._is_holding is True

    # Release
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=False)
    assert ear._is_holding is False


def test_mouse_release_finalizes_immediately_when_recording():
    """Test that releasing while recording finalizes immediately."""
    ear = Ear()
    ear.is_recording = True
    ear._recording_from_hold = True

    with patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=False)

    assert ear._is_holding is False
    mock_stop.assert_called_once_with(stop_session=True)


def test_key_release_finalizes_immediately_when_recording():
    """Right CMD release should finalize now (no extra silence wait)."""
    ear = Ear()
    ear.is_recording = True
    ear._cmd_press_time = time.time() - 1.0

    with patch("src.audio.ear._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_release(object())

    mock_stop.assert_called_once_with(stop_session=True)


def test_quick_cmd_tap_enters_toggle_mode_without_stopping():
    """Short Right CMD tap should keep recording active and arm toggle mode."""
    ear = Ear()
    ear.is_recording = True
    ear._cmd_press_time = time.time()

    with patch("src.audio.ear._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_release(object())

    assert ear._toggle_active is True
    mock_stop.assert_not_called()


def test_second_cmd_press_stops_recording_when_toggle_active():
    """Second Right CMD press should stop an active toggle recording."""
    ear = Ear()
    ear.is_recording = True
    ear._toggle_active = True

    with patch("src.audio.ear._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_press(object())

    assert ear._toggle_active is False
    mock_stop.assert_called_once_with(stop_session=True)


def test_key_release_does_nothing_when_toggle_already_active():
    """Releasing Right CMD should be ignored once toggle mode is active."""
    ear = Ear()
    ear.is_recording = True
    ear._toggle_active = True

    with patch("src.audio.ear._is_right_cmd", return_value=True), \
         patch.object(ear, "_stop_and_send") as mock_stop:
        ear.on_release(object())

    mock_stop.assert_not_called()


def test_stop_and_send_uses_no_streaming_path(monkeypatch):
    monkeypatch.setattr("src.audio.ear.RECORDING_MODE", "no_streaming")
    ear = Ear()
    ear.is_recording = True

    with patch.object(ear, "_stop_no_streaming") as mock_stop_no_streaming, \
         patch.object(ear, "_flush_current_chunk") as mock_flush:
        ear._stop_and_send(stop_session=True)

    mock_stop_no_streaming.assert_called_once_with()
    mock_flush.assert_not_called()


def test_stop_and_send_uses_silence_streaming_path(monkeypatch):
    monkeypatch.setattr("src.audio.ear.RECORDING_MODE", "silence_streaming")
    ear = Ear()

    with patch.object(ear, "_stop_no_streaming") as mock_stop_no_streaming, \
         patch.object(ear, "_flush_current_chunk") as mock_flush:
        ear._stop_and_send(stop_session=False)

    mock_stop_no_streaming.assert_not_called()
    mock_flush.assert_called_once_with(stop_session=False)


def test_silence_boundary_splits_chunk_while_recording_continues():
    """Test that a silence boundary sends a chunk but keeps recording active."""
    ear = Ear()
    ear.is_recording = True
    ear._total_frames = 8

    mock_gate = Mock()
    mock_gate.has_speech_started.return_value = True
    mock_gate.should_finalize.return_value = True
    mock_gate.silence_elapsed.return_value = 0.2
    mock_gate.flush.return_value = b"\x01\x00" * 8
    ear._utterance_gate = mock_gate
    ear._current_session_id = "session123"

    with patch.object(ear, "_send_audio_chunk_to_brain", return_value=True) as mock_send, \
         patch.object(ear, "_commit_recording_session") as mock_commit, \
         patch.object(ear, "_send_hud"):
        ear._record_loop_tick()

    mock_send.assert_called_once()
    mock_commit.assert_not_called()
    assert ear.is_recording is True
    assert ear._total_frames == 0


def test_ear_uses_shared_should_split_chunk_after_silence():
    import src.audio.ear as ear_module

    assert ear_module.should_split_chunk_after_silence is should_split_chunk_after_silence


def test_select_mic_shows_contiguous_choice_indexes_and_returns_selected_device_index(monkeypatch):
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

    assert select_mic(FakePyAudio()) == 4
    assert captured_prompts == ["Select Mic Index [default 1]: "]


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
    ear = Ear()
    ear.is_recording = True
    ear._current_session_id = "sess"
    ear._chunk_overlap_audio_bytes = 4
    ear._last_chunk_tail_bytes = b"\x01\x00\x02\x00"
    ear._total_frames = 4

    with patch.object(ear._utterance_gate, "silence_elapsed", return_value=0.0), \
         patch.object(ear._utterance_gate, "flush", return_value=b"\x03\x00\x04\x00\x05\x00\x06\x00"), \
         patch.object(ear, "_boost_audio_chunk", side_effect=lambda b: b), \
         patch.object(ear, "_send_audio_chunk_to_brain", return_value=True) as mock_send:
        ear._flush_current_chunk(stop_session=False)

    mock_send.assert_called_once_with(b"\x02\x00\x05\x00\x03\x00\x04\x00\x05\x00\x06\x00")
    assert ear._last_chunk_tail_bytes == b"\x03\x00\x04\x00\x05\x00\x06\x00"[-4:]


def test_flush_current_chunk_does_not_prepend_overlap_on_final_stop():
    ear = Ear()
    ear.is_recording = True
    ear._current_session_id = "sess"
    ear._chunk_overlap_audio_bytes = 4
    ear._last_chunk_tail_bytes = b"\x01\x00\x02\x00"
    ear._total_frames = 4

    with patch.object(ear._utterance_gate, "silence_elapsed", return_value=0.0), \
         patch.object(ear._utterance_gate, "flush", return_value=b"\x03\x00\x04\x00"), \
         patch.object(ear, "_boost_audio_chunk", side_effect=lambda b: b), \
         patch.object(ear, "_send_audio_chunk_to_brain", return_value=True) as mock_send, \
         patch.object(ear, "_commit_recording_session", return_value=True):
        ear._flush_current_chunk(stop_session=True)

    mock_send.assert_called_once_with(b"\x03\x00\x04\x00")
    assert ear._last_chunk_tail_bytes == b""


def test_audio_callback_streams_chunks_in_no_streaming_mode(monkeypatch):
    monkeypatch.setattr("src.audio.ear.RECORDING_MODE", "no_streaming")
    ear = Ear()
    ear.is_recording = True

    with patch.object(ear, "_stream_chunk_to_brain") as mock_stream:
        ear._audio_callback(b"\x01\x00" * 8, frame_count=8, time_info=None, status=None)

    mock_stream.assert_called_once()


def test_audio_callback_uses_boosted_audio_for_vad():
    """VAD should receive the boosted signal that's used for transcription."""
    ear = Ear()
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
    monkeypatch.setattr("src.audio.ear.RECORDING_MODE", "no_streaming")
    ear = Ear()

    with patch("src.audio.ear._is_right_cmd", return_value=True), \
         patch.object(ear, "_open_brain_stream", return_value=True) as mock_open, \
         patch.object(ear, "_start_volume_sender"), \
         patch.object(ear, "_send_hud"):
        ear.on_press(object())

    mock_open.assert_called_once_with()
    assert ear.is_recording is True


def test_flush_current_chunk_boosts_before_sending_to_brain():
    """Test that audio sent to Brain is boosted while VAD stays raw."""
    ear = Ear()
    ear.is_recording = True
    ear._total_frames = 4
    ear.gain_multiplier = 2.0

    raw = b"\x01\x00" * 4
    gate = Mock()
    gate.has_speech_started.return_value = True
    gate.silence_elapsed.return_value = 0.2
    gate.flush.return_value = raw
    ear._utterance_gate = gate
    ear._current_session_id = "session123"

    with patch.object(ear, "_send_audio_chunk_to_brain") as mock_send, \
         patch.object(ear, "_send_hud"):
        ear._flush_current_chunk(stop_session=True)

    sent_bytes = mock_send.call_args.args[0]
    assert sent_bytes == b"\x02\x00" * 4


def test_flush_current_chunk_sends_commit_even_if_last_chunk_empty():
    """If final flush is empty, Ear must still commit already-sent chunks."""
    ear = Ear()
    ear.is_recording = True
    ear._current_session_id = "session123"
    gate = Mock()
    gate.silence_elapsed.return_value = 0.0
    gate.flush.return_value = b""
    ear._utterance_gate = gate

    with patch.object(ear, "_commit_recording_session", return_value=True) as mock_commit:
        sent = ear._flush_current_chunk(stop_session=True)

    assert sent is False
    mock_commit.assert_called_once()
    # After stop, session ID persists (it's created once at app launch)
    assert ear._current_session_id is not None
    assert ear._chunk_seq == 0


def test_audio_chunk_send_uses_session_header_and_sequence():
    """Test that Ear formats chunk payloads with session id and chunk sequence."""
    ear = Ear()
    ear._current_session_id = "session123"
    ear._chunk_seq = 7

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

    with patch("src.audio.ear.socket.socket", return_value=FakeSocket()):
        sent = ear._send_audio_chunk_to_brain(b"\x01\x00" * 2)

    assert sent is True
    # New 4-part header: CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ
    assert captured["data"].startswith(b"CMD_AUDIO_CHUNK:session123:0:7\n\n")
    assert captured["data"].endswith(b"\x01\x00" * 2)


def test_session_event_send_uses_json_payload_and_session_header():
    """Test that Ear formats telemetry events with session id and JSON payload."""
    ear = Ear()
    ear._telemetry_enabled = True
    ear._current_session_id = "session123"

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

    with patch("src.audio.ear.socket.socket", return_value=FakeSocket()):
        sent = ear._send_session_event_to_brain(
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


def test_only_right_button_triggers_hold():
    """Test that only right mouse button triggers hold logic."""
    ear = Ear()

    # Left button press should be ignored
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.left, pressed=True)
    assert ear._is_holding is False

    # Right button press should work
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)
    assert ear._is_holding is True


def test_early_release_does_not_start_recording():
    """Test that releasing before 1 second does not start recording."""
    ear = Ear()

    # Press and immediately release (< 1 second)
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)
    ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=False)

    # Should not be recording
    assert ear.is_recording is False
    assert ear._recording_from_hold is False


def test_hold_one_second_starts_recording():
    """Test that holding for 1+ seconds starts recording."""
    import time as time_module

    ear = Ear()

    # Mock the brain stream and other dependencies
    with patch.object(ear, '_open_brain_stream', return_value=True):
        with patch.object(ear, '_send_hud'):
            with patch.object(ear, '_start_volume_sender'):
                # Press mouse button
                ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)

                # Wait 1.1 seconds (exceeds 1.0s threshold)
                time_module.sleep(1.1)

                # Call the record loop tick (single iteration)
                ear._record_loop_tick()

                # Should have started recording
                assert ear.is_recording is True, "is_recording should be True after 1.1s hold"
                assert ear._recording_from_hold is True, "_recording_from_hold should be True"


def test_hold_less_than_one_second_no_recording():
    """Test that holding < 1 second does not start recording."""
    import time as time_module

    ear = Ear()

    with patch.object(ear, '_open_brain_stream', return_value=True):
        with patch.object(ear, '_send_hud'):
            with patch.object(ear, '_start_volume_sender'):
                # Press mouse button
                ear.on_mouse_click(100, 100, sys.modules['pynput.mouse'].Button.right, pressed=True)

                # Wait only 0.5 seconds (below threshold)
                time_module.sleep(0.5)

                # Call the record loop tick
                ear._record_loop_tick()

                # Should NOT have started recording
                assert ear.is_recording is False, "is_recording should be False after 0.5s hold"
                assert ear._recording_from_hold is False, "_recording_from_hold should be False"


def test_record_loop_tick_skips_silence_finalize_in_no_streaming_mode(monkeypatch):
    monkeypatch.setattr("src.audio.ear.RECORDING_MODE", "no_streaming")
    ear = Ear()
    ear.is_recording = True
    ear._chunk_started_at = time.time() - 2.0
    ear._utterance_gate = Mock()
    ear._utterance_gate.has_speech_started.return_value = True
    ear._utterance_gate.should_finalize.return_value = True

    with patch.object(ear, "_stop_and_send") as mock_stop:
        ear._record_loop_tick()

    mock_stop.assert_not_called()


def test_record_loop_tick_finalizes_on_silence_in_silence_streaming_mode(monkeypatch):
    monkeypatch.setattr("src.audio.ear.RECORDING_MODE", "silence_streaming")
    monkeypatch.setattr("src.audio.ear.MIN_CHUNK_SECONDS_REQ_FOR_SPLITING_DUE_TO_SILENCE_STREAMING", 1.0)
    ear = Ear()
    ear.is_recording = True
    ear._chunk_started_at = time.time() - 2.0
    ear._utterance_gate = Mock()
    ear._utterance_gate.has_speech_started.return_value = True
    ear._utterance_gate.should_finalize.return_value = True
    ear._utterance_gate.silence_elapsed.return_value = 1.2

    with patch.object(ear, "_stop_and_send") as mock_stop:
        ear._record_loop_tick()

    mock_stop.assert_called_once_with(stop_session=False)
