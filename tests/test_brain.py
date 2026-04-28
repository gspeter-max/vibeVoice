"""
test_brain.py — Unit tests for src/backend/brain.py

These tests verify that brain.py correctly:
  - Routes raw audio to the engine for transcription
  - Handles model switching commands
  - Stitches and pastes final text after a recording session
  - Deduplicates overlapping text chunks from stateless engines

After Phase 2 wiring, brain.py no longer holds a separate 'backend' and 'model'.
It now holds a single 'engine' object that conforms to the TranscriptionEngine rulebook.
All mocks here use mock_engine with is_stateful() and transcribe_chunk() methods.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.backend.brain as brain
import src.backend.state as state
from src.streaming.streaming_shared_logic import remove_duplicate_chunk_prefix


class MockConn:
    """
    A fake socket connection used in tests.
    It returns audio chunks one at a time, and returns an empty
    bytes object at the end to signal that the connection is closed.
    """
    def __init__(self, *chunks):
        # Store the chunks and add an empty bytes sentinel at the end
        self._chunks = list(chunks) + [b""]

    def settimeout(self, _timeout):
        return None

    def recv(self, _size):
        return self._chunks.pop(0)

    def close(self):
        return None


@pytest.fixture(autouse=True)
def clear_session_store():
    """
    Runs before and after every test.
    Clears the global session_store so tests don't bleed into each other.
    """
    state.session_store.clear()
    yield
    state.session_store.clear()


def test_handle_connection_transcribes_audio(sample_audio_bytes):
    """
    Verify that when raw audio arrives (no CMD_ prefix), brain transcribes it
    using engine.transcribe_chunk() and pastes the result.
    """
    # Create a mock engine that returns "hello world" when asked to transcribe
    mock_engine = MagicMock()
    mock_engine.transcribe_chunk.return_value = "hello world"

    # Set the engine in brain's global state
    state.backend_info["engine"] = mock_engine

    conn = MockConn(sample_audio_bytes)

    with (
        patch("src.backend.brain.send_hud") as mock_hud,
        patch("src.backend.brain.paste_instantly") as mock_paste,
    ):
        brain.handle_connection(conn)

    # The engine must have been asked to transcribe exactly once
    mock_engine.transcribe_chunk.assert_called_once()
    # The transcribed text should have been pasted with a trailing space
    mock_paste.assert_called_once_with("hello world ")
    assert any(call.args[0] == "done" for call in mock_hud.call_args_list)


def test_handle_connection_no_streaming_buffers_raw_audio_until_socket_close(
    sample_audio_bytes, monkeypatch
):
    """
    Verify that in 'no_streaming' mode, audio is buffered and then transcribed
    all at once using engine.transcribe_chunk().
    """
    monkeypatch.setattr(brain, "RECORDING_MODE", "no_streaming")

    mock_engine = MagicMock()
    mock_engine.transcribe_chunk.return_value = "hello world"

    state.backend_info["engine"] = mock_engine

    conn = MockConn(sample_audio_bytes)

    with patch("src.backend.brain.send_hud"), patch("src.backend.brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_engine.transcribe_chunk.assert_called_once()
    mock_paste.assert_called_once_with("hello world ")


def test_handle_connection_switch_model_command():
    """
    Verify that when brain receives a CMD_SWITCH_MODEL command, it calls
    load_transcription_engine with the new model name and stores the result
    in engine_info['engine'].
    """
    mock_old_engine = MagicMock()
    mock_new_engine = MagicMock()

    state.backend_info["engine"] = mock_old_engine

    conn = MockConn(b"CMD_SWITCH_MODEL:tiny.en")

    with patch(
        "src.backend.brain.load_transcription_engine", return_value=mock_new_engine
    ) as mock_load:
        brain.handle_connection(conn)

    mock_load.assert_called_once_with("tiny.en")
    # The engine stored in brain must now be the new one
    assert state.backend_info["engine"] == mock_new_engine


def test_handle_connection_skips_too_short_audio():
    """
    Verify that audio that is all silence (too quiet) is skipped without calling
    engine.transcribe_chunk() at all.
    """
    mock_engine = MagicMock()
    state.backend_info["engine"] = mock_engine

    # This audio is all zeros, which will be rejected as silence
    short_audio = b"\x00\x00" * 1600
    conn = MockConn(short_audio)

    with patch("src.backend.brain.send_hud"), patch("src.backend.brain.paste_instantly") as mock_paste:
        brain.handle_connection(conn)

    mock_engine.transcribe_chunk.assert_not_called()
    mock_paste.assert_not_called()


def test_dedupe_with_last_chunk_removes_repeated_prefix():
    """
    Direct unit test for the deduplication helper function.
    Verifies that overlapping words from the previous chunk are trimmed off.
    """
    cleaned = remove_duplicate_chunk_prefix(
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    )

    assert cleaned == "doing H3 grid"


def test_handle_audio_chunk_dedupes_against_last_chunk_text():
    """
    Verify that for a STATELESS engine, two sequential audio chunks are deduplicated.
    The second chunk's text should have the repeated prefix from chunk 1 removed.

    Dedup flow:
      Chunk 0 raw text: "I want to see that things are happening fine"
      Chunk 1 raw text: "things are happening fine and doing H3 grid"
      Chunk 1 cleaned : "doing H3 grid"  (repeated prefix removed)
    """
    mock_engine = MagicMock()
    # is_stateful() returns False = stateless engine (uses deduplication)
    mock_engine.is_stateful.return_value = False
    mock_engine.transcribe_chunk.side_effect = [
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    ]

    state.backend_info["engine"] = mock_engine
    session_id = "session123"
    audio_bytes = b"\x00\x10" * 32000

    brain._handle_audio_chunk(session_id, 0, 0, audio_bytes)
    brain._handle_audio_chunk(session_id, 0, 1, audio_bytes)

    session = state.session_store[session_id]
    assert (
        session.recordings[0].transcript_parts[0]
        == "I want to see that things are happening fine"
    )
    assert session.recordings[0].transcript_parts[1] == "doing H3 grid"


def test_finalize_session_pastes_stitched_text_directly():
    """
    Verify that after all chunks arrive and the session is closed,
    _finalize_recording_if_ready stitches the parts and calls paste_instantly.
    """
    session_id = "session123"
    mock_engine = MagicMock()
    # SessionState now takes 'engine', not 'backend' and 'model'
    session = state.SessionState(engine=mock_engine)
    # Set up a completed recording at index 0
    rec = state.RecordingState()
    rec.received_count = 2
    rec.done_count = 2
    rec.closed = True
    rec.transcript_parts[0] = "hello"
    rec.transcript_parts[1] = "world"
    session.recordings[0] = rec
    state.session_store[session_id] = session

    with (
        patch("src.backend.brain.log.info") as mock_log,
        patch("src.backend.brain.send_hud"),
        patch("src.backend.brain.paste_instantly") as mock_paste,
    ):
        brain._finalize_recording_if_ready(session_id, 0)

    mock_paste.assert_called_once_with("hello world ")
    logged_messages = [call.args[0] for call in mock_log.call_args_list]
    assert any("[Brain] 🏁" in message for message in logged_messages)


def test_handle_session_event_writes_telemetry_file(tmp_path, monkeypatch):
    """
    Verify that a CMD_SESSION_EVENT triggers writing a telemetry JSON file to disk.
    """
    monkeypatch.setattr(brain, "STREAMING_TELEMETRY_ENABLED", True)
    monkeypatch.setattr(brain, "STREAMING_TELEMETRY_DIR", tmp_path)
    # Set a mock engine in global state so telemetry seed can read the model name
    mock_engine = MagicMock()
    mock_engine.model_name = "base.en"
    state.backend_info["engine"] = mock_engine

    blob = (
        b"CMD_SESSION_EVENT:session123:0\n\n"
        b'{"type":"vad_no_speech_warning","chunk_index":0,"max_score":0.017,"threshold":0.5,"last_energy":0.0074,"energy_threshold":0.05}'
    )

    brain._handle_session_event(blob)

    json_files = list(tmp_path.glob("*session123*.json"))
    assert len(json_files) == 1
    payload = json_files[0].read_text(encoding="utf-8")
    assert '"vad_no_speech_warning_seen": true' in payload


def test_brain_logs_match_pulse_format(capsys):
    """
    Verify that Brain logs follow the 'The Pulse' design.
    - Start: '[Brain] 🎙️  Recording started...'
    - End: '[Brain] 🏁 "text" (Xs)'
    """
    from unittest.mock import MagicMock
    import src.backend.brain as brain
    import src.backend.state as state
    import numpy as np

    mock_engine = MagicMock()
    mock_engine.is_stateful.return_value = False
    mock_engine.transcribe_chunk.return_value = "Hello"

    # Setup session
    session_id = "pulse_test_session"
    state.backend_info["engine"] = mock_engine
    
    # 1. Send first chunk (non-silent)
    brain._handle_audio_chunk(session_id, 0, 0, b"\x01\x10" * 1600)
    # 2. Commit session (needed for finalization)
    session = brain._get_or_create_session(session_id)
    rec = session.get_or_create_recording(0)
    rec.closed = True
    # 3. Finalize
    brain._finalize_recording_if_ready(session_id, 0)

    captured = capsys.readouterr()
    stdout = captured.out

    assert "[Brain] 🎙️  Recording started..." in stdout
    assert '[Brain] 🏁 "Hello" (' in stdout
    assert "s)" in stdout
