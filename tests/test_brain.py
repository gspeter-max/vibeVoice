"""
test_brain.py — Unit tests for src/backend/brain.py

These tests verify that brain.py correctly:
  - Routes raw audio to the engine for transcription
  - Handles model switching commands
  - Stitches and insert_transcriptes final text after a recording session
  - Deduplicates overlapping text chunks from stateless engines

After Phase 2 wiring, brain.py no longer holds a separate 'backend' and 'model'.
It now holds a single 'engine' object that conforms to the TranscriptionEngine rulebook.
All mocks here use mock_engine with is_stateful() and transcribe_chunk() methods.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.backend.brain as brain
import src.backend.state as state
from src.backend.state import BackendState, RecordingState, SessionState, SessionStates
from src.utils.settings import settings
from tests.conftest import MockConn

# @pytest.fixture(autouse=True)
# def clear_session_store():
#     """
#     Runs before and after every test.
#     Clears the global session_store so tests don't bleed into each other.
#     """
#     state.session_store.clear()
#     yield
#     state.session_store.clear()
#


@pytest.mark.parametrize(
    ("cmd", "func"),
    [
        ("CMD_SWITCH_MODEL:base.en", "_handle_switch_model"),
        ("CMD_SESSION_COMMIT:base.en", "_handle_session_commit"),
        ("CMD_SESSION_EVENT:base.en  \n\n", "_handle_session_event"),
        ("CMD_AUDIO_CHUNK:base.en  \n\n", "_handle_chunk_command"),
    ],
)
def test_handle_connection(cmd, func):
    """
    Verify that when raw audio arrives (no CMD_ prefix), brain transcribes it
    using engine.transcribe_chunk(), cleans it, and insert_transcriptes the result.
    """
    state = SessionStates(
        backend=MagicMock(),
    )
    Conn = MockConn(cmd.encode("utf-8"))
    # func_mock = MagicMock()
    # file_func = getattr(brain, func)
    # types.MethodType(file_func, func_mock)
    with patch.object(brain, func, return_value=None) as event_mock:
        print(f"event_mock  : {type(event_mock)}")
        print(f"brain.{func} : --> {type(getattr(brain, func))}")
        brain.handle_connection(Conn, state)

    # print(f"event_mock  : {type(func_mock)}")
    # print(f"brain.{func} : --> {type(getattr(brain, func))}")
    brain.handle_connection(Conn, state)
    event_mock.assert_called_once_with(cmd.encode("utf-8"), state)


def test__transcribe_raw_connection_audio():
    """
    Verify that in 'no_streaming' mode, audio is buffered and then transcribed
    all at once using engine.transcribe_chunk().
    """
    mock_engine = MagicMock()
    mock_engine.transcribe_chunk.return_value = "hello world"
    state = SessionStates(
        backend=BackendState(engine=mock_engine, model_name="mock-model"),
    )
    sample_audio_bytes = "hello world"
    conn = MockConn(sample_audio_bytes.encode("utf-8"))

    with (
        patch.object(brain, "send_hud"),
        patch.object(brain, "refine", return_value="hello world cleaned") as mock_groq,
        patch.object(brain, "insert_transcripte") as mock_insert_transcripte,
    ):
        brain.handle_connection(conn, state)

    mock_engine.transcribe_chunk.assert_called_once()
    mock_groq.assert_called_once_with("hello world")
    mock_insert_transcripte.assert_called_once_with("hello world cleaned")


def test_handle_connection_short_audio():
    """
    Verify that audio that is all silence (too quiet) is skipped without calling
    engine.transcribe_chunk() at all.
    """
    mock_engine = MagicMock()
    state = SessionStates(
        backend=BackendState(engine=mock_engine, model_name="mock-model"),
    )
    # This audio is all zeros, which will be rejected as silence
    short_audio = b"\x00\x00" * 1600
    conn = MockConn(short_audio)

    with (
        patch.object(brain, "send_hud"),
        patch.object(brain, "insert_transcripte") as mock_insert_transcripte,
    ):
        result = brain.handle_connection(conn, state)

    assert result == None
    mock_engine.transcribe_chunk.assert_not_called()
    mock_insert_transcripte.assert_not_called()


def test_handle_streaming_audio_chunk_dedupes_against_last_chunk_text():
    """
    Verify that for a STATELESS engine, two sequential audio chunks are deduplicated.
    The second chunk's text should have the repeated prefix from chunk 1 removed.

    Dedup flow:
      Chunk 0 raw text: "I want to see that things are happening fine"
      Chunk 1 raw text: "things are happening fine and doing H3 grid"
      Chunk 1 cleaned : "doing H3 grid"  (repeated prefix removed)
    """
    mock_engine = MagicMock()
    state = SessionStates(
        backend=BackendState(engine=mock_engine, model_name="mock-model"),
    )
    # is_stateful() returns False = stateless engine (uses deduplication)
    mock_engine.is_stateful.return_value = False
    mock_engine.transcribe_chunk.side_effect = [
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    ]

    session_id = "session123"
    audio_bytes = b"\x00\x10" * 32000

    brain.handle_streaming_audio_chunk(session_id, 0, 0, audio_bytes, state)
    brain.handle_streaming_audio_chunk(session_id, 0, 1, audio_bytes, state)

    session = state.sessions.get(session_id, None)
    assert session != None
    assert (
        session.recordings[0].transcript_parts[0] == "I want to see that things are happening fine"
    )
    assert session.recordings[0].transcript_parts[1] == "doing H3 grid"


def test_finalize_session_insert_transcriptes_stitched_text_directly():
    """
    Verify that after all chunks arrive and the session is closed,
    finalize_recording stitches the parts, cleans them using Groq, and calls insert_transcripte.
    """
    session_id = "session123"
    mock_engine = MagicMock()
    state = SessionStates(backend=BackendState(engine=mock_engine))
    state.sessions[session_id] = SessionState()
    session = state.sessions.get(session_id, None)

    assert session != None
    # Set up a completed recording at index 0
    rec = RecordingState()
    rec.received_count = 2
    rec.done_count = 2
    rec.closed = True
    rec.transcript_parts[0] = "hello"
    rec.transcript_parts[1] = "world"
    session.recordings[0] = rec

    with (
        patch("src.backend.brain.log.info") as mock_log,
        patch.object(brain, "_show_summary_table") as mock_summary_table,
        patch.object(brain, "send_hud"),
        patch.object(brain, "refine", return_value="hello world cleaned") as mock_groq,
        patch.object(brain, "insert_transcripte") as mock_insert_transcripte,
    ):
        brain.finalize_recording(session_id, 0, state)
    mock_groq.assert_called_once_with("hello world")
    mock_summary_table.assert_called_once()
    summary_args = mock_summary_table.call_args.args
    assert summary_args[:4] == (
        session_id,
        "hello world",
        "hello world cleaned",
        0.0,
    )
    assert isinstance(summary_args[4], float)
    mock_insert_transcripte.assert_called_once_with("hello world cleaned ")
    assert not any("[Brain] 🏁" in call.args[0] for call in mock_log.call_args_list if call.args)


def test_handle_session_event_writes_telemetry_file(tmp_path, monkeypatch):
    """
    Verify that a CMD_SESSION_EVENT triggers writing a telemetry JSON file to disk.
    """
    import src.backend.data_record.telemetry as telemetry

    # These constants were moved to settings — patch the settings object
    monkeypatch.setattr(telemetry.settings, "streaming_telemetry_enabled", True)
    monkeypatch.setattr(telemetry.settings, "streaming_telemetry_dir", tmp_path)
    # Set a mock engine in global state so telemetry seed can read the model name
    mock_engine = MagicMock()
    mock_engine.model_name = "base.en"
    state = SessionStates(backend=BackendState(engine=mock_engine))

    audio = (
        b"CMD_SESSION_EVENT:session123:0\n\n"
        b'{"type":"vad_no_speech_warning","chunk_index":0,"max_score":0.017,"threshold":0.5,"last_energy":0.0074,"energy_threshold":0.05}'
    )

    brain._handle_session_event(audio, state)
    json_files = list(tmp_path.glob("*session123*.json"))
    assert len(json_files) == 1
    payload = json_files[0].read_text(encoding="utf-8")
    assert '"vad_no_speech_warning_seen": true' in payload


def test_handle_streaming_audio_chunk_stateful_engine(monkeypatch):
    """
    Verify that handle_streaming_audio_chunk stores cumulative
    transcription text in transcript_parts when using a stateful engine.
    """
    with patch.object(brain, "finalize_recording", return_value=None) as mock_finalize, \
         patch.object(brain, "_normalize_audio") as mock_norm:

        monkeypatch.setattr(settings, "streaming_telemetry_enabled", False)

        session_id = "session1"
        mock_engine = MagicMock()
        mock_engine.is_stateful.return_value = True
        mock_engine.transcribe_chunk.return_value = "cumulative text"

        rec_mocker = MagicMock()
        rec_mocker.received_count = 0

        state = SessionStates(backend=BackendState(engine=mock_engine))
        session_mocker = MagicMock()
        session_mocker.get_or_create_recording.return_value = rec_mocker

        state.sessions = {session_id: session_mocker}
        mock_norm.return_value = np.zeros(1600)

        brain.handle_streaming_audio_chunk(session_id, 0, 0, b"\x00\x00" * 1600, state)

        mock_engine.transcribe_chunk.assert_called_once()
        assert rec_mocker.transcript_parts[0] == "cumulative text"
        mock_finalize.assert_called_once()


def test_finalize_recording_always_clears_engine_memory():
    """
    Verify that finalize_recording always calls engine.clear_internal_memory()
    after finalizing — regardless of whether the engine is stateful or stateless.
    (Stateless engines safely do nothing when clear_internal_memory is called.)
    """
    with patch.object(brain, "refine") as mock_refine, \
         patch.object(brain, "send_hud") as mock_hud, \
         patch.object(brain, "insert_transcripte") as mock_insert_transcripte:

        session_id = "session1"

        # Create a mock engine with a trackable clear_internal_memory method
        mock_engine = MagicMock()

        mock_rec = RecordingState()
        mock_rec.finalized = False
        mock_rec.closed = True
        mock_rec.done_count = 1
        mock_rec.received_count = 1
        mock_rec.stt_time = 0.5
        mock_rec.transcript_parts = {0: "final text"}

        session = SessionState(recordings={0: mock_rec})
        state = SessionStates(backend=BackendState(engine=mock_engine), sessions={session_id: session})
        brain.finalize_recording(session_id, 0, state)

        # clear_internal_memory must have been called exactly once after finalization
        mock_engine.clear_internal_memory.assert_called_once()
        mock_insert_transcripte.assert_called_once()
        mock_refine.assert_called_once()
        assert mock_hud.call_count == 2
