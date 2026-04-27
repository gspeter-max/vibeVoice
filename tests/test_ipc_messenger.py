# tests/test_ipc_messenger.py
import pytest
from src.ipc.messenger import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
    format_switch_model_message,
    parse_incoming_message
)

def test_format_audio_chunk_message_creates_correct_header():
    result = format_audio_chunk_message("session123", 0, 5, b"audio")
    assert result == b"CMD_AUDIO_CHUNK:session123:0:5\n\naudio"

def test_format_session_commit_message():
    result = format_session_commit_message("session123", 0)
    assert result == b"CMD_SESSION_COMMIT:session123:0"

def test_format_session_event_message():
    result = format_session_event_message("session123", 0, {"type": "test"})
    assert result == b'CMD_SESSION_EVENT:session123:0\n\n{"type":"test"}'

def test_format_switch_model_message():
    result = format_switch_model_message("parakeet-v2")
    assert result == b"CMD_SWITCH_MODEL:parakeet-v2"

def test_parse_incoming_message_handles_audio_chunk():
    raw_message = b"CMD_AUDIO_CHUNK:session123:0:5\n\naudio"
    result = parse_incoming_message(raw_message)
    assert result == {
        "command_type": "audio_chunk",
        "session_id": "session123",
        "recording_index": 0,
        "sequence_number": 5,
        "payload_bytes": b"audio"
    }

def test_parse_incoming_message_handles_edge_cases():
    # We check if it returns error for malformed messages
    assert parse_incoming_message(b"CMD_AUDIO_CHUNK:session123:0")["command_type"] == "error"
    assert parse_incoming_message(b"CMD_AUDIO_CHUNK:session123:0:5\n\n")["payload_bytes"] == b""
    assert parse_incoming_message(b"")["command_type"] == "raw_audio"
