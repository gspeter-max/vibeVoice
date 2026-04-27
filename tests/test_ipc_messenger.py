# tests/test_ipc_messenger.py
import pytest
import json
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

def test_parse_incoming_message_handles_session_commit():
    raw_message = b"CMD_SESSION_COMMIT:session456:1"
    result = parse_incoming_message(raw_message)
    assert result == {
        "command_type": "session_commit",
        "session_id": "session456",
        "recording_index": 1
    }

def test_parse_incoming_message_handles_session_event():
    payload = {"status": "ok", "value": 42}
    raw_message = b"CMD_SESSION_EVENT:session789:2\n\n" + json.dumps(payload).encode("utf-8")
    result = parse_incoming_message(raw_message)
    assert result == {
        "command_type": "session_event",
        "session_id": "session789",
        "recording_index": 2,
        "payload": payload
    }

def test_parse_incoming_message_handles_switch_model():
    raw_message = b"CMD_SWITCH_MODEL:nemotron-v1"
    result = parse_incoming_message(raw_message)
    assert result == {
        "command_type": "switch_model",
        "model_name": "nemotron-v1"
    }

def test_parse_incoming_message_handles_edge_cases():
    # 1. Malformed Audio Chunk (missing parts)
    res = parse_incoming_message(b"CMD_AUDIO_CHUNK:session123:0")
    assert res["command_type"] == "error"
    assert res["reason"] == "missing_separator" # Separator check happens first

    # 2. Malformed Audio Chunk (bad parts count)
    res = parse_incoming_message(b"CMD_AUDIO_CHUNK:session123:0\n\nbody")
    assert res["command_type"] == "error"
    assert res["reason"] == "bad_audio_chunk_format"

    # 3. Malformed Session Event (bad parts count)
    res = parse_incoming_message(b"CMD_SESSION_EVENT:session123\n\n{}")
    assert res["command_type"] == "error"
    assert res["reason"] == "bad_session_event_format"

    # 4. Malformed Session Commit (bad parts count)
    res = parse_incoming_message(b"CMD_SESSION_COMMIT:session123:0:extra")
    assert res["command_type"] == "error"
    assert res["reason"] == "bad_session_commit_format"

    # 5. Empty message
    assert parse_incoming_message(b"")["command_type"] == "raw_audio"
    assert parse_incoming_message(b"")["payload_bytes"] == b""

    # 6. Raw audio (no command prefix)
    assert parse_incoming_message(b"just some random audio bytes")["command_type"] == "raw_audio"
