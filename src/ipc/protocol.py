"""IPC wire protocol helpers for message formatting and parsing."""

from __future__ import annotations

import json
from typing import Any


def fmt_audio(
    session_id: str,
    recording_index: int,
    sequence_number: int,
    audio_bytes: bytes,
) -> bytes:
    """Build wire bytes for an audio chunk."""
    header = f"CMD_AUDIO_CHUNK:{session_id}:{recording_index}:{sequence_number}\n\n"
    return header.encode("utf-8") + audio_bytes


def fmt_commit(session_id: str, recording_index: int) -> bytes:
    """Build wire bytes to commit a recording session."""
    return f"CMD_SESSION_COMMIT:{session_id}:{recording_index}".encode("utf-8")


def fmt_event(
    session_id: str,
    recording_index: int,
    event_payload: dict,
) -> bytes:
    """Build wire bytes for a telemetry event."""
    header = f"CMD_SESSION_EVENT:{session_id}:{recording_index}\n\n"
    body = json.dumps(event_payload, separators=(",", ":"))
    return header.encode("utf-8") + body.encode("utf-8")


def fmt_switch(model_name: str) -> bytes:
    """Build wire bytes for switching transcription models."""
    return f"CMD_SWITCH_MODEL:{model_name}".encode("utf-8")


def parse_msg(raw_bytes: bytes) -> dict[str, Any]:
    """Parse raw wire bytes into a command dictionary."""
    if raw_bytes.startswith(b"CMD_SWITCH_MODEL:"):
        try:
            _, model_name = raw_bytes.decode("utf-8").strip().split(":", 1)
            return {"command_type": "switch_model", "model_name": model_name}
        except ValueError:
            return {"command_type": "error", "reason": "bad_switch_model_format"}

    if raw_bytes.startswith(b"CMD_SESSION_COMMIT:"):
        try:
            parts = raw_bytes.decode("utf-8").strip().split(":")
            if len(parts) == 3:
                return {
                    "command_type": "session_commit",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                }
            return {"command_type": "error", "reason": "bad_session_commit_format"}
        except ValueError:
            return {"command_type": "error", "reason": "bad_session_commit_format"}

    if raw_bytes.startswith(b"CMD_SESSION_EVENT:") and b"\n\n" in raw_bytes:
        try:
            header_bytes, payload_bytes = raw_bytes.split(b"\n\n", 1)
            parts = header_bytes.decode("utf-8").strip().split(":")
            if len(parts) == 3:
                return {
                    "command_type": "session_event",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                    "payload": json.loads(payload_bytes.decode("utf-8")),
                }
            return {"command_type": "error", "reason": "bad_session_event_format"}
        except ValueError:
            return {"command_type": "error", "reason": "bad_session_event_format"}

    if raw_bytes.startswith(b"CMD_AUDIO_CHUNK:"):
        if b"\n\n" not in raw_bytes:
            return {"command_type": "error", "reason": "missing_separator"}
        try:
            header_bytes, audio_data = raw_bytes.split(b"\n\n", 1)
            parts = header_bytes.decode("utf-8").strip().split(":")
            if len(parts) == 4:
                return {
                    "command_type": "audio_chunk",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                    "sequence_number": int(parts[3]),
                    "payload_bytes": audio_data,
                }
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}
        except ValueError:
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}

    if not raw_bytes:
        return {"command_type": "raw_audio", "payload_bytes": b""}

    return {"command_type": "raw_audio", "payload_bytes": raw_bytes}
