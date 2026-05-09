"""Canonical IPC protocol helpers for Ear and Brain.

This module owns the exact byte format used on the local socket wire.
The rules here must stay stable because tests and runtime code rely on the
current command prefixes, field order, and separator bytes.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def format_audio_chunk_message(
    session_id: str,
    recording_index: int,
    sequence_number: int,
    audio_bytes: bytes,
) -> bytes:
    """Build the exact wire message for one audio chunk.

    The header format is:
    `CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQUENCE_NUMBER`

    The header is followed by two newline bytes and then the raw audio bytes.
    This function must not change spacing, separators, or field order because
    the receiver parses these bytes exactly.
    """

    header_string = (
        f"CMD_AUDIO_CHUNK:{session_id}:{recording_index}:{sequence_number}\n\n"
    )
    return header_string.encode("utf-8") + audio_bytes


def format_session_commit_message(session_id: str, recording_index: int) -> bytes:
    """Build the exact wire message that closes one recording.

    The commit message contains no JSON body and no separator bytes.
    It is a single command string encoded as UTF-8 bytes.
    """

    command_string = f"CMD_SESSION_COMMIT:{session_id}:{recording_index}"
    return command_string.encode("utf-8")


def format_session_event_message(
    session_id: str,
    recording_index: int,
    event_payload: dict,
) -> bytes:
    """Build the exact wire message for one telemetry or session event.

    The body is compact JSON with no extra whitespace. The header/body
    boundary is always two newline bytes.
    """

    header_string = f"CMD_SESSION_EVENT:{session_id}:{recording_index}\n\n"
    compact_json_body = json.dumps(event_payload, separators=(",", ":"))
    return header_string.encode("utf-8") + compact_json_body.encode("utf-8")


def format_switch_model_message(model_name: str) -> bytes:
    """Build the exact wire message for a model switch command."""

    command_string = f"CMD_SWITCH_MODEL:{model_name}"
    return command_string.encode("utf-8")


def parse_incoming_message(raw_bytes: bytes) -> Dict[str, Any]:
    """Parse a raw socket payload into the existing compatibility dictionary.

    The returned dictionary shape is intentionally unchanged so the Brain side
    and the tests can keep using the current contract during the refactor.
    Unknown bytes still fall back to the historical `raw_audio` behavior.
    """

    if raw_bytes.startswith(b"CMD_SWITCH_MODEL:"):
        try:
            full_text = raw_bytes.decode("utf-8").strip()
            _, model_name = full_text.split(":", 1)
            return {"command_type": "switch_model", "model_name": model_name}
        except Exception:
            return {"command_type": "error", "reason": "bad_switch_model_format"}

    if raw_bytes.startswith(b"CMD_SESSION_COMMIT:"):
        try:
            full_text = raw_bytes.decode("utf-8").strip()
            parts = full_text.split(":")
            if len(parts) == 3:
                return {
                    "command_type": "session_commit",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                }
            return {"command_type": "error", "reason": "bad_session_commit_format"}
        except Exception:
            return {"command_type": "error", "reason": "bad_session_commit_format"}

    if raw_bytes.startswith(b"CMD_SESSION_EVENT:") and b"\n\n" in raw_bytes:
        try:
            header_bytes, payload_bytes = raw_bytes.split(b"\n\n", 1)
            header_text = header_bytes.decode("utf-8").strip()
            parts = header_text.split(":")
            if len(parts) == 3:
                return {
                    "command_type": "session_event",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                    "payload": json.loads(payload_bytes.decode("utf-8")),
                }
            return {"command_type": "error", "reason": "bad_session_event_format"}
        except Exception:
            return {"command_type": "error", "reason": "bad_session_event_format"}

    if raw_bytes.startswith(b"CMD_AUDIO_CHUNK:"):
        if b"\n\n" not in raw_bytes:
            return {"command_type": "error", "reason": "missing_separator"}
        try:
            header_bytes, audio_data = raw_bytes.split(b"\n\n", 1)
            header_text = header_bytes.decode("utf-8").strip()
            parts = header_text.split(":")
            if len(parts) == 4:
                return {
                    "command_type": "audio_chunk",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                    "sequence_number": int(parts[3]),
                    "payload_bytes": audio_data,
                }
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}
        except Exception:
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}

    if not raw_bytes:
        return {"command_type": "raw_audio", "payload_bytes": b""}

    return {"command_type": "raw_audio", "payload_bytes": raw_bytes}
