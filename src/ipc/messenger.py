"""Compatibility wrapper around the canonical IPC protocol module.

Tests and older imports still import these helpers from `src.ipc.messenger`.
During the refactor this file stays as the stable import surface while the real
protocol authority lives in `src.ipc.protocol`.
"""

from src.ipc.client import SOCKET_PATH, send_message_to_brain
from src.ipc.protocol import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
    format_switch_model_message,
    parse_incoming_message,
)
