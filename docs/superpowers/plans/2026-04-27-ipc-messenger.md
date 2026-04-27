# IPC Messenger Extraction Implementation Plan (Phase 1: Construction Only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent** : 
- We are building a new dedicated module `src/ipc/messenger.py` to handle all IPC (socket and messaging) logic for the application.
- **CRITICAL ARCHITECTURE RULE:** You are operating in a parallel worktree. **DO NOT MODIFY `src/audio/ear.py` OR `src/backend/brain.py`.** Your job is ONLY to create the new module and its tests. The integration will happen in a later phase.
- The protocol must remain EXACTLY the same as the current system, handling these formats:
    - `CMD_AUDIO_CHUNK:SESSION_ID:RECORDING_INDEX:SEQ\n\n<audio_bytes>`
    - `CMD_SESSION_COMMIT:SESSION_ID:RECORDING_INDEX`
    - `CMD_SESSION_EVENT:SESSION_ID:RECORDING_INDEX\n\n<json_payload>`
    - `CMD_SWITCH_MODEL:MODEL_NAME`
- **CRITICAL:** Do not assume anything, strictly follow the plan, and ask questions if you don't understand anything. Read the existing files if you need to understand the current payload shapes.

**Architecture:**
- Create `src/ipc/messenger.py`. This acts as the "Postman" between Ear and Brain. 
- Exposes formatting functions: `format_audio_chunk_message()`, `format_session_commit_message()`, `format_session_event_message()`, `format_switch_model_message()`.
- Exposes sending function: `send_message_to_brain()`.
- Exposes parsing function: `parse_incoming_message(raw_bytes)`.

**Important Rule to follow :**
- **CRITICAL: ** add detailed docs in functions and explain the code and logic in comments.  
- **CRITICAL:** make the code function name and variable name clear and easily to understand instead of short and confusing names.
- write code function name and docs and code like this: **developer gets highest speed to read the code**
- **Explain like a fresher** 

---
## Task Structure

### Task 1 : Read out instruction file
- [ ] read `/Users/apple/.gemini/GEMINI.md` file

### Task 2: The Messenger Module

**Files:**
- Create: `src/ipc/__init__.py`
- Create: `src/ipc/messenger.py`
- Create: `tests/test_ipc_messenger.py`

- [ ] **Step 1: Write the failing tests**
```python
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
    assert result == b"CMD_AUDIO_CHUNK:session123:0:5\\n\\naudio"

def test_format_session_commit_message():
    result = format_session_commit_message("session123", 0)
    assert result == b"CMD_SESSION_COMMIT:session123:0"

def test_format_session_event_message():
    result = format_session_event_message("session123", 0, {"type": "test"})
    assert result == b'CMD_SESSION_EVENT:session123:0\\n\\n{"type":"test"}'

def test_format_switch_model_message():
    result = format_switch_model_message("parakeet-v2")
    assert result == b"CMD_SWITCH_MODEL:parakeet-v2"

def test_parse_incoming_message_handles_audio_chunk():
    raw_message = b"CMD_AUDIO_CHUNK:session123:0:5\\n\\naudio"
    result = parse_incoming_message(raw_message)
    assert result == {
        "command_type": "audio_chunk",
        "session_id": "session123",
        "recording_index": 0,
        "sequence_number": 5,
        "payload_bytes": b"audio"
    }

def test_parse_incoming_message_handles_edge_cases():
    assert parse_incoming_message(b"CMD_AUDIO_CHUNK:session123:0")["command_type"] == "error"
    assert parse_incoming_message(b"CMD_AUDIO_CHUNK:session123:0:5\\n\\n")["payload_bytes"] == b""
    assert parse_incoming_message(b"")["command_type"] == "raw_audio"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ipc_messenger.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation for __init__.py**
```python
# src/ipc/__init__.py
"""
The IPC (Inter-Process Communication) package.
Handles all networking and message formatting between different parts of the application.
"""
```

- [ ] **Step 4: Write implementation for messenger.py**
```python
# src/ipc/messenger.py
import json
import socket
from typing import Dict, Any

SOCKET_PATH = "/tmp/parakeet.sock"

def format_audio_chunk_message(session_id: str, recording_index: int, sequence_number: int, audio_bytes: bytes) -> bytes:
    """
    Creates the raw byte message for an audio chunk.
    This tells the Brain 'Here is a piece of audio for this specific recording'.
    """
    header = f"CMD_AUDIO_CHUNK:{session_id}:{recording_index}:{sequence_number}\\n\\n".encode("utf-8")
    return header + audio_bytes

def format_session_commit_message(session_id: str, recording_index: int) -> bytes:
    """
    Creates the raw byte message to finish a recording.
    This tells the Brain 'The user has stopped talking, you can paste the text now'.
    """
    return f"CMD_SESSION_COMMIT:{session_id}:{recording_index}".encode("utf-8")

def format_session_event_message(session_id: str, recording_index: int, event_payload: dict) -> bytes:
    """
    Creates the raw byte message for a telemetry event (like a volume change or silence warning).
    """
    header = f"CMD_SESSION_EVENT:{session_id}:{recording_index}\\n\\n".encode("utf-8")
    body = json.dumps(event_payload, separators=(",", ":")).encode("utf-8")
    return header + body

def format_switch_model_message(model_name: str) -> bytes:
    """
    Creates the raw byte message to change the AI model.
    """
    return f"CMD_SWITCH_MODEL:{model_name}".encode("utf-8")

def send_message_to_brain(message_bytes: bytes, timeout_seconds: float = 5.0) -> bool:
    """
    Opens a socket to the Brain, sends the exact message bytes, and closes the socket.
    Returns True if successful, False if the connection failed.
    """
    if not message_bytes:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(SOCKET_PATH)
            client.sendall(message_bytes)
            client.shutdown(socket.SHUT_WR)
        return True
    except Exception:
        return False

def parse_incoming_message(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Reads the raw bytes received from the Ear and turns them into a simple dictionary.
    This makes it very easy for the Brain to understand what the Ear wants.
    """
    if raw_bytes.startswith(b"CMD_SWITCH_MODEL:"):
        try:
            model_name = raw_bytes.decode("utf-8").strip().split(":", 1)[1]
            return {"command_type": "switch_model", "model_name": model_name}
        except Exception:
            return {"command_type": "error", "reason": "bad_switch_model_format"}

    if raw_bytes.startswith(b"CMD_SESSION_COMMIT:"):
        try:
            _, session_id, rec_idx_str = raw_bytes.decode("utf-8").strip().split(":", 2)
            return {
                "command_type": "session_commit",
                "session_id": session_id,
                "recording_index": int(rec_idx_str)
            }
        except Exception:
            return {"command_type": "error", "reason": "bad_session_commit_format"}

    if raw_bytes.startswith(b"CMD_SESSION_EVENT:") and b"\\n\\n" in raw_bytes:
        try:
            header, payload_blob = raw_bytes.split(b"\\n\\n", 1)
            _, session_id, rec_idx_str = header.decode("utf-8").strip().split(":", 2)
            payload = json.loads(payload_blob.decode("utf-8"))
            return {
                "command_type": "session_event",
                "session_id": session_id,
                "recording_index": int(rec_idx_str),
                "payload": payload
            }
        except Exception:
            return {"command_type": "error", "reason": "bad_session_event_format"}

    if raw_bytes.startswith(b"CMD_AUDIO_CHUNK:") and b"\\n\\n" in raw_bytes:
        try:
            header, audio_bytes = raw_bytes.split(b"\\n\\n", 1)
            _, session_id, rec_idx_str, seq_text = header.decode("utf-8").strip().split(":", 3)
            return {
                "command_type": "audio_chunk",
                "session_id": session_id,
                "recording_index": int(rec_idx_str),
                "sequence_number": int(seq_text),
                "payload_bytes": audio_bytes
            }
        except Exception:
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}

    return {"command_type": "raw_audio", "payload_bytes": raw_bytes}
```

- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/test_ipc_messenger.py -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/ipc/ tests/test_ipc_messenger.py
git commit -m "feat: add ipc messenger module with exact protocol logic"
```
