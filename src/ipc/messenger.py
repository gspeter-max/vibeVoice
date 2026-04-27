# src/ipc/messenger.py
import json
import socket
from typing import Dict, Any

# This is the path to the Unix socket file. 
# Think of it like a temporary address where the Brain listens for mail.
SOCKET_PATH = "/tmp/parakeet.sock"

def format_audio_chunk_message(session_id: str, recording_index: int, sequence_number: int, audio_bytes: bytes) -> bytes:
    """
    Creates the raw byte message for an audio chunk.
    
    This function takes audio data and wraps it in a header so the Brain knows:
    1. Which session it belongs to.
    2. Which recording index it is part of.
    3. What order the chunk should be in (sequence number).

    Parameters:
        session_id: The unique ID of the current user session.
        recording_index: The count of how many times the user has recorded in this session.
        sequence_number: The order of this specific audio slice.
        audio_bytes: The actual raw audio data.

    Returns:
        A package of bytes ready to be sent over the socket.
    """
    # Create the header string: CMD_AUDIO_CHUNK:SESSION:INDEX:SEQ
    # We add two newlines (\n\n) to separate the header from the actual data.
    header_string = f"CMD_AUDIO_CHUNK:{session_id}:{recording_index}:{sequence_number}\n\n"
    
    # Convert the string to bytes using UTF-8 encoding
    header_bytes = header_string.encode("utf-8")
    
    # Combine the header bytes with the actual audio data bytes
    return header_bytes + audio_bytes

def format_session_commit_message(session_id: str, recording_index: int) -> bytes:
    """
    Creates the raw byte message to finish a recording.
    
    This tells the Brain: 'The user has stopped talking for this specific recording. 
    You can now finalize the processing and paste the text.'

    Parameters:
        session_id: The unique ID of the current user session.
        recording_index: The count of how many times the user has recorded.

    Returns:
        A byte string that acts as a "Commit" command.
    """
    # Simple command string: CMD_SESSION_COMMIT:SESSION:INDEX
    command_string = f"CMD_SESSION_COMMIT:{session_id}:{recording_index}"
    return command_string.encode("utf-8")

def format_session_event_message(session_id: str, recording_index: int, event_payload: dict) -> bytes:
    """
    Creates the raw byte message for a telemetry or status event.
    
    This is used to send updates like 'volume changed' or 'silence detected' to the Brain.
    
    Parameters:
        session_id: The unique ID of the current user session.
        recording_index: The count of how many times the user has recorded.
        event_payload: A dictionary containing the event details.

    Returns:
        A package of bytes with a header and a JSON-formatted body.
    """
    # Create the header for an event
    header_string = f"CMD_SESSION_EVENT:{session_id}:{recording_index}\n\n"
    header_bytes = header_string.encode("utf-8")
    
    # Convert the dictionary (event_payload) into a JSON string
    # We use separators=(",", ":") to make the JSON as small as possible (no spaces)
    json_body = json.dumps(event_payload, separators=(",", ":"))
    body_bytes = json_body.encode("utf-8")
    
    return header_bytes + body_bytes

def format_switch_model_message(model_name: str) -> bytes:
    """
    Creates the raw byte message to change the AI model being used.
    
    Parameters:
        model_name: The name of the model we want to switch to (e.g., 'parakeet-v2').

    Returns:
        A byte string that tells the Brain to switch models.
    """
    command_string = f"CMD_SWITCH_MODEL:{model_name}"
    return command_string.encode("utf-8")

def send_message_to_brain(message_bytes: bytes, timeout_seconds: float = 5.0) -> bool:
    """
    Connects to the Brain via a socket and sends the message.
    
    Think of this like making a quick phone call to the Brain to deliver a message.
    1. Open the connection.
    2. Say the message.
    3. Hang up.

    Parameters:
        message_bytes: The exact bytes we want to send.
        timeout_seconds: How long to wait for a connection before giving up.

    Returns:
        True if the message was sent successfully, False if anything went wrong.
    """
    # If there is nothing to send, just return False
    if not message_bytes:
        return False
        
    try:
        # Create a new socket. 
        # AF_UNIX means we are talking between programs on the same computer.
        # SOCK_STREAM means we want a reliable connection (like a phone call).
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client_socket:
            # Set a time limit so we don't wait forever if the Brain is stuck
            client_socket.settimeout(timeout_seconds)
            
            # Try to connect to the Brain's address
            client_socket.connect(SOCKET_PATH)
            
            # Send all the bytes in the message
            client_socket.sendall(message_bytes)
            
            # Tell the Brain we are done sending data
            client_socket.shutdown(socket.SHUT_WR)
            
        return True
    except Exception as error:
        # If any error happens (like the Brain is not running), we log nothing and return False
        return False

def parse_incoming_message(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Reads raw bytes and converts them back into a useful Python dictionary.
    
    This is the "Unpacker" that the Brain uses to understand what the Ear sent.
    
    Parameters:
        raw_bytes: The raw data received from the socket.

    Returns:
        A dictionary with a 'command_type' and relevant data fields.
    """
    # 1. Handle Model Switching
    if raw_bytes.startswith(b"CMD_SWITCH_MODEL:"):
        try:
            # Decode the bytes to a string and grab the part after the colon
            full_text = raw_bytes.decode("utf-8").strip()
            _, model_name = full_text.split(":", 1)
            return {"command_type": "switch_model", "model_name": model_name}
        except Exception:
            return {"command_type": "error", "reason": "bad_switch_model_format"}

    # 2. Handle Session Commits (Finalizing a recording)
    if raw_bytes.startswith(b"CMD_SESSION_COMMIT:"):
        try:
            full_text = raw_bytes.decode("utf-8").strip()
            parts = full_text.split(":")
            # Parts should be [CMD, SESSION_ID, RECORDING_INDEX]
            if len(parts) == 3:
                return {
                    "command_type": "session_commit",
                    "session_id": parts[1],
                    "recording_index": int(parts[2])
                }
            else:
                return {"command_type": "error", "reason": "bad_session_commit_format"}
        except Exception:
            return {"command_type": "error", "reason": "bad_session_commit_format"}

    # 3. Handle Session Events (Telemetry)
    if raw_bytes.startswith(b"CMD_SESSION_EVENT:") and b"\n\n" in raw_bytes:
        try:
            # Split the message at the double newline
            header_bytes, payload_bytes = raw_bytes.split(b"\n\n", 1)
            header_text = header_bytes.decode("utf-8").strip()
            parts = header_text.split(":")
            
            # Parse the JSON payload back into a dictionary
            payload_dict = json.loads(payload_bytes.decode("utf-8"))
            
            return {
                "command_type": "session_event",
                "session_id": parts[1],
                "recording_index": int(parts[2]),
                "payload": payload_dict
            }
        except Exception:
            return {"command_type": "error", "reason": "bad_session_event_format"}

    # 4. Handle Audio Chunks (The most common message)
    if raw_bytes.startswith(b"CMD_AUDIO_CHUNK:"):
        if b"\n\n" not in raw_bytes:
            return {"command_type": "error", "reason": "missing_separator"}
        try:
            # Split into header and audio data
            header_bytes, audio_data = raw_bytes.split(b"\n\n", 1)
            header_text = header_bytes.decode("utf-8").strip()
            parts = header_text.split(":")
            
            # Parts: [CMD, SESSION_ID, RECORDING_INDEX, SEQUENCE_NUMBER]
            if len(parts) == 4:
                return {
                    "command_type": "audio_chunk",
                    "session_id": parts[1],
                    "recording_index": int(parts[2]),
                    "sequence_number": int(parts[3]),
                    "payload_bytes": audio_data
                }
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}
        except Exception:
            return {"command_type": "error", "reason": "bad_audio_chunk_format"}

    # 5. Default Fallback
    # If it's empty or doesn't match a command, we treat it as raw audio or return an error
    if not raw_bytes:
        return {"command_type": "raw_audio", "payload_bytes": b""}
        
    return {"command_type": "raw_audio", "payload_bytes": raw_bytes}
