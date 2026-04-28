import os
from pathlib import Path
from src.utils.env_utils import get_float_from_environment
from src.streaming.streaming_session_telemetry import StreamingSessionTelemetryRecorder
from src.backend.state import (
    backend_info, session_store, session_store_lock, SessionState
)

# Constants
BACKEND = os.environ.get("BACKEND", "parakeet").lower().strip()
RECORDING_MODE = os.environ.get("RECORDING_MODE", "silence_streaming").strip().lower()
STREAMING_TELEMETRY_ENABLED = (
    os.environ.get("STREAMING_TELEMETRY_ENABLED", "0").strip() == "1"
)
STREAMING_TELEMETRY_DIR = Path(
    os.environ.get("STREAMING_TELEMETRY_DIR", "logs/streaming_sessions")
)


def _model_name_for_telemetry() -> str | None:
    """
    Returns the active model name for labeling telemetry sessions.
    This function checks the currently loaded backend and model to extract
    a human-readable name, such as 'base.en' or 'parakeet-tdt'. This ensures
    that every saved telemetry file can clearly show which AI model was used
    to generate the transcriptions during that specific session.
    """
    engine = backend_info.get("engine")
    if engine is None:
        return None
    return getattr(engine, "model_name", "unknown_model")


def _telemetry_seed() -> dict:
    """
    Constructs the initial configuration data for a new telemetry session.
    This dictionary includes the recording mode, current backend type, and model name,
    as well as all the VAD (Voice Activity Detection) parameters used in the session.
    By seeding this data at the start, we create a comprehensive 'header' in the
    telemetry file that documents the exact settings and hardware state for debugging.
    """
    return {
        "recording_mode": RECORDING_MODE,
        "backend": BACKEND,
        "model": _model_name_for_telemetry(),
        "telemetry_enabled": STREAMING_TELEMETRY_ENABLED,
        "flags": {
            "vad_no_speech_warning_seen": False,
            "dedup_trim_applied": False,
        },
        "config": {
            "vad_threshold": get_float_from_environment("VAD_THRESHOLD", 0.5),
            "silence_timeout_seconds": get_float_from_environment(
                "VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT",
                0.8,
            ),
            "energy_threshold": get_float_from_environment("VAD_ENERGY_THRESHOLD", 0.05),
            "energy_ratio": get_float_from_environment("VAD_ENERGY_RATIO", 2.5),
            "overlap_seconds": get_float_from_environment("OVERLAP_SECONDS", 1.0),
            "minimum_chunk_seconds": get_float_from_environment("MIN_CHUNK_SECONDS", 8.0),
        },
    }


def _telemetry_recorder_for_session(
    session_id: str,
) -> StreamingSessionTelemetryRecorder | None:
    """
    Retrieves or initializes a telemetry recorder for a given session ID.
    If telemetry is globally disabled, it returns None immediately. Otherwise,
    it looks for an existing recorder within the session store or creates a new one,
    ensuring that a session state object exists to hold the recorder. This acts
    as a centralized factory for managing logging objects during streaming.
    """
    if not STREAMING_TELEMETRY_ENABLED:
        return None

    with session_store_lock:
        session = session_store.get(session_id)
        if session and session.telemetry_recorder is not None:
            return session.telemetry_recorder

        recorder = StreamingSessionTelemetryRecorder(
            session_id=session_id,
            output_dir=STREAMING_TELEMETRY_DIR,
            summary_seed=_telemetry_seed(),
        )
        if session is None:
            session = SessionState(
                engine=backend_info.get("engine"),
                telemetry_recorder=recorder,
            )
            session_store[session_id] = session
        else:
            session.telemetry_recorder = recorder
        return recorder


def _update_chunk_telemetry_summary(
    session_id: str, recording_index: int, chunk_index: int, fields: dict
) -> None:
    """Updates the summary data for a specific chunk inside a given recording."""
    recorder = _telemetry_recorder_for_session(session_id)
    if recorder is None:
        return
    recorder.update_chunk_summary(recording_index, chunk_index, fields)


def _update_session_telemetry_summary(session_id: str, fields: dict) -> None:
    """
    Merges top-level session information into the overall telemetry report.
    This is used to record final session outcomes such as the total processing time,
    the final combined transcript, and whether the final paste was successful.
    It provides a high-level overview of the entire session's performance and
    success rate without needing to dig into every individual audio chunk.
    """
    recorder = _telemetry_recorder_for_session(session_id)
    if recorder is None:
        return
    recorder.update_session_summary(fields)


def _handle_session_telemetry_event(session_id: str, payload: dict) -> None:
    """
    Routes incoming telemetry commands from the Ear to the correct recorder logic.
    Updates the session-wide 'summary' flags, such as flagging that a VAD warning
    was seen, so the final report highlights potential microphone sensitivity issues.
    """
    event_type = str(payload.get("type", "session_event"))
    chunk_index = payload.get("chunk_index")
    recording_index = payload.get("recording_index", 0)

    if event_type == "vad_no_speech_warning":
        _update_session_telemetry_summary(
            session_id,
            {
                "flags": {
                    "vad_no_speech_warning_seen": True,
                    "dedup_trim_applied": False,
                }
            },
        )
        return

    fields = {
        key: value
        for key, value in payload.items()
        if key not in {"type", "chunk_index", "recording_index"}
    }

    if event_type == "chunk_sent_to_brain":
        _update_chunk_telemetry_summary(
            session_id, recording_index, int(chunk_index), fields
        )
        return

    if event_type == "silence_threshold_hit":
        _update_chunk_telemetry_summary(
            session_id, recording_index, int(chunk_index), fields
        )
        return