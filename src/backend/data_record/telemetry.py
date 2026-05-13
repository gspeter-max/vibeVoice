"""Write one JSON file per streaming session so telemetry stays readable and local."""

from __future__ import annotations

import json
import os
import tempfile
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.env_utils import get_float_from_environment
from src.backend.state import (
    backend_info, session_store, session_store_lock, SessionState
)

from src.utils.settings import settings


@dataclass
class StreamingSessionTelemetryRecorder:
    """
    Manages the lifecycle of a session's telemetry data and file output.
    This class acts as an in-memory buffer that collects summary
    statistics during a recording session. Every time data is updated, it
    atomically writes a JSON snapshot to the disk.
    """

    session_id: str
    output_dir: Path
    summary_seed: dict[str, Any]
    payload: dict[str, Any] = field(init=False)
    _filename: str = field(init=False)
    _audio_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """
        Initializes the internal payload structure for a new session.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._filename = f"{ts}_{self.session_id}.json"
        self._audio_dir = self.output_dir / f"{ts}_{self.session_id}_audio"
        self._audio_dir.mkdir(parents=True, exist_ok=True)

        self.payload = {
            "session_summary": {
                "session_id": self.session_id,
                "total_chunks_received": 0,
                "total_decode_seconds": 0.0,
                "final_text": "",
                "final_paste_success": None,
                "flags": {},
                **self.summary_seed,
            },
            "recordings": [],
        }

    def save_chunk_audio(self, recording_index: int, chunk_index: int, pcm_bytes: bytes, sample_rate: int = 16000) -> str | None:
        """
        Saves raw PCM16 bytes to a standard WAV file in the session's audio directory.
        Returns the relative path to the saved audio file.
        """
        if not pcm_bytes:
            return None
            
        filename = f"rec{recording_index}_chunk{chunk_index}.wav"
        file_path = self._audio_dir / filename
        
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
            
        return f"{self._audio_dir.name}/{filename}"

    def _session_file_path(self) -> Path:
        """Generates the absolute file path for the session's JSON report."""
        return self.output_dir / self._filename

    def _ensure_recording(self, rec_idx: int) -> dict[str, Any]:
        """
        Retrieves the telemetry data slot for a specific button press.
        """
        recordings = self.payload["recordings"]

        while len(recordings) <= rec_idx:
            recordings.append(
                {
                    "summary": {},
                    "chunks": [],
                }
            )

        return recordings[rec_idx]

    def _ensure_chunk(self, recording_index: int, chunk_index: int) -> dict[str, Any]:
        """
        Retrieves a specific audio chunk from inside a specific button press.
        """
        recording = self._ensure_recording(recording_index)
        chunks = recording["chunks"]

        while len(chunks) <= chunk_index:
            chunks.append(
                {
                    "summary": {},
                }
            )

        return chunks[chunk_index]

    def update_chunk_summary(
        self, recording_index: int, chunk_index: int, fields: dict[str, Any]
    ) -> None:
        """Updates summary statistics for a specific chunk."""
        chunk = self._ensure_chunk(recording_index, chunk_index)
        # Convert all fields to simple primitives to avoid JSON serialization errors (e.g. MagicMock in tests)
        safe_fields = {
            k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
            for k, v in fields.items()
        }
        chunk["summary"].update(safe_fields)
        self.write_snapshot()

    def update_session_summary(self, fields: dict[str, Any]) -> None:
        """Updates the high-level summary for the entire recording session."""
        flags = fields.get("flags")
        if isinstance(flags, dict):
            current_flags = dict(self.payload["session_summary"].get("flags", {}))
            for key, value in flags.items():
                if isinstance(value, bool) and isinstance(current_flags.get(key), bool):
                    current_flags[key] = current_flags.get(key, False) or value
                else:
                    current_flags[key] = value
            fields = dict(fields)
            fields["flags"] = current_flags

        # Convert all fields to simple primitives to avoid JSON serialization errors
        safe_fields = {
            k: v if isinstance(v, (str, int, float, bool, type(None), dict)) else str(v)
            for k, v in fields.items()
        }
        self.payload["session_summary"].update(safe_fields)
        self.write_snapshot()

    def write_snapshot(self) -> None:
        """Performs an atomic write of the current telemetry payload to a file."""
        destination = self._session_file_path()
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.output_dir, delete=False
        ) as tmp:
            json.dump(self.payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, destination)


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
    return str(getattr(engine, "model_name", "unknown_model"))


def _telemetry_seed() -> dict:
    """
    Constructs the initial configuration data for a new telemetry session.
    This dictionary includes the recording mode, current backend type, and model name,
    as well as all the VAD (Voice Activity Detection) parameters used in the session.
    By seeding this data at the start, we create a comprehensive 'header' in the
    telemetry file that documents the exact settings and hardware state for debugging.
    """
    return {
        "recording_mode": settings.recording_mode,
        "backend": settings.backend,
        "model": _model_name_for_telemetry(),
        "telemetry_enabled": settings.streaming_telemetry_enabled,
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
    if not settings.streaming_telemetry_enabled:
        return None

    with session_store_lock:
        session = session_store.get(session_id)
        if session and session.telemetry_recorder is not None:
            return session.telemetry_recorder

        recorder = StreamingSessionTelemetryRecorder(
            session_id=session_id,
            output_dir=settings.streaming_telemetry_dir,
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
