"""Write one JSON file per streaming session so telemetry stays readable and local."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


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

    def __post_init__(self) -> None:
        """
        Initializes the internal payload structure for a new session.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._filename = f"{ts}_{self.session_id}.json"

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
        chunk["summary"].update(fields)
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

        self.payload["session_summary"].update(fields)
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
