from typing import Any, Protocol

class TelemetryRecording(Protocol):
    """Protocol defining the structural interface for telemetry recording handlers."""

    def save_chunk_audio(
        self, recording_index: int, chunk_index: int, pcm_bytes: bytes, sample_rate: int = 16000
    ) -> str | None: ...

    def update_chunk_summary(
        self, recording_index: int, chunk_index: int, fields: dict[str, Any]
    ) -> None: ...

    def update_session_summary(self, fields: dict[str, Any]) -> None: ...
    def write_snapshot(self) -> None: ...
