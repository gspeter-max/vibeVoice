import threading
from dataclasses import dataclass, field
from src.streaming.streaming_session_telemetry import StreamingSessionTelemetryRecorder

# Global state
backend_info = {"engine": None}
backend_lock = threading.Lock()
session_store = {}
session_store_lock = threading.Lock()


@dataclass
class RecordingState:
    """
    Holds the transcription state for a single button press.
    One SessionState contains many RecordingState objects — one per button press.
    """

    received_count: int = 0
    done_count: int = 0
    closed: bool = False
    finalized: bool = False
    transcript_parts: dict = field(default_factory=dict)
    stt_time: float = 0.0


@dataclass
class SessionState:
    """
    State object for an entire application run (one Brain process lifetime).
    A session survives multiple button presses; each press is one RecordingState
    stored in the recordings dict keyed by its recording_index integer.
    """

    engine: object
    # recordings[0] = first button press, recordings[1] = second, …
    recordings: dict = field(default_factory=dict)
    stt_time: float = 0.0
    telemetry_recorder: StreamingSessionTelemetryRecorder | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create_recording(self, rec_idx: int) -> "RecordingState":
        """Returns the RecordingState for rec_idx, creating it if needed."""
        if rec_idx not in self.recordings:
            self.recordings[rec_idx] = RecordingState()
        return self.recordings[rec_idx]
