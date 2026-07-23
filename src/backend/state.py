import logging
import threading
from dataclasses import dataclass, field
from threading import Lock

from src.interfaces import TelemetryRecording, TranscriptionEngine

log = logging.getLogger(__name__)


@dataclass
class BackendState:
    """Holds the active transcription engine behind a lock."""

    _lock: Lock = field(default_factory=threading.Lock)
    engine: TranscriptionEngine | None = None
    model_name: str | None = None

    def get_engine(self) -> TranscriptionEngine | None:
        with self._lock:
            if self.engine is not None:
                return self.engine

            if self.engine is None:
                if self.model_name is not None:
                    self.load_tts(self.model_name)
                    return self.engine
                else:
                    log.error("Failed to load the transcription engine. model_name is None")
                    return None

    def set_engine(self, engine: TranscriptionEngine | None) -> None:
        with self._lock:
            self.engine = engine

    def load_tts(self, model_name):
        if model_name is None:
            return None
        if "nemotron" in model_name.lower():
            from src.models.nemotron_engine import NemotronEngine

            engine = NemotronEngine()
            self.engine = engine

        else:
            from src.models.parakeet_engine import ParakeetEngine

            engine = ParakeetEngine(model_name)
            self.engine = engine


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

    recordings: dict = field(default_factory=dict)
    stt_time: float = 0.0
    telemetry_recorder: TelemetryRecording | None = None

    def get_or_create_recording(self, rec_idx: int) -> "RecordingState":
        """Returns the RecordingState for rec_idx, creating it if needed."""
        if rec_idx not in self.recordings:
            self.recordings[rec_idx] = RecordingState()
        return self.recordings[rec_idx]


@dataclass
class SessionStates:
    """Top-level container: one engine shared across all sessions."""

    backend: BackendState = field(default_factory=BackendState)
    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, SessionState] = field(default_factory=dict)
