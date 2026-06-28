import threading
from typing import Any, Optional, Protocol

import pyaudio

from src.audio.ear_runtime.recording import LogState
from src.audio.vad_segmenter import SileroUtteranceGate

# We import the dependency classes used in method type hints
from src.streaming.capture_session import CaptureSession


class EarProtocol(Protocol):
    """Protocol defining the structural interface of an Ear controller."""

    # 1. Attributes
    pyaudio_inst: pyaudio.PyAudio
    stream: Optional[pyaudio.Stream]
    is_recording: bool
    last_rms: float
    gain_multiplier: float
    input_device_index: int
    active_mic_name: str
    current_model: str
    last_frequency_bands: dict[str, float]

    lock: threading.Lock
    total_frames: int

    @property
    def close_mic_stream(self) -> None: ...

    # 2. Methods
    def stop_no_streaming(self) -> None:
        """Finalize a non-streaming recording and close its raw audio socket."""
        ...

    def stop_and_send(
        self,
        session: CaptureSession,
        utr_gate: SileroUtteranceGate,
        log_state: LogState,
        stop_session: bool = True,
    ) -> None:
        """Unified method to stop recording and transmit the final data."""
        ...

    def record_loop(
        self,
        input_trigger: Optional[Any],
        utr_gate: SileroUtteranceGate,
        session: CaptureSession,
        log_state: LogState,
    ) -> None:
        """Run the main background loop that keeps the Ear process alive."""
        ...

    def record_loop_tick(
        self,
        input_trigger: Optional[Any],
        utr_gate: SileroUtteranceGate,
        session: CaptureSession,
        log_state: LogState,
    ) -> None:
        """Execute a single tick of the background recording controller loop."""
        ...

    def cleanup(self) -> None:
        """Perform a clean shutdown of the active Ear controller resources."""
        ...
