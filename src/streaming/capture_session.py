"""Ear-side capture session state.

This module owns the mutable state that describes one long-lived Ear session
across many individual button-press recordings.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from src.streaming.streaming_shared_logic import apply_last_chunk_overlap


@dataclass
class CaptureSession:
    """Store the Ear-side recording counters and overlap buffers.

    One `CaptureSession` instance lives for the whole Ear process lifetime.
    The session id stays stable, while the recording index advances only after
    a recording is committed. Chunk sequence numbers advance per chunk and reset
    after the final stop/commit of one recording.
    """

    sample_rate: int
    overlap_seconds: float
    current_session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    current_recording_index: int = 0
    current_chunk_sequence_number: int = 0
    chunk_started_at_seconds: float = 0.0
    last_chunk_tail_bytes: bytes = b""
    overlap_audio_byte_count_override: int | None = None

    @property
    def overlap_audio_byte_count(self) -> int:
        """Return the number of overlap bytes to keep from one chunk to the next."""

        if self.overlap_audio_byte_count_override is not None:
            return self.overlap_audio_byte_count_override
        return int(self.sample_rate * 2 * self.overlap_seconds)

    def begin_recording(self, now_seconds: float | None = None) -> None:
        """Start one new recording while keeping the same process-level session id."""

        if now_seconds is None:
            now_seconds = time.time()
        self.current_chunk_sequence_number = 0
        self.chunk_started_at_seconds = now_seconds

    def mark_chunk_sent(self) -> int:
        """Return the current chunk sequence number and advance to the next one."""

        sequence_number = self.current_chunk_sequence_number
        self.current_chunk_sequence_number += 1
        return sequence_number

    def mark_recording_committed(self) -> None:
        """Advance to the next recording slot after the current one is finalized."""

        self.current_recording_index += 1
        self.current_chunk_sequence_number = 0

    def clear_overlap_tail(self) -> None:
        """Forget any stored overlap bytes, usually after the final stop."""

        self.last_chunk_tail_bytes = b""

    def prepare_chunk_for_send(
        self,
        audio_chunk_for_brain: bytes,
        *,
        stop_session: bool,
        silence_seconds: float,
    ) -> bytes:
        """Apply overlap bytes and store the next overlap tail.

        The overlap rule is the same as before: prepend the previous chunk tail
        for non-final chunks and clear overlap state on the final stop.
        """

        overlap_result = apply_last_chunk_overlap(
            current_chunk_audio_bytes=audio_chunk_for_brain,
            last_chunk_tail_bytes=self.last_chunk_tail_bytes,
            overlap_audio_byte_count=self.overlap_audio_byte_count,
            silence_audio_byte_count=int(silence_seconds * self.sample_rate * 2),
            sample_rate=self.sample_rate,
            stop_session=stop_session,
        )
        self.last_chunk_tail_bytes = overlap_result.next_chunk_tail_bytes
        return overlap_result.overlapped_audio_bytes
