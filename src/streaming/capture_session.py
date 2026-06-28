"""Ear-side capture session state.

This module owns the mutable state that describes one long-lived Ear session
across many individual button-press recordings.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from src.streaming.session import apply_overlap


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
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    rec_idx: int = 0
    chunk_seq: int = 0
    chunk_start: float = 0.0
    tail: bytes = b""
    overlap_override: int | None = None

    @property
    def overlap_bytes(self) -> int:
        """Return the number of overlap bytes to keep from one chunk to the next."""
        if self.overlap_override is not None:
            return self.overlap_override
        return int(self.sample_rate * 2 * self.overlap_seconds)

    @property
    def chunk_age(self) -> float:
        """Return the age of the active chunk using the session-owned start time."""
        return max(0.0, time.time() - self.chunk_start)

    def begin(self, now: float | None = None) -> None:
        """Start one new recording while keeping the same process-level session id."""
        if now is None:
            now = time.time()
        self.chunk_seq = 0
        self.chunk_start = now

    def mark_sent(self) -> int:
        """Return the current chunk sequence number and advance to the next one."""
        seq = self.chunk_seq
        self.chunk_seq += 1
        return seq

    def commit(self) -> None:
        """Advance to the next recording slot after the current one is finalized."""
        self.rec_idx += 1
        self.chunk_seq = 0

    def stop(self) -> None:
        """Reset final-stop state that belongs to the capture session itself."""
        self.chunk_start = 0.0
        self.clear_tail()

    def mark_next(self, now: float | None = None) -> None:
        """Store the start time for the next chunk after a non-final send."""
        if now is None:
            now = time.time()
        self.chunk_start = now

    def clear_tail(self) -> None:
        """Forget any stored overlap bytes, usually after the final stop."""
        self.tail = b""

    def prep_chunk(
        self,
        audio: bytes,
        *,
        stop: bool,
        silence_seconds: float,
    ) -> bytes:
        """Apply overlap bytes and store the next overlap tail.

        The overlap rule is the same as before: prepend the previous chunk tail
        for non-final chunks and clear overlap state on the final stop.
        """
        overlap_result = apply_overlap(
            audio=audio,
            tail=self.tail,
            overlap_bytes=self.overlap_bytes,
            silence_bytes=int(silence_seconds * self.sample_rate * 2),
            rate=self.sample_rate,
            stop=stop,
        )
        self.tail = overlap_result.tail
        return overlap_result.audio
