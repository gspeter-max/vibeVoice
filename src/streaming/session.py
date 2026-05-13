# src/streaming/session.py
import time

from src.streaming.streaming_shared_logic import apply_last_chunk_overlap

class StreamingSession:
    """
    Remembers the audio state of the current recording session.
    It keeps track of the previous audio bytes so they can be prepended
    to the next audio chunk, creating a small overlap that improves
    transcription accuracy at chunk boundaries.
    """
    def __init__(self, overlap_seconds: float = 1.0, sample_rate: int = 16000):
        self._overlap_seconds = overlap_seconds
        self._sample_rate = sample_rate
        self._overlap_byte_count = int(self._sample_rate * 2 * self._overlap_seconds)
        self._last_chunk_tail_bytes = b""
        self._chunk_started_at = time.time()

    def process_outgoing_audio_chunk(self, audio_bytes: bytes, stop_session: bool, silence_seconds: float) -> bytes:
        """
        Takes the new audio bytes, adds the previous audio bytes to the start, 
        and saves the end of this audio for the next time.
        """
        silence_byte_count = int(silence_seconds * self._sample_rate * 2)
        result = apply_last_chunk_overlap(
            current_chunk_audio_bytes=audio_bytes,
            last_chunk_tail_bytes=self._last_chunk_tail_bytes,
            overlap_audio_byte_count=self._overlap_byte_count,
            silence_audio_byte_count=silence_byte_count,
            sample_rate=self._sample_rate,
            stop_session=stop_session
        )
        self._last_chunk_tail_bytes = result.next_chunk_tail_bytes
        self._chunk_started_at = time.time()
        
        return result.overlapped_audio_bytes

    def reset_audio_state(self):
        """
        Clears the saved audio bytes and resets the timer for a new recording.
        """
        self._last_chunk_tail_bytes = b""
        self._chunk_started_at = time.time()