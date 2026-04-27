# src/streaming/session.py
import time
from typing import Tuple, Dict

# Import the core logic functions that do the actual work without holding state
from src.streaming.streaming_shared_logic import (
    apply_last_chunk_overlap,
    analyze_duplicate_chunk_prefix,
    ChunkDeduplicationResult
)

class StreamingSession:
    """
    This class remembers the state of the current audio recording and text transcription.
    It keeps track of the previous audio bytes to add to the next chunk, and it keeps track
    of the previous text to remove duplicate words from the next text chunk.
    """
    def __init__(self, overlap_seconds: float = 1.0, sample_rate: int = 16000):
        # We store the settings for how much audio to overlap and the audio quality (sample rate)
        self._overlap_seconds = overlap_seconds
        self._sample_rate = sample_rate
        
        # Calculate how many bytes we need to save for the overlap. 
        # Number of samples = sample_rate * overlap_seconds. 
        # Each sample is 2 bytes long, so we multiply by 2.
        self._overlap_byte_count = int(self._sample_rate * 2 * self._overlap_seconds)
        
        # Keep track of the audio bytes we need to save for the next chunk
        self._last_chunk_tail_bytes = b""
        
        # Keep track of when the current chunk started processing
        self._chunk_started_at = time.time()
        
        # Keep track of the text parts we have processed, using a dictionary.
        # The key is the chunk number, and the value is the text string.
        self._transcript_parts: Dict[int, str] = {}

    def process_outgoing_audio_chunk(self, audio_bytes: bytes, stop_session: bool, silence_seconds: float) -> bytes:
        """
        Takes the new audio bytes, adds the previous audio bytes to the start, 
        and saves the end of this audio for the next time.

        Parameters:
        audio_bytes: The new audio data we just recorded.
        stop_session: A true or false value. True if this is the last audio piece.
        silence_seconds: How many seconds of silence was at the end of the audio.

        Returns:
        The combined audio bytes (previous end + new audio).
        """
        # Calculate the number of bytes for the silence duration
        silence_byte_count = int(silence_seconds * self._sample_rate * 2)
        
        # Call the core function to combine the old audio and the new audio
        result = apply_last_chunk_overlap(
            current_chunk_audio_bytes=audio_bytes,
            last_chunk_tail_bytes=self._last_chunk_tail_bytes,
            overlap_audio_byte_count=self._overlap_byte_count,
            silence_audio_byte_count=silence_byte_count,
            sample_rate=self._sample_rate,
            stop_session=stop_session
        )
        
        # Save the end of this new audio so we can use it next time
        self._last_chunk_tail_bytes = result.next_chunk_tail_bytes
        
        # Update the start time for the next chunk
        self._chunk_started_at = time.time()
        
        # Return the final audio that is ready to be sent
        return result.overlapped_audio_bytes

    def process_incoming_text_chunk(self, sequence_index: int, raw_text: str) -> Tuple[str, ChunkDeduplicationResult]:
        """
        Takes the new text, looks at the previous text, and removes words that are the same at the start.

        Parameters:
        sequence_index: The number order of this text chunk (0, 1, 2, etc).
        raw_text: The newly transcribed text string.

        Returns:
        A tuple containing the cleaned text string and the full analysis result object.
        """
        # Get the text from the previous chunk. If there is no previous chunk, use an empty string.
        last_text = self._transcript_parts.get(sequence_index - 1, "")
        
        # Call the core function to check for duplicate words and remove them
        analysis = analyze_duplicate_chunk_prefix(last_text, raw_text)
        
        # Save the clean text in our dictionary so the next chunk can check against it
        self._transcript_parts[sequence_index] = analysis.cleaned_text
        
        # Return the clean text and the analysis information
        return analysis.cleaned_text, analysis

    def get_full_transcript(self) -> str:
        """
        Combines all the separate clean text pieces into one long text string.

        Returns:
        The complete text string of the whole recording.
        """
        # Get all the text parts in order of their chunk number, ignoring empty strings
        parts = [part for seq, part in sorted(self._transcript_parts.items()) if part]
        
        # Join all the parts together with a space in between, and remove extra spaces at the ends
        return " ".join(parts).strip()
        
    def reset_audio_state(self):
        """
        Clears the saved audio bytes and resets the timer for a new recording.
        """
        # Set the saved audio bytes to empty
        self._last_chunk_tail_bytes = b""
        
        # Reset the start time to right now
        self._chunk_started_at = time.time()
