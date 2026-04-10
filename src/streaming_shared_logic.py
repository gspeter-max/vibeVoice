from __future__ import annotations

import re
from dataclasses import dataclass

# These are the default settings for the system.
# They control how long to wait for silence and how much audio to overlap.
DEFAULT_VAD_SCORE_THRESHOLD = 0.50
DEFAULT_SILENCE_TIMEOUT_SECONDS = 0.65
DEFAULT_VAD_ENERGY_THRESHOLD = 0.05
DEFAULT_ENERGY_RATIO = 2.5
DEFAULT_OVERLAP_SECONDS = 0.50
DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS = 12.0


@dataclass(frozen=True)
class ChunkSplitDecision:
    """
    This is a simple container to hold the answer to one question:
    'Should we cut the current audio chunk and start a new one now?'
    It also stores how long the chunk is and how long the silence has been.
    """
    should_split_now: bool
    chunk_age_seconds: float
    silence_duration_seconds: float


def should_split_chunk_after_silence(
    *,
    chunk_started_at_seconds: float,
    now_seconds: float,
    minimum_chunk_age_before_silence_split_seconds: float,
    utterance_gate_should_finalize_now: bool,
    silence_duration_seconds: float,
) -> ChunkSplitDecision:
    """
    This function decides if it is time to stop the current piece of audio 
    and start a new one.
    
    It checks two things:
    1. Is the current piece of audio longer than the minimum time allowed?
    2. Did the system detect enough silence to suggest the person stopped talking?
    
    If both are true, it returns 'True' for should_split_now.
    """
    # Calculate how many seconds have passed since this chunk started.
    chunk_age_seconds = max(0.0, now_seconds - chunk_started_at_seconds)
    
    # We only split if the chunk is old enough AND the silence detector says okay.
    should_split_now = (
        chunk_age_seconds > minimum_chunk_age_before_silence_split_seconds
        and utterance_gate_should_finalize_now
    )
    
    return ChunkSplitDecision(
        should_split_now=should_split_now,
        chunk_age_seconds=chunk_age_seconds,
        silence_duration_seconds=silence_duration_seconds,
    )


@dataclass(frozen=True)
class OverlapApplicationResult:
    """
    This container holds the audio data after we have joined the 
    previous chunk's end with the current chunk's beginning.
    """
    overlapped_audio_bytes: bytes
    next_pending_overlap_audio_bytes: bytes
    overlap_seconds_added_from_previous_chunk: float


def apply_previous_chunk_overlap(
    *,
    current_chunk_audio_bytes: bytes,
    previous_pending_overlap_audio_bytes: bytes,
    overlap_audio_byte_count: int,
    sample_rate: int,
    stop_session: bool,
) -> OverlapApplicationResult:
    """
    This function joins two pieces of audio together. 
    
    When we record in chunks, we keep a small bit from the end of the last chunk 
    and put it at the start of the next chunk. This ensures we don't lose 
    any sound in the middle of a word.
    
    It returns the new combined audio and also saves a bit of the current 
    audio to be used for the next time this function is called.
    """
    # If the session is over, we don't need to overlap anything.
    if stop_session:
        return OverlapApplicationResult(
            overlapped_audio_bytes=current_chunk_audio_bytes,
            next_pending_overlap_audio_bytes=b"",
            overlap_seconds_added_from_previous_chunk=0.0,
        )

    # Join the saved bit from last time to the start of the new audio.
    overlapped_audio_bytes = previous_pending_overlap_audio_bytes + current_chunk_audio_bytes
    
    # Save the end of the current audio to use for the NEXT chunk.
    if overlap_audio_byte_count > 0:
        next_pending_overlap_audio_bytes = current_chunk_audio_bytes[-overlap_audio_byte_count:]
    else:
        next_pending_overlap_audio_bytes = b""

    # Calculate how many seconds of audio we added from the previous chunk.
    overlap_seconds_added_from_previous_chunk = len(previous_pending_overlap_audio_bytes) / 2.0 / sample_rate
    
    return OverlapApplicationResult(
        overlapped_audio_bytes=overlapped_audio_bytes,
        next_pending_overlap_audio_bytes=next_pending_overlap_audio_bytes,
        overlap_seconds_added_from_previous_chunk=overlap_seconds_added_from_previous_chunk,
    )


def split_text_into_comparable_words(text: str) -> list[str]:
    """
    This function takes a full sentence and breaks it into a list of single words.
    Example: "Hello world" becomes ["Hello", "world"]
    """
    return [word for word in text.strip().split() if word]


def normalize_word_for_overlap_matching(original_word: str) -> str:
    """
    This function cleans a single word so it is easier to compare.
    
    1. It makes all letters lowercase (small).
    2. It removes marks like dots (.), commas (,), or marks (!) from the 
       start and the end of the word.
    
    Example: "Believed." becomes "believed"
    """
    lowered_word = original_word.lower()
    # This line uses a 'regular expression' to remove non-letters from ends.
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", lowered_word)


def build_original_words_and_overlap_matching_words(
    text: str,
) -> tuple[list[str], list[str]]:
    """
    This function takes a sentence and creates two lists of words:
    1. The 'original' words exactly as they were written (with dots and big letters).
    2. The 'matching' words that are cleaned up (small letters, no dots).
    
    We use the cleaned words to find matches, but we keep the original 
    words to show the final text to the user.
    """
    original_words = split_text_into_comparable_words(text)
    
    # Create the cleaned list by running every word through the cleaning function.
    overlap_matching_words = [
        normalized_word
        for normalized_word in (
            normalize_word_for_overlap_matching(original_word)
            for original_word in original_words
        )
        if normalized_word
    ]
    return original_words, overlap_matching_words


def should_skip_overlap_trim_because_result_is_too_small(
    current_original_words: list[str],
    trimmed_current_original_words: list[str],
    overlap_word_count: int,
) -> bool:
    """
    This is a safety check. 
    
    If we delete too many words from the new text, we might end up with 
    nothing left or just one tiny word. 
    
    If the system thinks the match is very long (3 or more words) but 
    deleting them would leave the new text almost empty, we skip the 
    deletion to avoid losing information.
    """
    return (
        len(trimmed_current_original_words) <= 1
        and overlap_word_count >= 3
        and len(current_original_words) == overlap_word_count + len(trimmed_current_original_words)
    )


def remove_duplicate_chunk_prefix(
    previous_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 8,
) -> str:
    """
    This is the main function that prevents words from appearing twice 
    in your transcript. 
    
    It compares the END of the previous text with the START of the new text.
    If it finds a match, it removes the repeated words from the new text.
    
    Step-by-step:
    1. Break both texts into cleaned words.
    2. Look for the largest possible match (up to 8 words).
    3. If the last 8 words of the old text match the first 8 of the new text, 
       delete those 8 from the new text.
    4. If not, try 7 words, then 6, then 5, then 4, then 3, then 2.
    5. If a match is found, return the new text without those words.
    """
    # If one of the texts is empty, there is nothing to compare.
    if not previous_chunk_text or not current_chunk_text:
        return current_chunk_text.strip()

    # Get the cleaned lists of words for both pieces of text.
    previous_original_words, previous_overlap_matching_words = (
        build_original_words_and_overlap_matching_words(previous_chunk_text)
    )
    current_original_words, current_overlap_matching_words = (
        build_original_words_and_overlap_matching_words(current_chunk_text)
    )
    
    # Decide the maximum number of words we should try to match.
    largest_possible_overlap = min(
        len(previous_overlap_matching_words),
        len(current_overlap_matching_words),
        len(previous_original_words),
        len(current_original_words),
        max_overlap_words,
    )

    # Start searching for matches. We start from the biggest number 
    # and count down to 0 so we can catch even a single repeated word.
    for overlap_word_count in range(largest_possible_overlap, 0, -1):
        # Compare the end of the old list to the start of the new list.
        if previous_overlap_matching_words[-overlap_word_count:] == current_overlap_matching_words[:overlap_word_count]:
            # If they ARE the same, we prepare to remove them.
            trimmed_current_original_words = current_original_words[overlap_word_count:]
            
            # Run the safety check to make sure we aren't deleting too much.
            if should_skip_overlap_trim_because_result_is_too_small(
                current_original_words,
                trimmed_current_original_words,
                overlap_word_count,
            ):
                return current_chunk_text.strip()

            # Join the remaining words back into a sentence and return it.
            return " ".join(trimmed_current_original_words).strip()

    # If no match was found after checking all numbers, return the text as is.
    return current_chunk_text.strip()


def normalize_text_for_word_error_rate(text: str) -> str:
    """
    This function is used for testing. It cleans a whole sentence by:
    1. Making everything lowercase.
    2. Removing all special marks (punctuation).
    3. Making sure there is only one space between words.
    
    This makes it easy to compare two sentences to see if the words 
    are the same, even if the spelling or dots are different.
    """
    lowered_text = text.lower()
    # Replace anything that isn't a letter or number with a space.
    punctuation_removed_text = re.sub(r"[^a-z0-9\s']", " ", lowered_text)
    # Turn multiple spaces into just one space.
    collapsed_spacing_text = re.sub(r"\s+", " ", punctuation_removed_text).strip()
    return collapsed_spacing_text
