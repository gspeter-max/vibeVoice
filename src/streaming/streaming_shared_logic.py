from __future__ import annotations

from difflib import SequenceMatcher
import re
from dataclasses import dataclass
from src import log
import numpy as np

# These are the default settings for the system.
# They control how long to wait for silence and how much audio to overlap.
DEFAULT_VAD_SCORE_THRESHOLD = 0.50
DEFAULT_SILENCE_TIMEOUT_SECONDS = 0.8
DEFAULT_VAD_ENERGY_THRESHOLD = 0.05
DEFAULT_ENERGY_RATIO = 2.5
DEFAULT_OVERLAP_SECONDS = 3.5
DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS = 8.0
SEMANTIC_OVERLAPPING_THRESHOLD = 0.70

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
    next_chunk_tail_bytes: bytes
    overlap_seconds_from_last_chunk: float


@dataclass(frozen=True)
class ChunkDeduplicationResult:
    """Hold the cleaned chunk text and the scores used to decide the trim.

    Brain uses this to write telemetry without recalculating the overlap.
    The inputs are two chunk texts, and the output is the cleaned text plus scores.
    """

    cleaned_text: str
    overlap_word_count: int
    char_score: float
    token_score: float
    combined_score: float
    trim_applied: bool
    skipped_because_result_too_small: bool


def _equalize_energy(overlap_bytes: bytes, current_bytes: bytes) -> bytes:
    """Boost overlap audio to match the RMS energy of the current chunk."""
    if not overlap_bytes or not current_bytes:
        return overlap_bytes

    # Safety: PCM-16 audio must have an even number of bytes.
    # We trim the buffers to the nearest word (2 bytes) to prevent frombuffer errors.
    o_trimmed = overlap_bytes[: len(overlap_bytes) // 2 * 2]
    c_trimmed = current_bytes[: len(current_bytes) // 2 * 2]

    overlap = np.frombuffer(o_trimmed, dtype=np.int16).astype(np.float32)
    current = np.frombuffer(c_trimmed, dtype=np.int16).astype(np.float32)

    overlap_rms = float(np.sqrt(np.mean(overlap**2)))
    current_rms = float(np.sqrt(np.mean(current**2)))

    # Avoid division by zero or near-silence processing
    if overlap_rms < 1.0 or current_rms < 1.0:
        return overlap_bytes

    # Cap gain to avoid distortion
    gain = min(current_rms / overlap_rms, 3.0)
    return (overlap * gain).clip(-32768, 32767).astype(np.int16).tobytes()


def apply_last_chunk_overlap(
    *,
    current_chunk_audio_bytes: bytes,
    last_chunk_tail_bytes: bytes,
    overlap_audio_byte_count: int,
    silence_audio_byte_count: int = 0,
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
            next_chunk_tail_bytes=b"",
            overlap_seconds_from_last_chunk=0.0,
        )

    # Equalize the energy of the overlap to match the new audio before joining
    equalized_overlap = _equalize_energy(
        last_chunk_tail_bytes, current_chunk_audio_bytes
    )
 
    # Join the equalized bit from last time to the start of the new audio.
    overlapped_audio_bytes = equalized_overlap + current_chunk_audio_bytes
    
    # Save the end of the current audio to use for the NEXT chunk.
    # We skip 'silence_audio_byte_count' at the end to anchor overlap to actual speech.
    if overlap_audio_byte_count > 0:
        # STRATEGY: Find where the actual speech ended.
        # Since the chunk was split after a silence timeout, the end of the 
        # buffer is pure silence. We subtract 'silence_audio_byte_count' 
        # to find the "speech tail," ensuring the overlap contains high-signal 
        # audio for the transcription engine's deduplication algorithm.
        
        speech_end = len(current_chunk_audio_bytes) - silence_audio_byte_count
        speech_start = max(0, speech_end - overlap_audio_byte_count)
        
        next_chunk_tail_bytes = current_chunk_audio_bytes[speech_start:speech_end]
    else:
        next_chunk_tail_bytes = b""

    # Calculate how many seconds of audio we added from the previous chunk.
    overlap_seconds_from_last_chunk = len(last_chunk_tail_bytes) / 2.0 / sample_rate
    
    return OverlapApplicationResult(
        overlapped_audio_bytes=overlapped_audio_bytes,
        next_chunk_tail_bytes=next_chunk_tail_bytes,
        overlap_seconds_from_last_chunk=overlap_seconds_from_last_chunk,
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
        norm for w in original_words if (norm := normalize_word_for_overlap_matching(w))
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


def character_similarity(words_a: list[str], words_b: list[str]) -> float:
    """
    Measures how similar two lists of words are based on their characters.
    It joins the words into strings and compares them using a sequence
    matching algorithm, returning a score from 0.0 to 1.0. This helps
    the system catch cases where the AI might have slightly different
    spellings for the same words in overlapping audio segments,
    ensuring that small typos don't prevent successful deduplication.
    """
    if not words_a or not words_b:
        return 0.0
    
    str_a = " ".join(words_a)
    str_b = " ".join(words_b)
    
    score = SequenceMatcher(None, str_a, str_b).ratio()
    log.debug(
        f"[Dedup] char_similarity",
        str_a=str_a,
        str_b=str_b,
        score=round(score, 4),
    )
    return score

def token_overlap_score(words_a: list[str], words_b: list[str]) -> float:
    """
    Calculates the percentage of words that are identical in both lists.
    By converting the word lists into 'sets', this function ignores the
    specific order of the words and focuses on whether the same unique
    vocabulary appears in both segments. This is particularly useful for
    catching overlaps where the AI might have reordered words slightly
    but is still clearly describing the same spoken phrase.
    """
    if not words_a or not words_b:
        return 0.0 
    
    set_a = set(words_a)
    set_b = set(words_b)

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    score = len(intersection) / len(union) if union else 0.0
    log.debug(
        f"[Dedup] token_overlap",
        words_a=words_a,
        words_b=words_b,
        shared_words=sorted(intersection),
        only_in_a=sorted(set_a - set_b),
        only_in_b=sorted(set_b - set_a),
        score=round(score, 4),
    )
    return score

def combined_overlap_score(
    words_a: list[str],
    words_b: list[str],
    char_weight: float = 0.6,
    token_weight: float = 0.4,
) -> float:
    """
    Combines character and token scores into a single confidence metric.
    By blending both character similarity and word-set overlap, we create
    a robust detection system that isn't easily fooled by typos or word
    reordering. This weighted average provides the final 'truth' score that
    the Brain uses to decide if it should delete a chunk of text as a
    duplicate, ensuring the final transcript remains clean and accurate.
    """
    char_score = character_similarity(words_a, words_b)
    token_score = token_overlap_score(words_a, words_b)

    combined = (char_score * char_weight) + (token_score * token_weight)
    log.debug(
        "[Dedup] combined_score",
        char_score=round(char_score, 4),
        token_score=round(token_score, 4),
        combined=round(combined, 4),
    )
    return combined


def analyze_duplicate_chunk_prefix(
    last_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 15,
) -> ChunkDeduplicationResult:
    """
    Analyzes two pieces of text to find and report any duplicated words.
    This function compares the end of the last transcript with the start
    of the new one, looking for overlapping words caused by audio overlap.
    It returns a detailed result containing the cleaned text, the number of
    words trimmed, and the confidence scores used for the decision. This
    detailed report is essential for both the Brain's logic and telemetry.
    """
    # Initialize a result with no trim applied.
    result = ChunkDeduplicationResult(
        cleaned_text=current_chunk_text.strip(),
        overlap_word_count=0,
        char_score=0.0,
        token_score=0.0,
        combined_score=0.0,
        trim_applied=False,
        skipped_because_result_too_small=False,
    )

    if not last_chunk_text or not current_chunk_text:
        return result

    prev_original, prev_normalized = build_original_words_and_overlap_matching_words(last_chunk_text)
    curr_original, curr_normalized = build_original_words_and_overlap_matching_words(current_chunk_text)

    largest_possible_overlap = min(
        len(prev_normalized),
        len(curr_normalized),
        len(prev_original),
        len(curr_original),
        max_overlap_words,
    )

    for overlap_word_count in range(largest_possible_overlap, 1, -1):
        prev_tail = prev_normalized[-overlap_word_count:]
        curr_head = curr_normalized[:overlap_word_count]
        
        # Use individual scores for telemetry reporting.
        char_score = character_similarity(prev_tail, curr_head)
        token_score = token_overlap_score(prev_tail, curr_head)
        combined_score = (char_score * 0.6) + (token_score * 0.4)

        if combined_score >= SEMANTIC_OVERLAPPING_THRESHOLD:
            trimmed = curr_original[overlap_word_count:]
            skipped = should_skip_overlap_trim_because_result_is_too_small(
                curr_original, trimmed, overlap_word_count
            )
            
            return ChunkDeduplicationResult(
                cleaned_text=current_chunk_text.strip() if skipped else " ".join(trimmed).strip(),
                overlap_word_count=overlap_word_count,
                char_score=char_score,
                token_score=token_score,
                combined_score=combined_score,
                trim_applied=not skipped,
                skipped_because_result_too_small=skipped,
            )

    return result


def remove_duplicate_chunk_prefix(
    last_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 15,
) -> str:
    """
    This is the main function that prevents words from appearing twice 
    in your transcript. 
    
    It compares the END of the previous text with the START of the new text.
    If it finds a match, it removes the repeated words from the new text.
    
    Step-by-step:
    1. Break both texts into cleaned words.
    2. Look for the largest possible match (up to 15 words).
    3. If the last 15 words of the old text match the first 15 of the new text, 
       delete those 15 from the new text.
    4. If not, try 14 words, then 13, then 12... down to 2.
    5. If a match is found, return the new text without those words.
    """
    analysis = analyze_duplicate_chunk_prefix(
        last_chunk_text,
        current_chunk_text,
        max_overlap_words=max_overlap_words,
    )
    if analysis.trim_applied:
        log.debug(
            "[Dedup] trimmed overlap",
            removed_count=analysis.overlap_word_count,
            score=round(analysis.combined_score, 4),
            last_chunk_words=len(split_text_into_comparable_words(last_chunk_text)),
            current_words=len(split_text_into_comparable_words(current_chunk_text)),
            result_words=len(split_text_into_comparable_words(analysis.cleaned_text)),
        )
    return analysis.cleaned_text


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
