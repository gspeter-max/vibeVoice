from __future__ import annotations

from difflib import SequenceMatcher
import re
from dataclasses import dataclass
from src import log

# These are the default settings for the system.
# They control how long to wait for silence and how much audio to overlap.
DEFAULT_VAD_SCORE_THRESHOLD = 0.50
DEFAULT_SILENCE_TIMEOUT_SECONDS = 0.63 
DEFAULT_VAD_ENERGY_THRESHOLD = 0.05
DEFAULT_ENERGY_RATIO = 2.5
DEFAULT_OVERLAP_SECONDS = 1
DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS = 8.0
SEMANTIC_OVERLAPPING_THRESHOLD = 0.82

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


def character_similarity(words_a: list[str], words_b: list[str]) -> float:
    """
    Compare two word lists as joined strings.
    Returns 0.0 (completely different) to 1.0 (identical).
    
    This handles:
      "overlapping" vs "overlaping"  → ~0.95
      "thinkin"     vs "thinking"    → ~0.93
      "bout"        vs "about"       → ~0.80
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
    Count what fraction of words appear in both lists.
    
    This handles:
      ["all", "these", "things"] vs ["these", "all", "things"]
      → 3/3 = 1.0  (perfect overlap despite order change)
    """
    if not words_a or not words_b:
        return 0.0 
    
    set_a = set(words_a)
    set_b = set(words_b)

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    score = len(intersection) / len(union)
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
    Blend character similarity and token overlap.
    
    Why both?
    
    Char-level alone:
      "the big plan" vs "big plan the"  → 0.72 (misses reorder)
    
    Token-level alone:  
      "overlapping" vs "overlaping"     → 0.50 (misses typo)
    
    Combined:
      Both cases → caught correctly
    """
    char_score = character_similarity(
        words_a=words_a,
        words_b=words_b,
    )

    token_score = token_overlap_score(
        words_a=words_a,
        words_b=words_b,
    )

    combined = (char_score * char_weight) + (token_score * token_weight)
    log.debug(
        "[Dedup] combined_score",
        char_score=round(char_score, 4),
        token_score=round(token_score, 4),
        char_weight=char_weight,
        token_weight=token_weight,
        combined=round(combined, 4),
    )
    return combined
    
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
        log.debug("[Dedup] Skipping — one or both texts are empty")
        return current_chunk_text.strip()

    # Get the cleaned lists of words for both pieces of text.
    # normalized = punctuation removed and lowercased
    prev_original, prev_normalized = (
        build_original_words_and_overlap_matching_words(previous_chunk_text)
    )
    curr_original, curr_normalized = (
        build_original_words_and_overlap_matching_words(current_chunk_text)
    )

    # Decide the maximum number of words we should try to match.
    largest_possible_overlap = min(
        len(prev_normalized),
        len(curr_normalized),
        len(prev_original),
        len(curr_original),
        max_overlap_words,
    )

    log.debug(
        "[Dedup] ─── Starting deduplication ───",
        prev_tail_words=prev_normalized[-largest_possible_overlap:],
        curr_head_words=curr_normalized[:largest_possible_overlap],
        prev_total_words=len(prev_normalized),
        curr_total_words=len(curr_normalized),
        max_overlap_words=max_overlap_words,
        largest_possible_overlap=largest_possible_overlap,
        threshold=SEMANTIC_OVERLAPPING_THRESHOLD,
    )

    # Start searching for matches. We start from the biggest number
    # and count down so we find the largest matching overlap.
    for overlap_word_count in range(largest_possible_overlap, 1, -1):
        prev_tail = prev_normalized[-overlap_word_count:]
        curr_head = curr_normalized[:overlap_word_count]

        log.debug(
            f"[Dedup] Trying overlap_count={overlap_word_count}",
            prev_tail=prev_tail,
            curr_head=curr_head,
        )

        score = combined_overlap_score(prev_tail, curr_head)

        if score >= SEMANTIC_OVERLAPPING_THRESHOLD:
            trimmed = curr_original[overlap_word_count:]

            log.debug(
                "[Dedup] ✅ Score passed threshold — checking safety",
                overlap_word_count=overlap_word_count,
                score=round(score, 4),
                threshold=SEMANTIC_OVERLAPPING_THRESHOLD,
                words_to_remove=curr_original[:overlap_word_count],
                words_remaining=trimmed,
            )

            if should_skip_overlap_trim_because_result_is_too_small(
                curr_original,
                trimmed,
                overlap_word_count,
            ):
                log.debug(
                    "[Dedup] ⚠️  Safety check blocked trim — result would be too small",
                    overlap_word_count=overlap_word_count,
                    trimmed_word_count=len(trimmed),
                )
                return current_chunk_text.strip()

            result = " ".join(trimmed).strip()
            log.debug(
                "[Dedup] ✂️  Duplicate removed",
                removed_words=curr_original[:overlap_word_count],
                removed_count=overlap_word_count,
                score=round(score, 4),
                result_preview=result[:120],
            )
            return result

        else:
            log.debug(
                f"[Dedup] ❌ Score below threshold at overlap_count={overlap_word_count}",
                score=round(score, 4),
                threshold=SEMANTIC_OVERLAPPING_THRESHOLD,
                gap=round(SEMANTIC_OVERLAPPING_THRESHOLD - score, 4),
            )

    # No match found at any overlap count.
    log.debug(
        "[Dedup] 🔴 No match found — returning current text unchanged",
        prev_tail_words=prev_normalized[-largest_possible_overlap:],
        curr_head_words=curr_normalized[:largest_possible_overlap],
        tried_counts=list(range(largest_possible_overlap, 1, -1)),
    )
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
