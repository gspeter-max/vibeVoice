"""Chunk overlap and deduplication helpers for streaming transcription."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import os
import pathlib
import re
import time
from typing import Final, Tuple, List
import wave

import numpy as np
import numpy.typing as npt

from src import log
from src.utils.settings import settings


@dataclass(frozen=True)
class SplitDecision:
    """Holds decision data on whether to split the current audio chunk.

    Attributes:
        should_split: True if the current chunk should be split now.
        age: The elapsed age of the current chunk in seconds.
        silence_len: The duration of the detected silence in seconds.
    """

    should_split: bool
    age: float
    silence_len: float


def should_split(
    *,
    start_time: float,
    now: float,
    min_age: float,
    gate_finalize: bool,
    silence_len: float,
) -> SplitDecision:
    """Determines if the active audio chunk should be split based on age and silence.

    Steps:
    1. Calculate the active chunk's age by subtracting the start time from the current time.
    2. Check if the chunk's age exceeds the minimum allowed age before a silence split.
    3. Trigger a split if the age threshold is met and the utterance gate requests finalization.
    4. Return the split decision containing the decision state, chunk age, and silence duration.
    """
    age: float = max(0.0, now - start_time)

    should_split_now: bool = age > min_age and gate_finalize

    return SplitDecision(
        should_split=should_split_now,
        age=age,
        silence_len=silence_len,
    )


@dataclass(frozen=True)
class OverlapResult:
    """Holds the result of applying overlap to the current audio chunk.

    Attributes:
        audio: The combined audio bytes (previous tail prepended to current chunk).
        tail: The audio bytes from the end of the current chunk saved for the next overlap.
        overlap_len: The length of the overlap prepended from the last chunk in seconds.
    """

    audio: bytes
    tail: bytes
    overlap_len: float


@dataclass(frozen=True)
class DedupResult:
    """Holds duplicate analysis results for overlapping text segments.

    Attributes:
        text: The text with any duplicate overlap trimmed.
        overlap_words: The number of overlapping words detected.
        char_score: The character similarity score between the overlap segments.
        token_score: The token-based similarity score between the overlap segments.
        combined_score: The combined similarity score of the character and token metrics.
        trimmed: True if the overlap trim was applied to the text.
        skipped: True if the trim was skipped because the remaining text would be too short.
    """

    text: str
    overlap_words: int
    char_score: float
    token_score: float
    combined_score: float
    trimmed: bool
    skipped: bool


def _equalize_energy(overlap_bytes: bytes, current_bytes: bytes) -> bytes:
    """Adjusts the energy of overlap audio to match the Root Mean Square (RMS) of current audio.

    Steps:
    1. Verify both input byte strings are non-empty; if either is empty, return overlap_bytes.
    2. Trim both byte arrays to an even length to ensure valid 16-bit PCM samples.
    3. Convert the trimmed bytes to float32 numpy arrays representing PCM audio samples.
    4. Calculate the RMS energy for both the overlap and current audio segments.
    5. If either RMS value is near zero, return the original overlap bytes.
    6. Calculate the scaling gain (current RMS / overlap RMS), capping the gain at 3.0.
    7. Scale the overlap samples, clip them to signed 16-bit integer limits, and convert back to bytes.
    """
    if not overlap_bytes or not current_bytes:
        return overlap_bytes
    o_trimmed: bytes = overlap_bytes[: len(overlap_bytes) // 2 * 2]
    c_trimmed: bytes = current_bytes[: len(current_bytes) // 2 * 2]

    overlap: npt.NDArray[np.float32] = np.frombuffer(o_trimmed, dtype=np.int16).astype(np.float32)
    current: npt.NDArray[np.float32] = np.frombuffer(c_trimmed, dtype=np.int16).astype(np.float32)

    overlap_rms: float = float(np.sqrt(np.mean(overlap**2)))
    current_rms: float = float(np.sqrt(np.mean(current**2)))

    if overlap_rms < 1.0 or current_rms < 1.0:
        return overlap_bytes

    gain: float = min(current_rms / overlap_rms, 3.0)
    boosted: npt.NDArray[np.float32] = overlap * gain
    boosted = np.tanh(boosted / 32768) * 32768
    return boosted.astype(np.int16).tobytes()


def apply_overlap(
    *,
    audio: bytes,
    tail: bytes,
    overlap_bytes: int,
    silence_bytes: int = 0,
    rate: int,
    stop: bool,
) -> OverlapResult:
    """Prepends the previous chunk's tail to the current chunk and extracts the new tail.

    Steps:
    1. If the session has stopped, return the current audio as is with no overlap or new tail.
    2. Adjust the volume/energy of the previous tail to match the current chunk's energy.
    3. Prepend the volume-matched tail to the start of the current chunk audio.
    4. Determine the new tail for the next chunk from the end of the current audio:
       - Exclude silence bytes at the end of the chunk to avoid capturing silent tails.
       - Extract up to overlap_bytes from the speech segment.
    5. Calculate the duration of the prepended overlap in seconds using sample rate.
    6. Return the combined audio, the new tail bytes, and the overlap duration.
    """
    if stop:
        return OverlapResult(
            audio=audio,
            tail=b"",
            overlap_len=0.0,
        )

    equalized_overlap: bytes = _equalize_energy(tail, audio)
    overlapped_audio_bytes: bytes = equalized_overlap + audio
    log.info(
        "-----------------------------------------------------------testing---------------------------------------------------"
    )
    next_chunk_tail_bytes: bytes
    if overlap_bytes > 0:
        speech_end: int = len(audio) - silence_bytes
        speech_start: int = max(0, speech_end - overlap_bytes)
        next_chunk_tail_bytes = audio[speech_start:speech_end]
    else:
        next_chunk_tail_bytes = b""

    overlap_len: float = len(tail) / 2.0 / rate
    testfile: pathlib.Path = pathlib.Path(os.getcwd()) / "equalized_overlap.wav"
    with wave.open(str(testfile), "wb") as waveFile:
        waveFile.setnchannels(1)
        waveFile.setsampwidth(2)
        waveFile.setframerate(settings.rate)
        waveFile.writeframes(equalized_overlap)

    testfile = pathlib.Path(os.getcwd()) / "overlapped_audio_bytes.wav"
    with wave.open(str(testfile), "wb") as waveFile:
        waveFile.setnchannels(1)
        waveFile.setsampwidth(2)
        waveFile.setframerate(settings.rate)
        waveFile.writeframes(overlapped_audio_bytes)

    log.info(
        "-----------------------------------------------------------testing---------------------------------------------------"
    )
    return OverlapResult(
        audio=overlapped_audio_bytes,
        tail=next_chunk_tail_bytes,
        overlap_len=overlap_len,
    )


def to_words(text: str) -> list[str]:
    """Splits a text string into a list of individual non-empty words.

    Steps:
    1. Strip leading and trailing whitespace from the input string.
    2. Split the string by whitespace.
    3. Filter out any empty strings and return the list of words.
    """
    return [word for word in text.strip().split() if word]


def norm_word(word: str) -> str:
    """Cleans a single word for robust comparison by lowercasing and stripping punctuation.

    Steps:
    1. Convert the word to lowercase.
    2. Remove non-alphanumeric characters (excluding apostrophes) from the start and end using regex.
    3. Return the cleaned word.
    """
    lowered_word: str = word.lower()
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", lowered_word)


def get_words(text: str) -> tuple[list[str], list[str]]:
    """Generates lists of original words and normalized words from a text string.

    Steps:
    1. Split the text into individual words to preserve original formatting.
    2. Normalize each word by lowercasing and removing punctuation.
    3. Filter out any empty normalized words.
    4. Return the list of original words and the list of normalized words.
    """
    original_words: list[str] = to_words(text)
    overlap_matching_words: list[str] = [norm for w in original_words if (norm := norm_word(w))]
    return original_words, overlap_matching_words


def should_skip_trim(
    orig: list[str],
    trimmed: list[str],
    overlap_words: int,
) -> bool:
    """Determines if trimming the overlap would leave the remaining text too short.

    Steps:
    1. Check if the trimmed word count is 1 or fewer.
    2. Check if the matched overlap is 3 or more words.
    3. Check if the original count equals the overlap count plus the trimmed count.
    4. Return True if all conditions are met, indicating the trim should be skipped.
    """
    return (
        len(trimmed) <= 1
        and overlap_words >= 3
        and len(orig) == overlap_words + len(trimmed)
    )


def char_sim(words_a: list[str], words_b: list[str]) -> float:
    """Computes character-level similarity between two lists of words.

    Steps:
    1. If either list is empty, return a similarity score of 0.0.
    2. Join each word list into a single space-separated string.
    3. Compute the sequence match ratio between the two joined strings.
    4. Log the comparison details and the computed score.
    5. Return the similarity score (0.0 to 1.0).
    """
    if not words_a or not words_b:
        return 0.0

    str_a: str = " ".join(words_a)
    str_b: str = " ".join(words_b)

    score: float = SequenceMatcher(None, str_a, str_b).ratio()
    log.debug(
        "[Dedup] char_similarity",
        str_a=str_a,
        str_b=str_b,
        score=round(score, 4),
    )
    return score


def token_sim(words_a: list[str], words_b: list[str]) -> float:
    """Computes token Jaccard similarity between two lists of words.

    Steps:
    1. If either list is empty, return 0.0.
    2. Convert both word lists to sets of unique words.
    3. Find the intersection (common words) and union (all unique words) of the two sets.
    4. Calculate the ratio of intersection size to union size.
    5. Log the word difference details and computed score.
    6. Return the Jaccard similarity score (0.0 to 1.0).
    """
    if not words_a or not words_b:
        return 0.0

    set_a: set[str] = set(words_a)
    set_b: set[str] = set(words_b)

    intersection: set[str] = set_a.intersection(set_b)
    union: set[str] = set_a.union(set_b)

    score: float = len(intersection) / len(union) if union else 0.0
    log.debug(
        "[Dedup] token_overlap",
        words_a=words_a,
        words_b=words_b,
        shared_words=sorted(intersection),
        only_in_a=sorted(set_a - set_b),
        only_in_b=sorted(set_b - set_a),
        score=round(score, 4),
    )
    return score


def dedup_prefix(
    last_text: str,
    curr_text: str,
    *,
    max_words: int = 15,
) -> DedupResult:
    """Detects and trims overlapping duplicate words at the start of curr_text.

    Steps:
    1. Initialize the default result with the stripped current text and zero overlap scores.
    2. If either string is empty, return the default result.
    3. Split and normalize both the last text and current text.
    4. Determine the maximum possible overlap length up to max_words.
    5. Iterate backwards from the largest possible overlap down to 2 words:
       - Extract the tail words of the last text and head words of the current text.
       - Calculate character similarity and token similarity between the segments.
       - Compute a weighted combined similarity score: 60% character similarity + 40% token similarity.
       - If combined score is >= the semantic overlapping threshold from settings:
         - Trim the overlapping words from the original current text.
         - Check if the trim should be skipped because the remaining text is too short.
         - Return the deduplication result with the appropriate trimmed text.
    6. Return the default result if no significant overlap is found.
    """
    result: DedupResult = DedupResult(
        text=curr_text.strip(),
        overlap_words=0,
        char_score=0.0,
        token_score=0.0,
        combined_score=0.0,
        trimmed=False,
        skipped=False,
    )

    if not last_text or not curr_text:
        return result

    _prev_original: list[str]
    prev_normalized: list[str]
    _prev_original, prev_normalized = get_words(last_text)

    curr_original: list[str]
    curr_normalized: list[str]
    curr_original, curr_normalized = get_words(curr_text)

    largest_possible_overlap: int = min(
        len(prev_normalized),
        len(curr_normalized),
        max_words,
    )

    for overlap_words in range(largest_possible_overlap, 1, -1):
        prev_tail: list[str] = prev_normalized[-overlap_words:]
        curr_head: list[str] = curr_normalized[:overlap_words]

        char_score: float = char_sim(prev_tail, curr_head)
        token_score: float = token_sim(prev_tail, curr_head)
        combined_score: float = (char_score * 0.6) + (token_score * 0.4)

        if combined_score >= settings.semantic_overlapping_threshold:
            trimmed_words: list[str] = curr_original[overlap_words:]
            skipped: bool = should_skip_trim(curr_original, trimmed_words, overlap_words)

            cleaned_text: str = (
                curr_text.strip() if skipped else " ".join(trimmed_words).strip()
            )

            return DedupResult(
                text=cleaned_text,
                overlap_words=overlap_words,
                char_score=char_score,
                token_score=token_score,
                combined_score=combined_score,
                trimmed=not skipped,
                skipped=skipped,
            )

    return result


def norm_text(text: str) -> str:
    """Normalizes text for word error rate evaluation by cleaning whitespace and punctuation.

    Steps:
    1. Convert the entire text to lowercase.
    2. Replace all non-alphanumeric characters (except apostrophes) with spaces.
    3. Collapse multiple consecutive whitespaces into a single space and strip boundaries.
    4. Return the normalized text.
    """
    lowered_text: str = text.lower()
    punctuation_removed_text: str = re.sub(r"[^a-z0-9\s']", " ", lowered_text)
    collapsed_spacing_text: str = re.sub(r"\s+", " ", punctuation_removed_text).strip()
    return collapsed_spacing_text


class StreamingSession:
    """Tracks the audio overlap state and timing across a streaming session."""

    _overlap_secs: float
    _rate: int
    _overlap_bytes: int
    _tail: bytes
    _start_time: float

    def __init__(
        self,
        overlap_secs: float = 1.0,
        rate: int = settings.rate,
    ):
        """Initializes a streaming session with configured overlap duration and sample rate."""
        self._overlap_secs = overlap_secs
        self._rate = rate
        self._overlap_bytes = int(self._rate * 2 * self._overlap_secs)
        self._tail = b""
        self._start_time = time.time()

    def process_chunk(
        self,
        audio: bytes,
        stop: bool,
        silence_secs: float,
    ) -> bytes:
        """Processes an incoming audio chunk by applying overlap and updating session state.

        Steps:
        1. Calculate the number of silence bytes based on silence duration and sample rate.
        2. Apply overlap to the current audio chunk using stored tail bytes and config.
        3. Save the new tail bytes returned from overlap application.
        4. Reset the chunk start timer to the current time.
        5. Return the overlapped audio bytes.
        """
        silence_bytes: int = int(silence_secs * self._rate * 2)
        result: OverlapResult = apply_overlap(
            audio=audio,
            tail=self._tail,
            overlap_bytes=self._overlap_bytes,
            silence_bytes=silence_bytes,
            rate=self._rate,
            stop=stop,
        )
        self._tail = result.tail
        self._start_time = time.time()
        return result.audio

    def reset(self) -> None:
        """Resets the streaming session's audio tail and start timer."""
        self._tail = b""
        self._start_time = time.time()
