from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSplitDecision:
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
    chunk_age_seconds = max(0.0, now_seconds - chunk_started_at_seconds)
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
    if stop_session:
        return OverlapApplicationResult(
            overlapped_audio_bytes=current_chunk_audio_bytes,
            next_pending_overlap_audio_bytes=b"",
            overlap_seconds_added_from_previous_chunk=0.0,
        )

    overlapped_audio_bytes = previous_pending_overlap_audio_bytes + current_chunk_audio_bytes
    if overlap_audio_byte_count > 0:
        next_pending_overlap_audio_bytes = current_chunk_audio_bytes[-overlap_audio_byte_count:]
    else:
        next_pending_overlap_audio_bytes = b""

    overlap_seconds_added_from_previous_chunk = len(previous_pending_overlap_audio_bytes) / 2.0 / sample_rate
    return OverlapApplicationResult(
        overlapped_audio_bytes=overlapped_audio_bytes,
        next_pending_overlap_audio_bytes=next_pending_overlap_audio_bytes,
        overlap_seconds_added_from_previous_chunk=overlap_seconds_added_from_previous_chunk,
    )


def split_text_into_comparable_words(text: str) -> list[str]:
    return [word for word in text.strip().split() if word]


def remove_duplicate_chunk_prefix(
    previous_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 8,
) -> str:
    if not previous_chunk_text or not current_chunk_text:
        return current_chunk_text.strip()

    previous_chunk_words = split_text_into_comparable_words(previous_chunk_text)
    current_chunk_words = split_text_into_comparable_words(current_chunk_text)
    largest_possible_overlap = min(
        len(previous_chunk_words),
        len(current_chunk_words),
        max_overlap_words,
    )

    for overlap_word_count in range(largest_possible_overlap, 0, -1):
        if previous_chunk_words[-overlap_word_count:] == current_chunk_words[:overlap_word_count]:
            return " ".join(current_chunk_words[overlap_word_count:]).strip()

    return current_chunk_text.strip()


def normalize_text_for_word_error_rate(text: str) -> str:
    lowered_text = text.lower()
    punctuation_removed_text = re.sub(r"[^a-z0-9\s']", " ", lowered_text)
    collapsed_spacing_text = re.sub(r"\s+", " ", punctuation_removed_text).strip()
    return collapsed_spacing_text
