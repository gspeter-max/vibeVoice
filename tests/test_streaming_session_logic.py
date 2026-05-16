from src.streaming.session import (
    apply_last_chunk_overlap,
    analyze_duplicate_chunk_prefix,
    normalize_text_for_word_error_rate,
    should_split_chunk_after_silence,
)


def test_should_split_chunk_after_silence_requires_both_gate_and_age():
    split_decision = should_split_chunk_after_silence(
        chunk_started_at_seconds=10.0,
        now_seconds=20.5,
        minimum_chunk_age_before_silence_split_seconds=12.0,
        utterance_gate_should_finalize_now=True,
        silence_duration_seconds=0.70,
    )

    assert split_decision.should_split_now is False
    assert split_decision.chunk_age_seconds == 10.5


def test_apply_last_chunk_overlap_keeps_first_chunk_clean():
    overlap_application_result = apply_last_chunk_overlap(
        current_chunk_audio_bytes=b"\x01\x00\x02\x00\x03\x00\x04\x00",
        last_chunk_tail_bytes=b"",
        overlap_audio_byte_count=4,
        sample_rate=16000,
        stop_session=False,
    )

    assert overlap_application_result.overlapped_audio_bytes == b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert overlap_application_result.next_chunk_tail_bytes == b"\x03\x00\x04\x00"
    assert overlap_application_result.overlap_seconds_from_last_chunk == 0.0


def test_apply_last_chunk_overlap_skips_silence():
    # 16 bytes total. 4 bytes of "silence" at the end. 4 bytes of overlap requested.
    audio = b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x00\x00\x00\x00"
    result = apply_last_chunk_overlap(
        current_chunk_audio_bytes=audio,
        last_chunk_tail_bytes=b"",
        overlap_audio_byte_count=4,
        silence_audio_byte_count=4,
        sample_rate=16000,
        stop_session=False,
    )
    # Should take bytes from index 8 to 12 (0x05 0x00 0x06 0x00)
    # Index 12 to 16 is the "silence" (0x00 0x00 0x00 0x00)
    assert result.next_chunk_tail_bytes == b"\x05\x00\x06\x00"


def test_analyze_duplicate_chunk_prefix_removes_exact_overlapping_words_from_start():
    """
    When chunk 2 starts with the same words that chunk 1 ended with,
    analyze_ should remove those repeated words and return only the new content.

    Example:
      chunk 1 ends with:   "things are happening fine"
      chunk 2 starts with: "things are happening fine and doing work"
      Result should be:    "and doing work"
    """
    result = analyze_duplicate_chunk_prefix(
        "things are happening fine",
        "things are happening fine and doing work",
        max_overlap_words=8,
    )
    assert result.cleaned_text == "and doing work"
    assert result.trim_applied is True


def test_analyze_duplicate_chunk_prefix_ignores_letter_case_and_punctuation_when_matching():
    """
    The matching should be case-insensitive and should ignore punctuation at word edges.
    "that I made." and "That I made" should be treated as the same words.

    Example:
      chunk 1 ends with:   "that I made."
      chunk 2 starts with: "That I made a few months ago..."
      Result should be:    "a few months ago while writing an article for Italian Wired."
    """
    result = analyze_duplicate_chunk_prefix(
        "that I made.",
        "That I made a few months ago while writing an article for Italian Wired.",
        max_overlap_words=8,
    )
    assert result.cleaned_text == "a few months ago while writing an article for Italian Wired."
    assert result.trim_applied is True


def test_analyze_duplicate_chunk_prefix_keeps_original_text_when_trim_would_leave_almost_nothing():
    """
    Safety check: if removing the overlapping words would leave the new chunk with
    1 or fewer words, we skip the trim to avoid losing real content.

    Example:
      chunk 1 ends with:   "once in my"
      chunk 2 starts with: "once in my life."
      Removing "once in my" would leave only "life." — just 1 word.
      So the full text "once in my life." should be kept unchanged.
    """
    result = analyze_duplicate_chunk_prefix(
        "once in my",
        "once in my life.",
        max_overlap_words=8,
    )
    assert result.cleaned_text == "once in my life."
    assert result.skipped_because_result_too_small is True
    assert result.trim_applied is False


def test_analyze_duplicate_chunk_prefix_reports_trim_details():
    result = analyze_duplicate_chunk_prefix(
        "things are happening fine",
        "things are happening fine and more",
        max_overlap_words=8,
    )

    assert result.trim_applied is True
    assert result.overlap_word_count == 4
    assert result.cleaned_text == "and more"
    assert result.combined_score >= 0.82


def test_normalize_text_for_word_error_rate_removes_case_and_punctuation_noise():
    assert normalize_text_for_word_error_rate("HELLO, World!!") == "hello world"
