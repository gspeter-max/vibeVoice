from src.streaming.streaming_shared_logic import (
    apply_last_chunk_overlap,
    analyze_duplicate_chunk_prefix,
    normalize_text_for_word_error_rate,
    remove_duplicate_chunk_prefix,
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


def test_remove_duplicate_chunk_prefix_removes_exact_overlap_only():
    assert remove_duplicate_chunk_prefix(
        "things are happening fine",
        "things are happening fine and doing work",
        max_overlap_words=8,
    ) == "and doing work"


def test_remove_duplicate_chunk_prefix_ignores_case_and_edge_punctuation_for_matching():
    assert remove_duplicate_chunk_prefix(
        "that I made.",
        "That I made a few months ago while writing an article for Italian Wired.",
        max_overlap_words=8,
    ) == "a few months ago while writing an article for Italian Wired."


def test_remove_duplicate_chunk_prefix_keeps_text_when_overlap_trim_would_be_too_small():
    assert remove_duplicate_chunk_prefix(
        "once in my",
        "once in my life.",
        max_overlap_words=8,
    ) == "once in my life."


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
