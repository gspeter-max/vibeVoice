from src.streaming.session import (
    _equalize_energy,
    apply_overlap,
    dedup_prefix,
    norm_text,
    should_split,
)


def test_should_split_chunk_after_silence_requires_both_gate_and_age():
    split_decision = should_split(
        start_time=10.0,
        now=20.5,
        min_age=12.0,
        gate_finalize=True,
        silence_len=0.70,
    )

    assert split_decision.should_split is False
    assert split_decision.age == 10.5


def test_should_split_negative_age_clamp():
    # Test that a negative age (clock skew) is clamped to 0.0
    split_decision = should_split(
        start_time=20.0,
        now=10.0,  # now < start_time
        min_age=5.0,
        gate_finalize=True,
        silence_len=0.5,
    )
    assert split_decision.age == 0.0
    assert split_decision.should_split is False


def test_equalize_energy_pcm_alignment():
    # Test with odd length bytes
    overlap_bytes = b"\x01\x00\x02\x00\x03"  # 5 bytes
    current_bytes = b"\x04\x00\x05\x00"      # 4 bytes
    result = _equalize_energy(overlap_bytes, current_bytes)
    # The result should be of even length (clipped/trimmed)
    assert len(result) == 4


def test_equalize_energy_zero_energy():
    # Test with silent overlap or silent current audio (RMS < 1.0)
    overlap_bytes = b"\x00\x00" * 10
    current_bytes = b"\x00\x00" * 10
    result = _equalize_energy(overlap_bytes, current_bytes)
    assert result == overlap_bytes


def test_equalize_energy_gain_capping():
    # Test with large energy differences (gain should be capped at 3.0)
    overlap_bytes = b"\x01\x00" * 10  # low amplitude
    current_bytes = b"\x00\x7f" * 10  # high amplitude
    result = _equalize_energy(overlap_bytes, current_bytes)
    # Gain should scale up but not exceed 3.0
    assert len(result) == len(overlap_bytes)
    assert result != overlap_bytes


def test_dedup_prefix_empty_inputs():
    result = dedup_prefix("", "")
    assert result.text == ""
    assert result.overlap_words == 0
    assert result.trimmed is False


def test_apply_last_chunk_overlap_keeps_first_chunk_clean():
    overlap_result = apply_overlap(
        audio=b"\x01\x00\x02\x00\x03\x00\x04\x00",
        tail=b"",
        overlap_bytes=4,
        rate=16000,
        stop=False,
    )

    assert overlap_result.audio == b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert overlap_result.tail == b"\x03\x00\x04\x00"
    assert overlap_result.overlap_len == 0.0


def test_apply_last_chunk_overlap_skips_silence():
    # 16 bytes total. 4 bytes of "silence" at the end. 4 bytes of overlap requested.
    audio = b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x00\x00\x00\x00"
    result = apply_overlap(
        audio=audio,
        tail=b"",
        overlap_bytes=4,
        silence_bytes=4,
        rate=16000,
        stop=False,
    )
    # Should take bytes from index 8 to 12 (0x05 0x00 0x06 0x00)
    # Index 12 to 16 is the "silence" (0x00 0x00 0x00 0x00)
    assert result.tail == b"\x05\x00\x06\x00"


def test_analyze_duplicate_chunk_prefix_removes_exact_overlapping_words_from_start():
    result = dedup_prefix(
        "things are happening fine",
        "things are happening fine and doing work",
        max_words=8,
    )
    assert result.text == "and doing work"
    assert result.trimmed is True


def test_analyze_duplicate_chunk_prefix_ignores_letter_case_and_punctuation_when_matching():
    result = dedup_prefix(
        "that I made.",
        "That I made a few months ago while writing an article for Italian Wired.",
        max_words=8,
    )
    assert result.text == "a few months ago while writing an article for Italian Wired."
    assert result.trimmed is True


def test_analyze_duplicate_chunk_prefix_keeps_original_text_when_trim_would_leave_almost_nothing():
    result = dedup_prefix(
        "once in my",
        "once in my life.",
        max_words=8,
    )
    assert result.text == "once in my life."
    assert result.skipped is True
    assert result.trimmed is False


def test_analyze_duplicate_chunk_prefix_reports_trim_details():
    result = dedup_prefix(
        "things are happening fine",
        "things are happening fine and more",
        max_words=8,
    )

    assert result.trimmed is True
    assert result.overlap_words == 4
    assert result.text == "and more"
    assert result.combined_score >= 0.82


def test_normalize_text_for_word_error_rate_removes_case_and_punctuation_noise():
    assert norm_text("HELLO, World!!") == "hello world"
