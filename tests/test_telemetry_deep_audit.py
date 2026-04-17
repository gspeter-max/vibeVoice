import json
import pytest
from pathlib import Path
from streaming_session_telemetry import StreamingSessionTelemetryRecorder


def test_telemetry_structural_audit(tmp_path: Path):
    """
    DEEP AUDIT: Structural Integrity Check.
    Verifies that every required key in the JSON schema exists and has the correct type.
    This ensures that changing a variable name (like 'prev' to 'prev_text') without
    updating the telemetry will fail this test.
    """
    recorder = StreamingSessionTelemetryRecorder(
        session_id="audit_session_001",
        output_dir=tmp_path,
        summary_seed={"model": "base.en", "backend": "faster_whisper"},
    )

    recorder.update_chunk_summary(
        0,
        0,
        {
            "previous_chunk_text": "hello",
            "raw_text": "hello world",
            "cleaned_text_after_dedup": "world",
            "dedup_stats": {
                "overlap_word_count": 1,
                "trim_applied": True,
                "combined_score": 0.95,
                "char_score": 0.98,
                "token_score": 0.92,
                "skipped_too_small": False,
            },
        },
    )

    recorder.write_snapshot()
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    with open(json_files[0], "r") as f:
        data = json.load(f)

    assert "session_summary" in data, "MISSING KEY: session_summary"
    assert "recordings" in data, "MISSING KEY: recordings"

    summary = data["session_summary"]
    required_summary_keys = [
        "session_id",
        "total_chunks_received",
        "total_decode_seconds",
        "model",
        "backend",
    ]
    for key in required_summary_keys:
        assert key in summary, f"STRUCTURAL FAILURE: Session summary missing '{key}'"

    assert len(data["recordings"]) > 0, "MISSING DATA: recordings list is empty"
    chunk = data["recordings"][0]["chunks"][0]
    assert "summary" in chunk
    chunk_summary = chunk["summary"]

    critical_keys = [
        "previous_chunk_text",
        "raw_text",
        "cleaned_text_after_dedup",
        "dedup_stats",
    ]
    for key in critical_keys:
        assert key in chunk_summary, (
            f"REGRESSION DETECTED: Chunk summary missing critical key '{key}'"
        )

    stats = chunk_summary["dedup_stats"]
    assert isinstance(stats["combined_score"], float)
    assert isinstance(stats["overlap_word_count"], int)
    assert isinstance(stats["trim_applied"], bool)


def test_telemetry_out_of_order_chaos_audit(tmp_path: Path):
    """
    DEEP AUDIT: Chaos Handling.
    Verifies that chunks arriving out of order (Chunk 5 before Chunk 0)
    doesn't crash the recorder and maintains list integrity.
    """
    recorder = StreamingSessionTelemetryRecorder(
        session_id="chaos_session", output_dir=tmp_path, summary_seed={}
    )

    recorder.update_chunk_summary(0, 5, {"data": "I arrived early"})
    recorder.update_chunk_summary(0, 0, {"data": "I arrived late"})

    recorder.write_snapshot()
    json_files = list(tmp_path.glob("*chaos_session*.json"))
    assert len(json_files) == 1
    with open(json_files[0], "r") as f:
        data = json.load(f)

    assert len(data["recordings"]) == 1, (
        "CHAOS FAILURE: Should have exactly 1 recording (index 0)"
    )
    assert len(data["recordings"][0]["chunks"]) == 6, (
        "CHAOS FAILURE: List didn't expand to accommodate high-index chunk"
    )
    assert data["recordings"][0]["chunks"][5]["summary"]["data"] == "I arrived early"
    assert data["recordings"][0]["chunks"][0]["summary"]["data"] == "I arrived late"


def test_telemetry_atomic_write_safety_audit(tmp_path: Path):
    """
    DEEP AUDIT: Safety & Persistence.
    Verifies that write_snapshot produces a valid JSON even if called rapidly.
    """
    recorder = StreamingSessionTelemetryRecorder(
        session_id="atomic_session", output_dir=tmp_path, summary_seed={}
    )

    for i in range(100):
        recorder.update_session_summary({"flags": {f"event_{i}": True}})

    json_files = list(tmp_path.glob("*atomic_session*.json"))
    assert len(json_files) == 1
    with open(json_files[0], "r") as f:
        data = json.load(f)
        assert len(data["session_summary"]["flags"]) == 100


def test_telemetry_metadata_type_audit(tmp_path: Path):
    """
    DEEP AUDIT: Type Strictness.
    Ensures float values aren't being stored as strings, which breaks analysis tools.
    """
    recorder = StreamingSessionTelemetryRecorder(
        session_id="type_session", output_dir=tmp_path, summary_seed={}
    )

    recorder.update_session_summary({"total_decode_seconds": 12.3456})

    json_files = list(tmp_path.glob("*type_session*.json"))
    assert len(json_files) == 1
    with open(json_files[0], "r") as f:
        data = json.load(f)
        val = data["session_summary"]["total_decode_seconds"]
        assert isinstance(val, (float, int)), (
            f"TYPE FAILURE: Expected float but got {type(val)}"
        )
