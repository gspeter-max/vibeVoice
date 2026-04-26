from pathlib import Path

from src.streaming.streaming_session_telemetry import StreamingSessionTelemetryRecorder


def test_recorder_writes_one_session_file_atomically(tmp_path: Path):
    recorder = StreamingSessionTelemetryRecorder(
        session_id="session123",
        output_dir=tmp_path,
        summary_seed={"recording_mode": "silence_streaming"},
    )

    recorder.update_session_summary({"flags": {"test_flag": True}})
    recorder.write_snapshot()

    session_files = list(tmp_path.glob("*.json"))
    assert len(session_files) == 1
    payload = session_files[0].read_text(encoding="utf-8")

    assert '"session_id": "session123"' in payload
    assert '"recording_mode": "silence_streaming"' in payload


def test_recorder_updates_chunk_summary_fields(tmp_path: Path):
    recorder = StreamingSessionTelemetryRecorder(
        session_id="session123",
        output_dir=tmp_path,
        summary_seed={"recording_mode": "silence_streaming"},
    )

    recorder.update_chunk_summary(
        0,
        0,
        {
            "audio_bytes": 290048,
            "raw_text_before_overlap": "hello there",
            "cleaned_text_after_dedup": "hello there",
        },
    )

    recorder.write_snapshot()
    session_files = list(tmp_path.glob("*.json"))
    assert len(session_files) == 1
    payload = session_files[0].read_text(encoding="utf-8")

    assert '"audio_bytes": 290048' in payload
    assert '"cleaned_text_after_dedup": "hello there"' in payload


def test_recorder_keeps_true_flags_when_later_updates_are_false(tmp_path: Path):
    recorder = StreamingSessionTelemetryRecorder(
        session_id="session123",
        output_dir=tmp_path,
        summary_seed={"recording_mode": "silence_streaming"},
    )

    recorder.update_session_summary({"flags": {"dedup_trim_applied": True}})
    recorder.update_session_summary({"flags": {"dedup_trim_applied": False}})

    session_files = list(tmp_path.glob("*.json"))
    assert len(session_files) == 1
    payload = session_files[0].read_text(encoding="utf-8")

    assert '"dedup_trim_applied": true' in payload
