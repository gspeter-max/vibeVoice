import sys
from types import SimpleNamespace

import numpy as np
from streaming_shared_logic import normalize_text_for_word_error_rate

from evaluation.parakeet_v2_streaming_evaluation import (
    DEFAULT_OUTPUT_JSON_FILE_PATH,
    add_previous_chunk_overlap_to_current_chunk_audio,
    build_command_line_argument_parser,
    calculate_word_error_rate_for_final_streaming_text,
    load_one_dataset_item_audio_and_reference_text,
    load_pcm16k_audio_array_from_audio_feature_value,
    main,
    remove_repeated_words_from_current_chunk_text,
    run_fake_microphone_stream_for_one_dataset_item,
    save_streaming_evaluation_run_to_json_file,
    split_audio_bytes_into_microphone_frames,
)


def test_split_audio_bytes_into_microphone_frames_keeps_order():
    pcm16_audio_bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00"

    frames = split_audio_bytes_into_microphone_frames(
        pcm16_audio_bytes,
        frame_samples=2,
    )

    assert frames == [
        b"\x01\x00\x02\x00",
        b"\x03\x00\x04\x00",
    ]


def test_add_previous_chunk_overlap_to_current_chunk_audio():
    overlapped_audio, next_overlap_audio = add_previous_chunk_overlap_to_current_chunk_audio(
        previous_chunk_overlap_audio=b"\x01\x00\x02\x00",
        current_chunk_audio=b"\x03\x00\x04\x00\x05\x00\x06\x00",
        overlap_audio_bytes=4,
        stop_session=False,
    )

    assert overlapped_audio == b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00"
    assert next_overlap_audio == b"\x05\x00\x06\x00"


def test_remove_repeated_words_from_current_chunk_text():
    cleaned_text = remove_repeated_words_from_current_chunk_text(
        previous_chunk_text="things are happening fine",
        current_chunk_text="things are happening fine and doing H3 grid",
        max_overlap_words=8,
    )

    assert cleaned_text == "and doing H3 grid"


def test_calculate_word_error_rate_for_final_streaming_text():
    word_error_rate = calculate_word_error_rate_for_final_streaming_text(
        reference_text="hello world",
        final_streaming_text="hello world",
    )

    assert word_error_rate == 0.0


def test_build_command_line_argument_parser_uses_clear_defaults():
    argument_parser = build_command_line_argument_parser()

    parsed_arguments = argument_parser.parse_args([])

    assert parsed_arguments.dataset_name_to_load == "hf-internal-testing/librispeech_asr_dummy"
    assert parsed_arguments.dataset_config_name_to_load == "clean"
    assert parsed_arguments.dataset_split_to_load == "validation"
    assert parsed_arguments.audio_column_name_to_load == "audio"
    assert parsed_arguments.text_column_name_to_load == "text"
    assert parsed_arguments.single_sample_number_to_test == 0
    assert parsed_arguments.overlap_seconds_between_chunks == 0.50
    assert parsed_arguments.output_json_file_path == DEFAULT_OUTPUT_JSON_FILE_PATH


def test_build_command_line_argument_parser_accepts_dataset_overrides():
    argument_parser = build_command_line_argument_parser()

    parsed_arguments = argument_parser.parse_args(
        [
            "--dataset-name-to-load",
            "custom/dataset",
            "--dataset-config-name-to-load",
            "clean",
            "--dataset-split-to-load",
            "validation",
            "--audio-column-name-to-load",
            "sound",
            "--text-column-name-to-load",
            "words",
            "--single-sample-number-to-test",
            "7",
        ]
    )

    assert parsed_arguments.dataset_name_to_load == "custom/dataset"
    assert parsed_arguments.dataset_config_name_to_load == "clean"
    assert parsed_arguments.dataset_split_to_load == "validation"
    assert parsed_arguments.audio_column_name_to_load == "sound"
    assert parsed_arguments.text_column_name_to_load == "words"
    assert parsed_arguments.single_sample_number_to_test == 7


def test_build_command_line_argument_parser_accepts_multi_sample_arguments():
    argument_parser = build_command_line_argument_parser()

    parsed_arguments = argument_parser.parse_args(
        [
            "--first-sample-number-to-test",
            "2",
            "--how-many-samples-to-test",
            "3",
        ]
    )

    assert parsed_arguments.first_sample_number_to_test == 2
    assert parsed_arguments.how_many_samples_to_test == 3


def test_load_one_dataset_item_audio_and_reference_text_uses_dataset_config_name_without_trust_remote_code(monkeypatch):
    captured_load_dataset_call = {}

    class FakeDataset:
        def cast_column(self, audio_column_name, audio_config):
            assert audio_column_name == "audio"
            assert audio_config.sampling_rate == 16000
            assert audio_config.decode is False
            return self

        def __getitem__(self, example_index):
            assert example_index == 5
            return {
                "audio": {"path": "/tmp/fake_audio.wav", "bytes": None},
                "text": " hello there ",
            }

    def fake_load_dataset(dataset_name, dataset_config_name=None, *, split=None):
        captured_load_dataset_call.update(
            {
                "dataset_name": dataset_name,
                "dataset_config_name": dataset_config_name,
                "split": split,
            }
        )
        return FakeDataset()

    class FakeAudio:
        def __init__(self, sampling_rate, decode=False):
            self.sampling_rate = sampling_rate
            self.decode = decode

    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.load_pcm16k_audio_array_from_audio_feature_value",
        lambda audio_feature_value: [0.1, -0.1],
    )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Audio=FakeAudio, load_dataset=fake_load_dataset),
    )

    audio_array, reference_text = load_one_dataset_item_audio_and_reference_text(
        dataset_name="hf-internal-testing/librispeech_asr_dummy",
        dataset_config_name="clean",
        dataset_split="validation",
        audio_column_name="audio",
        text_column_name="text",
        example_index=5,
    )

    assert audio_array == [0.1, -0.1]
    assert reference_text == "hello there"
    assert captured_load_dataset_call == {
        "dataset_name": "hf-internal-testing/librispeech_asr_dummy",
        "dataset_config_name": "clean",
        "split": "validation",
    }


def test_load_pcm16k_audio_array_from_audio_feature_value_reads_wav_path(tmp_path):
    audio_file_path = tmp_path / "sample.wav"
    audio_file_path.write_bytes(b"fake")
    original_audio_array = np.array([0.0, 0.25, -0.25, 0.1], dtype=np.float32)

    sys.modules["soundfile"] = SimpleNamespace(
        read=lambda path, dtype="float32": (original_audio_array, 16000)
    )

    loaded_audio_array = load_pcm16k_audio_array_from_audio_feature_value(
        {"path": str(audio_file_path), "bytes": None}
    )

    assert np.allclose(loaded_audio_array, original_audio_array, atol=1e-4)


def test_run_fake_microphone_stream_tracks_chunk_event_text_states(monkeypatch):
    class FakeUtteranceGate:
        def __init__(self, *args, **kwargs):
            self.should_finalize_call_count = 0
            self.flush_call_count = 0

        def push(self, frame_bytes, now):
            return True

        def should_finalize(self, now):
            self.should_finalize_call_count += 1
            return self.should_finalize_call_count in {2, 3}

        def flush(self):
            self.flush_call_count += 1
            if self.flush_call_count == 1:
                return b"\x01\x00\x02\x00\x03\x00\x04\x00"
            if self.flush_call_count == 2:
                return b"\x05\x00\x06\x00\x07\x00\x08\x00"
            return b""

        def silence_elapsed(self, now):
            return 0.70

    transcriptions_by_audio_bytes = {
        b"\x01\x00\x02\x00\x03\x00\x04\x00": "hello there",
        b"\x05\x00\x06\x00\x07\x00\x08\x00": "there friend",
        b"\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00": "hello there friend",
    }

    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.load_evaluation_model",
        lambda: object(),
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.transcribe_one_audio_chunk",
        lambda _model, chunk_audio_bytes: transcriptions_by_audio_bytes[chunk_audio_bytes],
    )
    monkeypatch.setitem(
        sys.modules,
        "src.vad_segmenter",
        SimpleNamespace(SileroUtteranceGate=FakeUtteranceGate, SileroVAD=lambda *_args, **_kwargs: object()),
    )

    streaming_result = run_fake_microphone_stream_for_one_dataset_item(
        pcm16_audio_bytes=(b"\x00\x00" * 16000) * 3,
        reference_text="HELLO, there friend!",
        vad_model_path="/tmp/fake_silero.onnx",
        silence_timeout_seconds=0.65,
        vad_threshold=0.50,
        energy_threshold=0.05,
        energy_ratio=2.5,
        overlap_seconds=0.000125,
        max_overlap_words=8,
        frame_samples=16000,
        minimum_chunk_age_before_silence_split_seconds=0.0,
    )

    assert streaming_result["final_streaming_text"] == "hello there friend"
    assert streaming_result["final_word_error_rate"] == 0.0
    assert streaming_result["chunk_events"] == [
        {
            "chunk_index": 0,
            "split_reason": "silence_threshold_hit",
            "chunk_age_seconds_when_split_happened": 1.0,
            "silence_duration_seconds_when_split_happened": 0.7,
            "chunk_duration_seconds_before_overlap": 0.00025,
            "overlap_seconds_added_from_previous_chunk": 0.0,
            "raw_chunk_text_without_overlap": "hello there",
            "raw_chunk_text_with_overlap": "hello there",
            "cleaned_chunk_text_after_dedup": "hello there",
        },
        {
            "chunk_index": 1,
            "split_reason": "silence_threshold_hit",
            "chunk_age_seconds_when_split_happened": 1.0,
            "silence_duration_seconds_when_split_happened": 0.7,
            "chunk_duration_seconds_before_overlap": 0.00025,
            "overlap_seconds_added_from_previous_chunk": 0.000125,
            "raw_chunk_text_without_overlap": "there friend",
            "raw_chunk_text_with_overlap": "hello there friend",
            "cleaned_chunk_text_after_dedup": "friend",
        },
    ]


def test_save_streaming_evaluation_run_to_json_file(tmp_path):
    output_json_file_path = tmp_path / "nested" / "report.json"

    save_streaming_evaluation_run_to_json_file(
        output_json_file_path=str(output_json_file_path),
        streaming_evaluation_run_report={"run_summary": {}, "sample_results": []},
    )

    assert output_json_file_path.exists()
    assert output_json_file_path.read_text(encoding="utf-8") == '{\n  "run_summary": {},\n  "sample_results": []\n}'


def test_calculate_word_error_rate_uses_normalized_text():
    normalized_reference_text = normalize_text_for_word_error_rate("HELLO, WORLD!")
    normalized_final_streaming_text = normalize_text_for_word_error_rate("hello world")

    assert calculate_word_error_rate_for_final_streaming_text(
        reference_text=normalized_reference_text,
        final_streaming_text=normalized_final_streaming_text,
    ) == 0.0


def test_main_prints_progress_logs_and_final_result(monkeypatch, capsys):
    captured_dataset_request = {}
    captured_run_request = {}
    captured_json_report = {}

    def fake_load_one_dataset_item_audio_and_reference_text(**kwargs):
        captured_dataset_request.update(kwargs)
        return [0.0, 0.25, -0.25], "hello world"

    def fake_convert_audio_array_to_pcm16_audio_bytes(audio_array):
        assert audio_array == [0.0, 0.25, -0.25]
        return b"\x01\x00\x02\x00"

    def fake_run_fake_microphone_stream_for_one_dataset_item(**kwargs):
        captured_run_request.update(kwargs)
        return {
            "reference_text": "hello world",
            "final_streaming_text": "hello world",
            "final_word_error_rate": 0.0,
            "chunk_count": 1,
            "chunk_texts": ["hello world"],
            "chunk_durations_seconds": [0.25],
        }

    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.load_one_dataset_item_audio_and_reference_text",
        fake_load_one_dataset_item_audio_and_reference_text,
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.convert_audio_array_to_pcm16_audio_bytes",
        fake_convert_audio_array_to_pcm16_audio_bytes,
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.run_fake_microphone_stream_for_one_dataset_item",
        fake_run_fake_microphone_stream_for_one_dataset_item,
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.save_streaming_evaluation_run_to_json_file",
        lambda **kwargs: captured_json_report.update(kwargs),
    )

    main(
        [
            "--dataset-name-to-load",
            "toy/data",
            "--dataset-config-name-to-load",
            "toy_config",
            "--dataset-split-to-load",
            "train",
            "--single-sample-number-to-test",
            "3",
            "--vad-model-file-path",
            "/tmp/fake_silero.onnx",
        ]
    )

    command_output = capsys.readouterr().out

    assert "[Evaluation] Loading dataset item 3..." in command_output
    assert "[Evaluation] Converting audio to PCM16 bytes..." in command_output
    assert "[Evaluation] Running fake microphone stream..." in command_output
    assert "[Evaluation] Final result for example 3:" in command_output
    assert "final_word_error_rate" in command_output
    assert captured_dataset_request == {
        "dataset_name": "toy/data",
        "dataset_config_name": "toy_config",
        "dataset_split": "train",
        "audio_column_name": "audio",
        "text_column_name": "text",
        "example_index": 3,
    }
    assert captured_run_request["pcm16_audio_bytes"] == b"\x01\x00\x02\x00"
    assert captured_run_request["reference_text"] == "hello world"
    assert captured_run_request["vad_model_path"] == "/tmp/fake_silero.onnx"
    assert captured_json_report["output_json_file_path"] == DEFAULT_OUTPUT_JSON_FILE_PATH


def test_main_prints_multi_sample_matrix_and_summary(monkeypatch, capsys):
    captured_example_indexes = []
    captured_json_report = {}

    def fake_load_one_dataset_item_audio_and_reference_text(**kwargs):
        example_index = kwargs["example_index"]
        captured_example_indexes.append(example_index)
        return [0.0, 0.25, -0.25], f"reference-{example_index}"

    def fake_convert_audio_array_to_pcm16_audio_bytes(audio_array):
        return b"\x01\x00\x02\x00"

    def fake_run_fake_microphone_stream_for_one_dataset_item(**kwargs):
        reference_text = kwargs["reference_text"]
        example_index = int(reference_text.split("-")[-1])
        if example_index == 2:
            return {
                "reference_text": "reference-2",
                "final_streaming_text": "reference-2",
                "final_word_error_rate": 0.0,
                "chunk_count": 1,
                "chunk_texts": ["reference-2"],
                "chunk_durations_seconds": [1.0],
            }
        return {
            "reference_text": "reference-3",
            "final_streaming_text": "reference-3 wrong",
            "final_word_error_rate": 0.5,
            "chunk_count": 2,
            "chunk_texts": ["reference-3", "wrong"],
            "chunk_durations_seconds": [1.5, 0.5],
        }

    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.load_one_dataset_item_audio_and_reference_text",
        fake_load_one_dataset_item_audio_and_reference_text,
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.convert_audio_array_to_pcm16_audio_bytes",
        fake_convert_audio_array_to_pcm16_audio_bytes,
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.run_fake_microphone_stream_for_one_dataset_item",
        fake_run_fake_microphone_stream_for_one_dataset_item,
    )
    monkeypatch.setattr(
        "evaluation.parakeet_v2_streaming_evaluation.save_streaming_evaluation_run_to_json_file",
        lambda **kwargs: captured_json_report.update(kwargs),
    )

    main(
        [
            "--first-sample-number-to-test",
            "2",
            "--how-many-samples-to-test",
            "2",
        ]
    )

    command_output = capsys.readouterr().out

    assert captured_example_indexes == [2, 3]
    assert "example_index" in command_output
    assert "average_final_word_error_rate" in command_output
    assert "average_chunk_count" in command_output
    assert "reference-2" in command_output
    assert "reference-3 wrong" in command_output
    assert captured_json_report["output_json_file_path"] == DEFAULT_OUTPUT_JSON_FILE_PATH
