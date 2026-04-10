import argparse
import io
import json
import os
from pathlib import Path
from typing import Any
from src import log 
import numpy as np
from src.streaming_shared_logic import (
    DEFAULT_ENERGY_RATIO,
    DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_SILENCE_TIMEOUT_SECONDS,
    DEFAULT_VAD_ENERGY_THRESHOLD,
    DEFAULT_VAD_SCORE_THRESHOLD,
    apply_previous_chunk_overlap,
    normalize_text_for_word_error_rate,
    remove_duplicate_chunk_prefix,
    should_split_chunk_after_silence,
)


PARAKEET_V2_MODEL_NAME = "parakeet-tdt-0.6b-v2"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_SAMPLES = 512
DEFAULT_DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"
DEFAULT_DATASET_CONFIG_NAME = None
DEFAULT_DATASET_SPLIT = "validation"
DEFAULT_AUDIO_COLUMN_NAME = "audio"
DEFAULT_TEXT_COLUMN_NAME = "text"
DEFAULT_EXAMPLE_INDEX = 0
DEFAULT_START_EXAMPLE_INDEX = 0
DEFAULT_MAX_EXAMPLE_COUNT = 1
DEFAULT_VAD_MODEL_PATH = os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx")
DEFAULT_MAX_OVERLAP_WORDS = 8
DEFAULT_OUTPUT_JSON_FILE_PATH = "evaluation/result/streaming_evaluation_last_run.json"


def load_evaluation_model():
    from src.backend_parakeet import load_model

    return load_model(PARAKEET_V2_MODEL_NAME)


def resolve_dataset_config_name_to_load(
    *,
    dataset_name: str,
    dataset_config_name: str | None,
) -> str | None:
    if dataset_config_name:
        return dataset_config_name
    if dataset_name == DEFAULT_DATASET_NAME:
        return "clean"
    return None


def load_one_dataset_item_audio_and_reference_text(
    *,
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_split: str,
    audio_column_name: str,
    text_column_name: str,
    example_index: int,
):
    selected_dataset_items = load_selected_dataset_items_audio_and_reference_text(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        dataset_split=dataset_split,
        audio_column_name=audio_column_name,
        text_column_name=text_column_name,
        first_sample_number_to_load=example_index,
        how_many_samples_to_load=1,
        use_streaming_dataset_load=False,
    )
    selected_dataset_item = selected_dataset_items[0]
    return selected_dataset_item["audio_array"], selected_dataset_item["reference_text"]


def load_selected_dataset_items_audio_and_reference_text(
    *,
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_split: str,
    audio_column_name: str,
    text_column_name: str,
    first_sample_number_to_load: int,
    how_many_samples_to_load: int,
    use_streaming_dataset_load: bool,
) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    resolved_dataset_config_name = resolve_dataset_config_name_to_load(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
    )
    dataset = load_dataset(
        dataset_name,
        resolved_dataset_config_name,
        split=dataset_split,
        streaming=use_streaming_dataset_load,
    )
    dataset = dataset.cast_column(
        audio_column_name,
        Audio(sampling_rate=DEFAULT_SAMPLE_RATE, decode=False),
    )

    selected_dataset_items: list[dict[str, Any]] = []

    if use_streaming_dataset_load:
        streamed_dataset_items = dataset.skip(first_sample_number_to_load).take(how_many_samples_to_load)
        for sample_offset, dataset_item in enumerate(streamed_dataset_items):
            audio_array = load_pcm16k_audio_array_from_audio_feature_value(dataset_item[audio_column_name])
            reference_text = str(dataset_item[text_column_name]).strip()
            selected_dataset_items.append(
                {
                    "example_index": first_sample_number_to_load + sample_offset,
                    "audio_array": audio_array,
                    "reference_text": reference_text,
                }
            )
        return selected_dataset_items

    for example_index in range(
        first_sample_number_to_load,
        first_sample_number_to_load + how_many_samples_to_load,
    ):
        dataset_item = dataset[example_index]
        audio_array = load_pcm16k_audio_array_from_audio_feature_value(dataset_item[audio_column_name])
        reference_text = str(dataset_item[text_column_name]).strip()
        selected_dataset_items.append(
            {
                "example_index": example_index,
                "audio_array": audio_array,
                "reference_text": reference_text,
            }
        )

    return selected_dataset_items


def load_pcm16k_audio_array_from_audio_feature_value(audio_feature_value: dict[str, Any]) -> np.ndarray:
    import soundfile

    audio_path = audio_feature_value.get("path")
    audio_bytes = audio_feature_value.get("bytes")

    if audio_bytes is not None:
        audio_array, sample_rate = soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
    elif audio_path:
        audio_array, sample_rate = soundfile.read(audio_path, dtype="float32")
    else:
        raise ValueError("Audio feature value does not contain a path or bytes payload.")

    audio_array = np.asarray(audio_array, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if sample_rate != DEFAULT_SAMPLE_RATE:
        audio_array = resample_audio_array_to_default_sample_rate(
            audio_array=audio_array,
            source_sample_rate=sample_rate,
        )

    return audio_array


def resample_audio_array_to_default_sample_rate(
    *,
    audio_array: np.ndarray,
    source_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == DEFAULT_SAMPLE_RATE:
        return audio_array.astype(np.float32)

    source_duration_seconds = len(audio_array) / float(source_sample_rate)
    target_sample_count = max(1, int(round(source_duration_seconds * DEFAULT_SAMPLE_RATE)))
    source_positions = np.linspace(0.0, 1.0, num=len(audio_array), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_sample_count, endpoint=False)
    resampled_audio_array = np.interp(target_positions, source_positions, audio_array)
    return np.asarray(resampled_audio_array, dtype=np.float32)


def convert_audio_array_to_pcm16_audio_bytes(audio_array: np.ndarray) -> bytes:
    pcm16_audio_array = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm16_audio_array.tobytes()


def split_audio_bytes_into_microphone_frames(
    pcm16_audio_bytes: bytes,
    *,
    frame_samples: int = DEFAULT_FRAME_SAMPLES,
) -> list[bytes]:
    frame_bytes = frame_samples * 2
    return [
        pcm16_audio_bytes[start_index:start_index + frame_bytes]
        for start_index in range(0, len(pcm16_audio_bytes), frame_bytes)
        if pcm16_audio_bytes[start_index:start_index + frame_bytes]
    ]


def add_previous_chunk_overlap_to_current_chunk_audio(
    previous_chunk_overlap_audio: bytes,
    current_chunk_audio: bytes,
    overlap_audio_bytes: int,
    *,
    stop_session: bool,
) -> tuple[bytes, bytes]:
    overlap_application_result = apply_previous_chunk_overlap(
        current_chunk_audio_bytes=current_chunk_audio,
        previous_pending_overlap_audio_bytes=previous_chunk_overlap_audio,
        overlap_audio_byte_count=overlap_audio_bytes,
        sample_rate=DEFAULT_SAMPLE_RATE,
        stop_session=stop_session,
    )
    return (
        overlap_application_result.overlapped_audio_bytes,
        overlap_application_result.next_pending_overlap_audio_bytes,
    )


def split_text_into_comparable_words(text: str) -> list[str]:
    return [word for word in text.strip().split() if word]


def remove_repeated_words_from_current_chunk_text(
    previous_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 8,
) -> str:
    return remove_duplicate_chunk_prefix(
        previous_chunk_text,
        current_chunk_text,
        max_overlap_words=max_overlap_words,
    )


def transcribe_one_audio_chunk(
    parakeet_v2_model,
    chunk_audio_bytes: bytes,
) -> str:
    from src.backend_parakeet import transcribe

    chunk_audio_array = np.frombuffer(chunk_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return transcribe(parakeet_v2_model, chunk_audio_array).strip()


def calculate_word_error_rate_for_final_streaming_text(
    *,
    reference_text: str,
    final_streaming_text: str,
) -> float:
    try:
        import jiwer
    except ModuleNotFoundError:
        reference_words = reference_text.split()
        hypothesis_words = final_streaming_text.split()

        if not reference_words:
            return 0.0 if not hypothesis_words else 1.0

        distances = [
            list(range(len(hypothesis_words) + 1))
        ]

        for reference_index, reference_word in enumerate(reference_words, start=1):
            current_row = [reference_index]
            for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
                substitution_cost = 0 if reference_word == hypothesis_word else 1
                current_row.append(
                    min(
                        distances[reference_index - 1][hypothesis_index] + 1,
                        current_row[hypothesis_index - 1] + 1,
                        distances[reference_index - 1][hypothesis_index - 1] + substitution_cost,
                    )
                )
            distances.append(current_row)

        return distances[-1][-1] / len(reference_words)

    return jiwer.wer(reference_text, final_streaming_text)


def build_chunk_event(
    *,
    chunk_index: int,
    split_reason: str,
    chunk_age_seconds_when_split_happened: float,
    silence_duration_seconds_when_split_happened: float | None,
    chunk_duration_seconds_before_overlap: float,
    overlap_seconds_added_from_previous_chunk: float,
    raw_chunk_text_without_overlap: str,
    raw_chunk_text_with_overlap: str,
    cleaned_chunk_text_after_dedup: str,
) -> dict[str, Any]:
    return {
        "chunk_index": chunk_index,
        "split_reason": split_reason,
        "chunk_age_seconds_when_split_happened": chunk_age_seconds_when_split_happened,
        "silence_duration_seconds_when_split_happened": silence_duration_seconds_when_split_happened,
        "chunk_duration_seconds_before_overlap": chunk_duration_seconds_before_overlap,
        "overlap_seconds_added_from_previous_chunk": overlap_seconds_added_from_previous_chunk,
        "raw_chunk_text_without_overlap": raw_chunk_text_without_overlap,
        "raw_chunk_text_with_overlap": raw_chunk_text_with_overlap,
        "cleaned_chunk_text_after_dedup": cleaned_chunk_text_after_dedup,
    }


def save_streaming_evaluation_run_to_json_file(
    *,
    output_json_file_path: str,
    streaming_evaluation_run_report: dict[str, Any],
) -> None:
    output_path = Path(output_json_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(streaming_evaluation_run_report, indent=2), encoding="utf-8")


def build_single_example_result_row(
    *,
    example_index: int,
    streaming_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "example_index": example_index,
        "chunk_count": int(streaming_result["chunk_count"]),
        "total_audio_seconds": round(sum(streaming_result["chunk_durations_seconds"]), 3),
        "final_word_error_rate": float(streaming_result["final_word_error_rate"]),
         "final_streaming_text": str(streaming_result["final_streaming_text"]),
        "reference_text": str(streaming_result["reference_text"]),
    }


def print_single_example_result_matrix(single_example_result_rows: list[dict[str, Any]]) -> None:
    log.info(
        "example_index | chunk_count | total_audio_seconds | final_word_error_rate | final_streaming_text",
    )
    log.info("-" * 96)
    for single_example_result_row in single_example_result_rows:
        final_streaming_text = single_example_result_row["final_streaming_text"]
        shortened_final_streaming_text = (
            final_streaming_text[:57] + "..." if len(final_streaming_text) > 60 else final_streaming_text
        )
        log.info(
            f"{single_example_result_row['example_index']:>13} | "
            f"{single_example_result_row['chunk_count']:>11} | "
            f"{single_example_result_row['total_audio_seconds']:>19.3f} | "
            f"{single_example_result_row['final_word_error_rate']:>21.4f} | "
            f"{shortened_final_streaming_text}",
        )


def print_multi_sample_summary(single_example_result_rows: list[dict[str, Any]]) -> None:
    if not single_example_result_rows:
        log.info("average_final_word_error_rate: 0.0")
        log.info("average_chunk_count: 0.0")
        log.info("average_total_audio_seconds: 0.0")
        return

    sample_count = len(single_example_result_rows)
    average_final_word_error_rate = sum(row["final_word_error_rate"] for row in single_example_result_rows) / sample_count
    average_chunk_count = sum(row["chunk_count"] for row in single_example_result_rows) / sample_count
    average_total_audio_seconds = sum(row["total_audio_seconds"] for row in single_example_result_rows) / sample_count

    log.info("Summary")
    log.info(f"sample_count: {sample_count}")
    log.info(f"average_final_word_error_rate: {average_final_word_error_rate:.4f}")
    log.info(f"average_chunk_count: {average_chunk_count:.2f}")
    log.info(f"average_total_audio_seconds: {average_total_audio_seconds:.3f}")


def run_fake_microphone_stream_for_one_dataset_item(
    *,
    pcm16_audio_bytes: bytes,
    reference_text: str,
    vad_model_path: str,
    silence_timeout_seconds: float,
    vad_threshold: float,
    energy_threshold: float,
    energy_ratio: float,
    overlap_seconds: float,
    max_overlap_words: int,
    frame_samples: int = DEFAULT_FRAME_SAMPLES,
    minimum_chunk_age_before_silence_split_seconds: float = DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
) -> dict[str, Any]:
    from src.vad_segmenter import SileroUtteranceGate, SileroVAD

    log.info(f"[Evaluation] Loading {PARAKEET_V2_MODEL_NAME} model...")
    parakeet_v2_model = load_evaluation_model()
    log.info("[Evaluation] Loading Silero VAD model...")
    vad_engine = SileroVAD(vad_model_path)
    utterance_gate = SileroUtteranceGate(
        vad_engine,
        frame_samples=frame_samples,
        voice_threshold=vad_threshold,
        silence_timeout_s=silence_timeout_seconds,
        energy_threshold=energy_threshold,
        energy_ratio=energy_ratio,
    )

    overlap_audio_bytes = int(DEFAULT_SAMPLE_RATE * 2 * overlap_seconds)
    previous_chunk_overlap_audio = b""
    previous_chunk_text = ""
    current_time_seconds = 0.0
    frame_duration_seconds = frame_samples / DEFAULT_SAMPLE_RATE

    chunk_texts: list[str] = []
    chunk_durations_seconds: list[float] = []
    chunk_events: list[dict[str, Any]] = []
    chunk_started_at_seconds: float | None = None
    chunk_index = 0

    for frame_bytes in split_audio_bytes_into_microphone_frames(
        pcm16_audio_bytes,
        frame_samples=frame_samples,
    ):
        utterance_gate.push(frame_bytes, now=current_time_seconds)
        if chunk_started_at_seconds is None:
            chunk_started_at_seconds = current_time_seconds

        silence_duration_seconds = utterance_gate.silence_elapsed(current_time_seconds)
        split_decision = should_split_chunk_after_silence(
            chunk_started_at_seconds=chunk_started_at_seconds,
            now_seconds=current_time_seconds,
            minimum_chunk_age_before_silence_split_seconds=minimum_chunk_age_before_silence_split_seconds,
            utterance_gate_should_finalize_now=utterance_gate.should_finalize(current_time_seconds),
            silence_duration_seconds=silence_duration_seconds,
        )
        if split_decision.should_split_now:
            raw_chunk_audio = utterance_gate.flush()
            if not raw_chunk_audio:
                current_time_seconds += frame_duration_seconds
                continue

            raw_chunk_text_without_overlap = transcribe_one_audio_chunk(
                parakeet_v2_model,
                raw_chunk_audio,
            )
            overlap_application_result = apply_previous_chunk_overlap(
                current_chunk_audio_bytes=raw_chunk_audio,
                previous_pending_overlap_audio_bytes=previous_chunk_overlap_audio,
                overlap_audio_byte_count=overlap_audio_bytes,
                sample_rate=DEFAULT_SAMPLE_RATE,
                stop_session=False,
            )
            previous_chunk_overlap_audio = overlap_application_result.next_pending_overlap_audio_bytes
            raw_chunk_text_with_overlap = transcribe_one_audio_chunk(
                parakeet_v2_model,
                overlap_application_result.overlapped_audio_bytes,
            )
            cleaned_chunk_text_after_dedup = remove_duplicate_chunk_prefix(
                previous_chunk_text,
                raw_chunk_text_with_overlap,
                max_overlap_words=max_overlap_words,
            )
            chunk_duration_seconds_before_overlap = len(raw_chunk_audio) / 2 / DEFAULT_SAMPLE_RATE
            chunk_events.append(
                build_chunk_event(
                    chunk_index=chunk_index,
                    split_reason="silence_threshold_hit",
                    chunk_age_seconds_when_split_happened=split_decision.chunk_age_seconds,
                    silence_duration_seconds_when_split_happened=split_decision.silence_duration_seconds,
                    chunk_duration_seconds_before_overlap=chunk_duration_seconds_before_overlap,
                    overlap_seconds_added_from_previous_chunk=(
                        overlap_application_result.overlap_seconds_added_from_previous_chunk
                    ),
                    raw_chunk_text_without_overlap=raw_chunk_text_without_overlap,
                    raw_chunk_text_with_overlap=raw_chunk_text_with_overlap,
                    cleaned_chunk_text_after_dedup=cleaned_chunk_text_after_dedup,
                )
            )

            if cleaned_chunk_text_after_dedup:
                chunk_texts.append(cleaned_chunk_text_after_dedup)
                previous_chunk_text = cleaned_chunk_text_after_dedup

            chunk_durations_seconds.append(chunk_duration_seconds_before_overlap)
            chunk_started_at_seconds = current_time_seconds
            chunk_index += 1

        current_time_seconds += frame_duration_seconds

    final_chunk_audio = utterance_gate.flush()
    if final_chunk_audio:
        raw_final_chunk_text_without_overlap = transcribe_one_audio_chunk(
            parakeet_v2_model,
            final_chunk_audio,
        )
        final_overlap_application_result = apply_previous_chunk_overlap(
            current_chunk_audio_bytes=final_chunk_audio,
            previous_pending_overlap_audio_bytes=previous_chunk_overlap_audio,
            overlap_audio_byte_count=overlap_audio_bytes,
            sample_rate=DEFAULT_SAMPLE_RATE,
            stop_session=True,
        )
        raw_final_chunk_text_with_overlap = transcribe_one_audio_chunk(
            parakeet_v2_model,
            final_overlap_application_result.overlapped_audio_bytes,
        )
        cleaned_final_chunk_text_after_dedup = remove_duplicate_chunk_prefix(
            previous_chunk_text,
            raw_final_chunk_text_with_overlap,
            max_overlap_words=max_overlap_words,
        )
        chunk_duration_seconds_before_overlap = len(final_chunk_audio) / 2 / DEFAULT_SAMPLE_RATE
        final_chunk_age_seconds = 0.0
        if chunk_started_at_seconds is not None:
            final_chunk_age_seconds = max(0.0, current_time_seconds - chunk_started_at_seconds)
        chunk_events.append(
            build_chunk_event(
                chunk_index=chunk_index,
                split_reason="final_flush_on_stop",
                chunk_age_seconds_when_split_happened=final_chunk_age_seconds,
                silence_duration_seconds_when_split_happened=None,
                chunk_duration_seconds_before_overlap=chunk_duration_seconds_before_overlap,
                overlap_seconds_added_from_previous_chunk=(
                    final_overlap_application_result.overlap_seconds_added_from_previous_chunk
                ),
                raw_chunk_text_without_overlap=raw_final_chunk_text_without_overlap,
                raw_chunk_text_with_overlap=raw_final_chunk_text_with_overlap,
                cleaned_chunk_text_after_dedup=cleaned_final_chunk_text_after_dedup,
            )
        )
        if cleaned_final_chunk_text_after_dedup:
            chunk_texts.append(cleaned_final_chunk_text_after_dedup)
            chunk_durations_seconds.append(chunk_duration_seconds_before_overlap)

    final_streaming_text = " ".join(chunk_texts).strip()
    normalized_reference_text = normalize_text_for_word_error_rate(reference_text)
    normalized_final_streaming_text = normalize_text_for_word_error_rate(final_streaming_text)
    final_word_error_rate = calculate_word_error_rate_for_final_streaming_text(
        reference_text=normalized_reference_text,
        final_streaming_text=normalized_final_streaming_text,
    )

    return {
        "reference_text": reference_text,
        "final_streaming_text": final_streaming_text,
        "final_word_error_rate": final_word_error_rate,
        "chunk_count": len(chunk_events),
        "chunk_texts": chunk_texts,
        "chunk_durations_seconds": chunk_durations_seconds,
        "chunk_events": chunk_events,
    }


def build_command_line_argument_parser() -> argparse.ArgumentParser:
    command_line_argument_parser = argparse.ArgumentParser(
        description="Run one standalone streaming evaluation example with Parakeet v2.",
    )
    command_line_argument_parser.add_argument("--dataset-name-to-load", default=DEFAULT_DATASET_NAME)
    command_line_argument_parser.add_argument("--dataset-config-name-to-load", default=DEFAULT_DATASET_CONFIG_NAME)
    command_line_argument_parser.add_argument("--dataset-split-to-load", default=DEFAULT_DATASET_SPLIT)
    command_line_argument_parser.add_argument("--audio-column-name-to-load", default=DEFAULT_AUDIO_COLUMN_NAME)
    command_line_argument_parser.add_argument("--text-column-name-to-load", default=DEFAULT_TEXT_COLUMN_NAME)
    command_line_argument_parser.add_argument("--single-sample-number-to-test", type=int, default=DEFAULT_EXAMPLE_INDEX)
    command_line_argument_parser.add_argument("--first-sample-number-to-test", type=int, default=DEFAULT_START_EXAMPLE_INDEX)
    command_line_argument_parser.add_argument("--how-many-samples-to-test", type=int, default=DEFAULT_MAX_EXAMPLE_COUNT)
    command_line_argument_parser.add_argument("--use-streaming-dataset-load", action="store_true")
    command_line_argument_parser.add_argument("--vad-model-file-path", default=DEFAULT_VAD_MODEL_PATH)
    command_line_argument_parser.add_argument(
        "--silence-timeout-seconds-before-cutting-chunk",
        type=float,
        default=DEFAULT_SILENCE_TIMEOUT_SECONDS,
    )
    command_line_argument_parser.add_argument("--voice-detection-score-threshold", type=float, default=DEFAULT_VAD_SCORE_THRESHOLD)
    command_line_argument_parser.add_argument("--energy-fallback-threshold", type=float, default=DEFAULT_VAD_ENERGY_THRESHOLD)
    command_line_argument_parser.add_argument("--energy-fallback-ratio", type=float, default=DEFAULT_ENERGY_RATIO)
    command_line_argument_parser.add_argument("--overlap-seconds-between-chunks", type=float, default=DEFAULT_OVERLAP_SECONDS)
    command_line_argument_parser.add_argument(
        "--minimum-chunk-age-before-silence-split-seconds",
        type=float,
        default=DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
    )
    command_line_argument_parser.add_argument("--maximum-overlap-words-to-remove", type=int, default=DEFAULT_MAX_OVERLAP_WORDS)
    command_line_argument_parser.add_argument("--microphone-frame-sample-count", type=int, default=DEFAULT_FRAME_SAMPLES)
    command_line_argument_parser.add_argument("--output-json-file-path", default=DEFAULT_OUTPUT_JSON_FILE_PATH)
    return command_line_argument_parser


def main(command_line_arguments: list[str] | None = None):
    parsed_command_line_arguments = build_command_line_argument_parser().parse_args(command_line_arguments)
    start_example_index = parsed_command_line_arguments.first_sample_number_to_test
    if parsed_command_line_arguments.how_many_samples_to_test == 1:
        start_example_index = parsed_command_line_arguments.single_sample_number_to_test

    single_example_result_rows: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []

    selected_dataset_items = load_selected_dataset_items_audio_and_reference_text(
        dataset_name=parsed_command_line_arguments.dataset_name_to_load,
        dataset_config_name=parsed_command_line_arguments.dataset_config_name_to_load,
        dataset_split=parsed_command_line_arguments.dataset_split_to_load,
        audio_column_name=parsed_command_line_arguments.audio_column_name_to_load,
        text_column_name=parsed_command_line_arguments.text_column_name_to_load,
        first_sample_number_to_load=start_example_index,
        how_many_samples_to_load=parsed_command_line_arguments.how_many_samples_to_test,
        use_streaming_dataset_load=parsed_command_line_arguments.use_streaming_dataset_load,
    )

    for selected_dataset_item in selected_dataset_items:
        example_index = selected_dataset_item["example_index"]
        log.info(f"[Evaluation] Loading dataset item {example_index}...")
        audio_array = selected_dataset_item["audio_array"]
        reference_text = selected_dataset_item["reference_text"]
        log.info("[Evaluation] Converting audio to PCM16 bytes...")
        pcm16_audio_bytes = convert_audio_array_to_pcm16_audio_bytes(audio_array)

        log.info("[Evaluation] Running fake microphone stream...")
        streaming_result = run_fake_microphone_stream_for_one_dataset_item(
            pcm16_audio_bytes=pcm16_audio_bytes,
            reference_text=reference_text,
            vad_model_path=parsed_command_line_arguments.vad_model_file_path,
            silence_timeout_seconds=parsed_command_line_arguments.silence_timeout_seconds_before_cutting_chunk,
            vad_threshold=parsed_command_line_arguments.voice_detection_score_threshold,
            energy_threshold=parsed_command_line_arguments.energy_fallback_threshold,
            energy_ratio=parsed_command_line_arguments.energy_fallback_ratio,
            overlap_seconds=parsed_command_line_arguments.overlap_seconds_between_chunks,
            max_overlap_words=parsed_command_line_arguments.maximum_overlap_words_to_remove,
            frame_samples=parsed_command_line_arguments.microphone_frame_sample_count,
            minimum_chunk_age_before_silence_split_seconds=(
                parsed_command_line_arguments.minimum_chunk_age_before_silence_split_seconds
            ),
        )

        log.info(f"[Evaluation] Final result for example {example_index}:")
        log.info(streaming_result)
        sample_results.append(
            {
                "example_index": example_index,
                **streaming_result,
            }
        )
        single_example_result_rows.append(
            build_single_example_result_row(
                example_index=example_index,
                streaming_result=streaming_result,
            )
        )

    print_single_example_result_matrix(single_example_result_rows)
    print_multi_sample_summary(single_example_result_rows)
    if single_example_result_rows:
        sample_count = len(single_example_result_rows)
        average_final_word_error_rate = (
            sum(row["final_word_error_rate"] for row in single_example_result_rows) / sample_count
        )
        average_chunk_count = sum(row["chunk_count"] for row in single_example_result_rows) / sample_count
        average_total_audio_seconds = sum(row["total_audio_seconds"] for row in single_example_result_rows) / sample_count
    else:
        sample_count = 0
        average_final_word_error_rate = 0.0
        average_chunk_count = 0.0
        average_total_audio_seconds = 0.0

    save_streaming_evaluation_run_to_json_file(
        output_json_file_path=parsed_command_line_arguments.output_json_file_path,
        streaming_evaluation_run_report={
            "run_summary": {
                "sample_count": sample_count,
                "average_final_word_error_rate": average_final_word_error_rate,
                "average_chunk_count": average_chunk_count,
                "average_total_audio_seconds": average_total_audio_seconds,
            },
            "sample_results": sample_results,
        },
    )


if __name__ == "__main__":
    main()
