#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import string
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jiwer
import numpy as np
from datasets import Audio, load_dataset
import soundfile as sf

import brain

warnings.filterwarnings("ignore")

DEFAULT_MODELS = [
    "parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2",
]
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_OVERLAP_SECONDS = 0.0
DEFAULT_MIN_DECODE_SECONDS = 10.0
DEFAULT_OVERLAP_VALUES = [round(i * 0.1, 2) for i in range(10)]
DEFAULT_DATASET_NAME = "librispeech_asr"
DEFAULT_DATASET_CONFIG = "clean"
DEFAULT_SPLIT = "test"

TEXT_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass
class Sample:
    audio: np.ndarray
    reference: str
    duration: float


@dataclass
class Result:
    model: str
    mode: str
    samples: int
    overlap_seconds: float
    min_decode_seconds: float
    wer: float
    rtf: float
    total_time: float
    draft_passes: float
    duplicate_pressure: float
    audio_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR models on LibriSpeech and streaming overlap settings.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names to benchmark.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of dataset samples to evaluate.")
    parser.add_argument("--mode", choices=("batch", "streaming", "compare", "sweep"), default="sweep", help="Evaluation mode.")
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS, help="Audio overlap retained between streaming steps.")
    parser.add_argument("--min-decode-seconds", type=float, default=DEFAULT_MIN_DECODE_SECONDS, help="Minimum new audio collected before each streaming decode.")
    parser.add_argument("--overlap-values", nargs="+", type=float, default=DEFAULT_OVERLAP_VALUES, help="Overlap values to sweep in streaming mode.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional local JSON or JSONL manifest with {'audio': path, 'text': transcript} entries.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--report-json", default=str(ROOT_DIR / "logs" / "evaluation_report.json"))
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    audio_array, samplerate = sf.read(path, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    if samplerate != 16000:
        target_length = int(round(len(audio_array) * 16000 / samplerate))
        if target_length <= 1 or len(audio_array) <= 1:
            audio_array = np.zeros(max(1, target_length), dtype=np.float32)
        else:
            x_old = np.linspace(0.0, 1.0, num=len(audio_array), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
            audio_array = np.interp(x_new, x_old, audio_array).astype(np.float32)
        samplerate = 16000
    return audio_array, samplerate


def _load_sample_from_record(record: dict, base_dir: Path | None = None) -> Sample:
    audio_value = record.get("audio")
    if audio_value is None:
        audio_value = record.get("path")

    if audio_value is None:
        raise ValueError("Manifest record must contain an 'audio' or 'path' field.")

    if isinstance(audio_value, dict):
        audio_value = audio_value.get("path") or audio_value.get("file") or audio_value.get("uri")
    if audio_value is None:
        raise ValueError("Manifest record audio field is empty.")

    audio_path = Path(audio_value)
    if not audio_path.is_absolute() and base_dir is not None:
        audio_path = base_dir / audio_path

    audio_array, samplerate = _load_audio_file(audio_path)
    reference_text = str(record.get("text", record.get("reference", ""))).strip()
    if not reference_text:
        raise ValueError(f"Manifest record {audio_path} is missing reference text.")

    return Sample(audio=audio_array, reference=reference_text, duration=len(audio_array) / float(samplerate))


def load_manifest_samples(manifest_path: str, sample_count: int) -> list[Sample]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise RuntimeError(f"Manifest file not found: {manifest}")

    print(f"Loading first {sample_count} samples from manifest {manifest}...")
    records: list[dict] = []
    raw_text = manifest.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise RuntimeError(f"Manifest file is empty: {manifest}")

    if raw_text.lstrip().startswith("["):
        records = json.loads(raw_text)
    else:
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    samples: list[Sample] = []
    for record in records:
        if len(samples) >= sample_count:
            break
        samples.append(_load_sample_from_record(record, manifest.parent))

    if not samples:
        raise RuntimeError(f"No samples were loaded from manifest: {manifest}")

    total_duration = sum(sample.duration for sample in samples)
    print(f"Loaded {len(samples)} samples. Total audio duration: {total_duration:.2f} seconds.")
    return samples


def load_samples(dataset_name: str, dataset_config: str, split: str, sample_count: int) -> list[Sample]:
    print(f"Loading first {sample_count} samples from {dataset_name}/{dataset_config} ({split})...")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        raise RuntimeError(f"Error initializing dataset: {exc}") from exc

    samples: list[Sample] = []
    print("Fetching and decoding samples manually...")
    try:
        for index, sample in enumerate(dataset):
            if index >= sample_count:
                break

            audio_data = sample["audio"]
            audio_bytes = None
            audio_path = None
            if isinstance(audio_data, dict):
                audio_bytes = audio_data.get("bytes")
                audio_path = audio_data.get("path")

            if audio_bytes:
                audio_source = io.BytesIO(audio_bytes)
            elif audio_path:
                audio_source = audio_path
            else:
                print(f"Warning: sample {index} did not provide audio bytes or path.")
                continue

            if isinstance(audio_source, io.BytesIO):
                audio_source.seek(0)
                audio_array, samplerate = sf.read(audio_source, dtype="float32")
            else:
                audio_array, samplerate = sf.read(audio_source, dtype="float32")

            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            if samplerate != 16000:
                target_length = int(round(len(audio_array) * 16000 / samplerate))
                if target_length <= 1 or len(audio_array) <= 1:
                    audio_array = np.zeros(max(1, target_length), dtype=np.float32)
                else:
                    x_old = np.linspace(0.0, 1.0, num=len(audio_array), endpoint=False)
                    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
                    audio_array = np.interp(x_new, x_old, audio_array).astype(np.float32)
                samplerate = 16000
            reference_text = sample["text"].lower()
            samples.append(
                Sample(
                    audio=audio_array,
                    reference=reference_text,
                    duration=len(audio_array) / float(samplerate),
                )
            )
    except Exception as exc:
        raise RuntimeError(f"Error during sample fetching: {exc}") from exc

    if not samples:
        raise RuntimeError("No samples were loaded. Check connection or dataset accessibility.")

    total_duration = sum(sample.duration for sample in samples)
    print(f"Loaded {len(samples)} samples. Total audio duration: {total_duration:.2f} seconds.")
    return samples


def load_backend_model(model_name: str):
    backend, model = brain.load_backend(model_name)
    return backend, model


def warmup_backend(backend, model, sample: Sample) -> None:
    try:
        backend.transcribe(model, sample.audio)
    except Exception:
        pass


def transcribe_audio(backend, model, audio: np.ndarray) -> str:
    return backend.transcribe(model, audio)


def streaming_transcribe(
    backend,
    model,
    sample: Sample,
    *,
    min_decode_seconds: float,
    overlap_seconds: float,
) -> tuple[str, int, float, float]:
    sr = 16000
    pcm = np.clip(sample.audio * 32768.0, -32768, 32767).astype(np.int16)
    step_samples = max(1, int(min_decode_seconds * sr))
    overlap_samples = max(0, int(overlap_seconds * sr))

    committed_text = ""
    previous_draft = ""
    tail_audio = np.array([], dtype=np.int16)
    draft_passes = 0
    duplicate_pressure = 0.0
    decode_time = 0.0

    for start in range(0, len(pcm), step_samples):
        chunk = pcm[start : start + step_samples]
        if len(chunk) == 0:
            continue

        window = np.concatenate([tail_audio, chunk]) if len(tail_audio) else chunk
        window_audio = window.astype(np.float32) / 32768.0

        t0 = time.perf_counter()
        draft_text = transcribe_audio(backend, model, window_audio)
        decode_time += time.perf_counter() - t0
        draft_passes += 1

        new_piece = brain.stitch_draft(previous_draft, draft_text)
        committed_text += new_piece

        previous_words = len(normalize_text(previous_draft).split())
        current_words = len(normalize_text(draft_text).split())
        new_words = len(normalize_text(new_piece).split())
        duplicate_pressure += max(0, previous_words + current_words - new_words)

        previous_draft = draft_text
        tail_audio = window[-overlap_samples:].copy() if overlap_samples > 0 else np.array([], dtype=np.int16)

    final_text = committed_text.strip()
    return final_text, draft_passes, duplicate_pressure, decode_time


def batch_transcribe(backend, model, sample: Sample) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = transcribe_audio(backend, model, sample.audio)
    return text, time.perf_counter() - t0


def evaluate_batch_mode(model_name: str, samples: list[Sample], backend, model) -> Result:
    predictions: list[str] = []
    references = [sample.reference for sample in samples]
    total_audio_duration = sum(sample.duration for sample in samples)
    total_inference_time = 0.0

    for index, sample in enumerate(samples, start=1):
        pred_text, elapsed = batch_transcribe(backend, model, sample)
        normalized_pred = normalize_text(pred_text)
        predictions.append(normalized_pred)
        total_inference_time += elapsed
        print(
            f"[{index:02d}] {elapsed:6.2f}s | drafts= 1 | "
            f"ref_words={len(normalize_text(sample.reference).split()):3d} | "
            f"hyp_words={len(normalize_text(normalized_pred).split()):3d}"
        )

    wer = jiwer.wer(references, predictions, reference_transform=TEXT_TRANSFORM, hypothesis_transform=TEXT_TRANSFORM)
    rtf = total_inference_time / total_audio_duration if total_audio_duration else 0.0
    print(f"Processed {len(samples)} samples.")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Real-time factor (RTF): {rtf:.3f}x (lower is faster)")
    print(f"Word error rate (WER): {wer * 100:.2f}% (lower is better)")

    return Result(
        model=model_name,
        mode="batch",
        samples=len(samples),
        overlap_seconds=0.0,
        min_decode_seconds=0.0,
        wer=wer * 100,
        rtf=rtf,
        total_time=total_inference_time,
        draft_passes=1.0,
        duplicate_pressure=0.0,
        audio_seconds=total_audio_duration,
    )


def evaluate_streaming_setting(
    model_name: str,
    samples: list[Sample],
    backend,
    model,
    *,
    min_decode_seconds: float,
    overlap_seconds: float,
) -> Result:
    print(f"\n--- Streaming: min_decode={min_decode_seconds:.2f}s overlap={overlap_seconds:.2f}s ---")
    predictions: list[str] = []
    references = [sample.reference for sample in samples]
    total_audio_duration = sum(sample.duration for sample in samples)
    total_inference_time = 0.0
    total_draft_passes = 0
    total_duplicate_pressure = 0.0

    for index, sample in enumerate(samples, start=1):
        pred_text, draft_passes, duplicate_pressure, elapsed = streaming_transcribe(
            backend,
            model,
            sample,
            min_decode_seconds=min_decode_seconds,
            overlap_seconds=overlap_seconds,
        )

        normalized_pred = normalize_text(pred_text)
        predictions.append(normalized_pred)
        total_inference_time += elapsed
        total_draft_passes += draft_passes
        total_duplicate_pressure += duplicate_pressure

        print(
            f"[{index:02d}] {elapsed:6.2f}s | drafts={draft_passes:2d} | "
            f"ref_words={len(normalize_text(sample.reference).split()):3d} | "
            f"hyp_words={len(normalize_text(normalized_pred).split()):3d}"
        )

    wer = jiwer.wer(references, predictions, reference_transform=TEXT_TRANSFORM, hypothesis_transform=TEXT_TRANSFORM)
    rtf = total_inference_time / total_audio_duration if total_audio_duration else 0.0
    dup_pressure = total_duplicate_pressure / max(1, len(samples))

    print(f"Processed {len(samples)} samples.")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Real-time factor (RTF): {rtf:.3f}x (lower is faster)")
    print(f"Word error rate (WER): {wer * 100:.2f}% (lower is better)")
    print(f"Avg draft passes/sample: {total_draft_passes / max(1, len(samples)):.2f}")
    print(f"Avg duplicate pressure/sample: {dup_pressure:.2f} words")

    return Result(
        model=model_name,
        mode="streaming",
        samples=len(samples),
        overlap_seconds=overlap_seconds,
        min_decode_seconds=min_decode_seconds,
        wer=wer * 100,
        rtf=rtf,
        total_time=total_inference_time,
        draft_passes=total_draft_passes / max(1, len(samples)),
        duplicate_pressure=dup_pressure,
        audio_seconds=total_audio_duration,
    )


def evaluate_model(model_name: str, samples: list[Sample], args: argparse.Namespace) -> list[Result]:
    print(f"\n{'=' * 72}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'=' * 72}")

    backend, model = load_backend_model(model_name)
    warmup_backend(backend, model, samples[0])

    results: list[Result] = []

    if args.mode == "batch":
        results.append(evaluate_batch_mode(model_name, samples, backend, model))
        return results

    if args.mode == "streaming":
        results.append(
            evaluate_streaming_setting(
                model_name,
                samples,
                backend,
                model,
                min_decode_seconds=args.min_decode_seconds,
                overlap_seconds=args.overlap_seconds,
            )
        )
        return results

    if args.mode == "compare":
        results.append(evaluate_batch_mode(model_name, samples, backend, model))
        results.append(
            evaluate_streaming_setting(
                model_name,
                samples,
                backend,
                model,
                min_decode_seconds=args.min_decode_seconds,
                overlap_seconds=args.overlap_seconds,
            )
        )
        return results

    for overlap_seconds in args.overlap_values:
        results.append(
            evaluate_streaming_setting(
                model_name,
                samples,
                backend,
                model,
                min_decode_seconds=args.min_decode_seconds,
                overlap_seconds=float(overlap_seconds),
            )
        )

    return results


def print_summary(results: list[Result]) -> None:
    if not results:
        print("No models were successfully evaluated.")
        raise SystemExit(1)

    print("\n\n" + "╔" + "═" * 106 + "╗")
    print("║" + " " * 38 + "FINAL BENCHMARK SUMMARY" + " " * 39 + "║")
    print(
        "╠"
        + "═" * 18
        + "╦"
        + "═" * 11
        + "╦"
        + "═" * 11
        + "╦"
        + "═" * 12
        + "╦"
        + "═" * 13
        + "╦"
        + "═" * 12
        + "╦"
        + "═" * 14
        + "╦"
        + "═" * 14
        + "╣"
    )
    print(
        f"║ {'Model':<16} ║ {'Mode':<9} ║ {'Overlap':<9} ║ {'Min Decode':<10} ║ "
        f"{'WER (%)':<11} ║ {'RTF':<10} ║ {'Drafts/Sample':<12} ║ {'Dup Press.':<12} ║"
    )
    print(
        "╠"
        + "═" * 18
        + "╬"
        + "═" * 11
        + "╬"
        + "═" * 11
        + "╬"
        + "═" * 12
        + "╬"
        + "═" * 13
        + "╬"
        + "═" * 12
        + "╬"
        + "═" * 14
        + "╬"
        + "═" * 14
        + "╣"
    )
    for result in results:
        print(
            f"║ {result.model[:16]:<16} ║ {result.mode:<9} ║ {result.overlap_seconds:<9.2f} ║ {result.min_decode_seconds:<10.2f} ║ "
            f"{result.wer:<11.2f} ║ {result.rtf:<10.3f} ║ {result.draft_passes:<12.2f} ║ {result.duplicate_pressure:<12.2f} ║"
        )
    print(
        "╚"
        + "═" * 18
        + "╩"
        + "═" * 11
        + "╩"
        + "═" * 11
        + "╩"
        + "═" * 12
        + "╩"
        + "═" * 13
        + "╩"
        + "═" * 12
        + "╩"
        + "═" * 14
        + "╩"
        + "═" * 14
        + "╝"
    )
    print("* WER: Word Error Rate (lower is better)")
    print("* RTF: Real-Time Factor. E.g., 0.1 means 10s of audio takes 1s to transcribe.")
    print("* Drafts/Sample: average number of overlap decode passes per sample.")
    print("* Dup Press.: approximate repeated-word pressure removed by stitching.")


def write_report_json(path: str, results: list[Result]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    print(f"\nWrote JSON report to {report_path}")


def main() -> None:
    args = parse_args()
    if args.manifest:
        samples = load_manifest_samples(args.manifest, args.samples)
    else:
        samples = load_samples(args.dataset_name, args.dataset_config, args.split, args.samples)

    all_results: list[Result] = []
    for model_name in args.models:
        try:
            all_results.extend(evaluate_model(model_name, samples, args))
        except Exception as exc:
            print(f"Error evaluating {model_name}: {exc}")

    print_summary(all_results)
    write_report_json(args.report_json, all_results)


if __name__ == "__main__":
    main()
