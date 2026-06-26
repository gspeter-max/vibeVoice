"""
Backend: NVIDIA Parakeet-TDT (via sherpa-onnx)
==============================================
Highly optimized TDT (Token-and-Duration Transducer) architecture.
Runs significantly faster than Whisper on CPU.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from typing import Any

import numpy as np

from src import log
from src.utils.env_utils import get_integer_from_environment
from src.utils.settings import settings

try:
    import sherpa_onnx

    _SHERPA_ONNX_IMPORT_ERROR = None
except (ImportError, OSError) as exc:  # pragma: no cover - depends on platform wheels
    _SHERPA_ONNX_IMPORT_ERROR = exc


def _get_model_paths(model_name: str) -> tuple[str, str]:
    """Return the local model folder and the download URL for one model name."""
    is_moonshine = "moonshine" in model_name

    # We make the folder name based on if it is a moonshine model or a normal nemo model
    if is_moonshine:
        folder_name = f"sherpa-onnx-{model_name}-en-int8"
    else:
        folder_name = f"sherpa-onnx-nemo-{model_name}-int8"

    base_cache = os.path.expanduser("~/.cache/parakeet-flow/models")
    cache_path = os.path.join(base_cache, folder_name)

    download_url = (
        f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{folder_name}.tar.bz2"
    )

    return cache_path, download_url


def _download_and_extract(url: str, dest_dir: str) -> None:
    """
    Downloads the compressed model file from the internet and extracts it.
    """
    filename = url.split("/")[-1]
    archive_path = os.path.join(dest_dir, filename)

    # Make the folder if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    log.info(f"⬇️ Downloading {filename} (this may take a minute)...")
    try:
        # Download the file
        urllib.request.urlretrieve(url, archive_path)

        log.info(f"📦 Extracting {filename}...")
        # Extract the file
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=dest_dir)

        log.info("✅ Done.")
    finally:
        # Delete the compressed file
        if os.path.exists(archive_path):
            os.remove(archive_path)


def load_model(model_name: str | None = None) -> sherpa_onnx.OfflineRecognizer | Any:
    """Load the requested sherpa-onnx model into memory."""

    # Check if the sherpa_onnx library is installed
    if isinstance(_SHERPA_ONNX_IMPORT_ERROR, BaseException):
        raise RuntimeError(
            "sherpa-onnx is unavailable in this environment"
        ) from _SHERPA_ONNX_IMPORT_ERROR

    # Get the exact model name
    name = model_name or settings.stt_model

    # Get the folder path and download link for the model
    cache_path, download_url = _get_model_paths(name)
    clean_name = name.replace("nemo-", "")

    # If the model is not on your computer, download and unzip it
    if not os.path.exists(cache_path):
        log.info(f"Model not found. Initiating auto-download for {clean_name}...")
        parent_dir = os.path.dirname(cache_path)
        _download_and_extract(download_url, parent_dir)

    log.info(f"\n[sherpa-onnx] Loading {clean_name} (INT8) from {cache_path}...")

    # Find out how many CPU threads to use for speed
    threads = get_integer_from_environment("PARAKEET_THREADS", 6)
    log.info(f"[sherpa-onnx] Using {threads} threads")

    is_moonshine = "moonshine" in clean_name
    is_ctc = "ctc" in clean_name

    # Load the model into memory based on what type of model it is
    if is_moonshine:
        recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=f"{cache_path}/preprocess.onnx",
            encoder=f"{cache_path}/encode.int8.onnx",
            uncached_decoder=f"{cache_path}/uncached_decode.int8.onnx",
            cached_decoder=f"{cache_path}/cached_decode.int8.onnx",
            tokens=f"{cache_path}/tokens.txt",
            num_threads=threads,
            debug=False,
        )
    elif is_ctc:
        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=f"{cache_path}/model.int8.onnx",
            tokens=f"{cache_path}/tokens.txt",
            num_threads=threads,
            sample_rate=settings.rate,
            feature_dim=80,
            debug=False,
        )
    else:
        # Use the from_transducer factory method which is available in the Python API
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{cache_path}/encoder.int8.onnx",
            decoder=f"{cache_path}/decoder.int8.onnx",
            joiner=f"{cache_path}/joiner.int8.onnx",
            tokens=f"{cache_path}/tokens.txt",
            num_threads=threads,
            sample_rate=settings.rate,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",  # CRITICAL for Parakeet-TDT
            debug=False,
        )

    log.info("[sherpa-onnx] ✅ Model loaded.")
    return recognizer


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer | Any, audio: np.ndarray) -> str:
    """Run one audio array through the loaded recognizer and return text."""
    stream = recognizer.create_stream()
    stream.accept_waveform(settings.rate, audio)
    recognizer.decode_stream(stream)

    text = stream.result.text.strip()
    return text
