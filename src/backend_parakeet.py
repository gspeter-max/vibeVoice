"""
Backend: NVIDIA Parakeet-TDT (via sherpa-onnx)
==============================================
Highly optimized TDT (Token-and-Duration Transducer) architecture.
Runs significantly faster than Whisper on CPU.
"""

from __future__ import annotations

import os
import numpy as np
from src import log

try:
    import sherpa_onnx
    _SHERPA_ONNX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on platform wheels
    sherpa_onnx = None  # type: ignore[assignment]
    _SHERPA_ONNX_IMPORT_ERROR = exc

# Default model on startup if this backend is forced
CURRENT_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"


def _get_model_details(model_name: str) -> tuple[str, str]:
    """
    Centralized logic for mapping model names to folder names and download URLs.
    This ensures that the loader and the downloader always look in the same place.
    """
    name = model_name.replace("nemo-", "")
    is_moonshine = "moonshine" in name

    # Folder naming convention used by k2-fsa/sherpa-onnx releases
    if is_moonshine:
        folder = f"sherpa-onnx-{name}-en-int8"
    elif "streaming" in name:
        folder = f"sherpa-onnx-nemo-{name}-int8"
    else:
        folder = f"sherpa-onnx-nemo-{name}-int8"

    base_path = os.path.expanduser("~/.cache/parakeet-flow/models")
    model_dir = os.path.join(base_path, folder)
    download_url = (
        f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{folder}.tar.bz2"
    )

    return model_dir, download_url


def _download_and_extract(url: str, dest_dir: str):
    """
    Downloads a .tar.bz2 model archive and extracts it to the cache directory.
    """
    import tarfile
    import urllib.request

    filename = url.split("/")[-1]
    archive_path = os.path.join(dest_dir, filename)

    os.makedirs(dest_dir, exist_ok=True)

    log.info(f"⬇️ Downloading {filename} (this may take a minute)...")
    try:
        urllib.request.urlretrieve(url, archive_path)

        log.info(f"📦 Extracting {filename}...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=dest_dir)

        log.info("✅ Done.")
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)


def load_model(model_name=None) -> sherpa_onnx.OfflineRecognizer:
    global CURRENT_MODEL_NAME

    if sherpa_onnx is None:
        raise RuntimeError(
            "sherpa-onnx is unavailable in this environment"
            + (f": {_SHERPA_ONNX_IMPORT_ERROR}" if _SHERPA_ONNX_IMPORT_ERROR else "")
        )

    target_name = model_name or CURRENT_MODEL_NAME
    model_dir, download_url = _get_model_details(target_name)
    CURRENT_MODEL_NAME = target_name.replace("nemo-", "")

    if not os.path.exists(model_dir):
        log.info(f"Model not found. Initiating auto-download for {CURRENT_MODEL_NAME}...")
        cache_base = os.path.dirname(model_dir)
        _download_and_extract(download_url, cache_base)

    log.info(f"\n[sherpa-onnx] Loading {CURRENT_MODEL_NAME} (INT8) from {model_dir}...")

    # Use PARAKEET_THREADS env var, or default to all cores
    thread_env = os.environ.get("PARAKEET_THREADS")
    num_threads = int(thread_env) if thread_env else 6

    log.info(f"[sherpa-onnx] Using {num_threads} threads")

    is_moonshine = "moonshine" in CURRENT_MODEL_NAME
    is_ctc = "ctc" in CURRENT_MODEL_NAME

    if is_moonshine:
        recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=f"{model_dir}/preprocess.onnx",
            encoder=f"{model_dir}/encode.int8.onnx",
            uncached_decoder=f"{model_dir}/uncached_decode.int8.onnx",
            cached_decoder=f"{model_dir}/cached_decode.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=num_threads,
            debug=False,
        )
    elif is_ctc:
        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=f"{model_dir}/model.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            debug=False,
        )
    else:
        # Use the from_transducer factory method which is available in the Python API
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            joiner=f"{model_dir}/joiner.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",  # CRITICAL for Parakeet-TDT
            debug=False,
        )

    log.info(f"[sherpa-onnx] ✅ Model loaded.")
    return recognizer
def transcribe(model: sherpa_onnx.OfflineRecognizer, audio_array: np.ndarray) -> str:
    """
    Transcribe a float32 numpy array (16kHz, mono, normalized -1..1).
    """
    stream = model.create_stream()
    stream.accept_waveform(16000, audio_array)
    model.decode_stream(stream)
    return stream.result.text.strip()
