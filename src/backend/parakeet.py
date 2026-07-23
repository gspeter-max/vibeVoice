"""
Backend: NVIDIA Parakeet-TDT (via sherpa-onnx)
==============================================
Highly optimized TDT (Token-and-Duration Transducer) architecture.
Runs significantly faster than Whisper on CPU.
"""

from __future__ import annotations

import enum
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src import log
from src.utils.env_utils import get_integer_from_environment
from src.utils.settings import settings

try:
    import sherpa_onnx
except (ModuleNotFoundError, ImportError, OSError) as exc:
    import sys

    print(sys.platform)
    if sys.platform == "darwin":
        try:
            import ctypes
            import os

            import onnxruntime

            capi_dir = os.path.dirname(onnxruntime.capi.__file__)
            for f in os.listdir(capi_dir):
                if f.startswith("libonnxruntime") and f.endswith(".dylib"):
                    ctypes.CDLL(os.path.join(capi_dir, f), mode=ctypes.RTLD_GLOBAL)
                    break
            import sherpa_onnx
        except Exception:
            raise exc
    else:
        raise exc


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration representing a model's local destination directory and its
    associated remote archive package file URL.
    """

    download_url: str
    folder_name: str


class ModelDownloadConfig:
    """
    Registry utility to resolve remote and local filesystem configurations
    for supported Speech-to-Text (STT) models.
    """

    # Base URL hosting sherpa-onnx precompiled neural networks
    BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"

    @classmethod
    def create(cls, folder_name: str) -> ModelConfig:
        """
        Creates a ModelConfig pointing to a specific tar.bz2 archive under BASE_URL.
        """
        return ModelConfig(
            download_url=f"{cls.BASE_URL}{folder_name}.tar.bz2", folder_name=folder_name
        )

    @classmethod
    def for_moonshine(cls, model_name: str) -> ModelConfig:
        """
        Generates configuration for Moonshine models (e.g., moonshine-tiny, moonshine-base).
        """
        return cls.create(f"sherpa-onnx-{model_name}-en-int8")

    @classmethod
    def for_sense_voice(cls) -> ModelConfig:
        """
        Generates configuration for the multilingual SenseVoice model.
        """
        return cls.create("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")

    @classmethod
    def for_nemo(cls, model_name: str) -> ModelConfig:
        """
        Generates configuration for standard NeMo conformer/transducer models.
        """
        return cls.create(f"sherpa-onnx-nemo-{model_name}-int8")


def _get_model_paths(model_name: str) -> tuple[str, str]:
    """Return the local model folder and the download URL for one model name."""
    clean_name = model_name.lower().replace("-", "_")

    if "moonshine" in clean_name:
        config = ModelDownloadConfig.for_moonshine(model_name)
    elif "sensevoice" in clean_name or "sense_voice" in clean_name:
        config = ModelDownloadConfig.for_sense_voice()
    else:
        config = ModelDownloadConfig.for_nemo(model_name)

    base_cache = os.path.expanduser("~/.cache/parakeet-flow/models")
    cache_path = os.path.join(base_cache, config.folder_name)
    return cache_path, config.download_url


def _download_and_extract(url: str, dest_dir: str) -> None:
    """
    Downloads the compressed model file from the internet and extracts it.
    """
    filename = url.split("/")[-1]
    archive_path = os.path.join(dest_dir, filename)

    # Make the folder if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    log.debug(f"⬇️ Downloading {filename} (this may take a minute)...")
    try:
        # Download the file
        urllib.request.urlretrieve(url, archive_path)

        log.debug(f"📦 Extracting {filename}...")
        # Extract the file
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=dest_dir)

        log.debug("✅ Done.")
    finally:
        # Delete the compressed file
        if os.path.exists(archive_path):
            os.remove(archive_path)


class ModelType(Enum):
    MOONSHINE = enum.auto()
    CTC = enum.auto()
    SENSE_VOICE = enum.auto()
    TRANSOUCER = enum.auto()


def get_model_type(model_name):
    clean_name = model_name.lower().replace("-", "").replace("_", "")
    if "moonshine" in clean_name:
        return ModelType.MOONSHINE
    elif "ctc" in clean_name:
        return ModelType.CTC
    elif "sensevoice" in clean_name:
        return ModelType.SENSE_VOICE
    else:
        return ModelType.TRANSOUCER


def load_model(model_name: str | None = None) -> sherpa_onnx.OfflineRecognizer | Any:
    # Get the exact model name
    name = model_name or settings.stt_model

    # Get the folder path and download link for the model
    cache_path, download_url = _get_model_paths(name)
    model_type = get_model_type(name)
    # If the model is not on your computer, download and unzip it
    if not os.path.exists(cache_path):
        log.debug(f"Model not found. Initiating auto-download for {model_type}...")
        parent_dir = os.path.dirname(cache_path)
        _download_and_extract(download_url, parent_dir)

    log.debug(f"\n[sherpa-onnx] Loading {model_type} (INT8) from {cache_path}...")

    # Fine out how many CPU threads to use for speed
    threads = get_integer_from_environment("PARAKEET_THREADS", 6)
    log.debug(f"[sherpa-onnx] Using {threads} threads")

    if model_type == ModelType.MOONSHINE:
        recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=f"{cache_path}/preprocess.onnx",
            encoder=f"{cache_path}/encode.int8.onnx",
            uncached_decoder=f"{cache_path}/uncached_decode.int8.onnx",
            cached_decoder=f"{cache_path}/cached_decode.int8.onnx",
            tokens=f"{cache_path}/tokens.txt",
            num_threads=threads,
            debug=False,
        )

    elif model_type == ModelType.SENSE_VOICE:
        log.debug("[sherpa-onnx] SenseVoice model configuration selected")
        # Note: We disable Inverse Text Normalization (use_itn=False) because it utilizes
        # expensive rule FST matching on CPU. Since the output text goes directly through
        # a downstream LLM refiner (llm_refine), formatting/punctuation can be handled
        # there much faster and more reliably.
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{cache_path}/model.int8.onnx",
            tokens=f"{cache_path}/tokens.txt",
            use_itn=True,
            language="en",
        )
    elif model_type == ModelType.CTC:
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

    log.debug("[sherpa-onnx] ✅ Model loaded.")
    return recognizer


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer | Any, audio: np.ndarray) -> str:
    """Run one audio array through the loaded recognizer and return text."""
    stream = recognizer.create_stream()
    stream.accept_waveform(settings.rate, audio)
    recognizer.decode_stream(stream)

    text = stream.result.text.strip()
    return text
