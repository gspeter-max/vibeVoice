"""
Backend: faster-whisper + distil-large-v3 (INT8, CPU)
=====================================================
Uses CTranslate2 under the hood which automatically applies:
  - Intel oneMKL for matrix ops on Intel CPUs
  - Runtime AVX/AVX2/AVX512 kernel dispatch
  - INT8 quantization (halves memory bandwidth, 3-6x faster than PyTorch)

Model: distil-large-v3  — 6x faster than large-v3, <1% WER increase, English only
Compute: int8           — fastest on Intel CPU
"""

import os
import platform
import multiprocessing
import numpy as np
from faster_whisper import WhisperModel

# Default model on startup
CURRENT_MODEL_NAME = "base.en"
_model_instance = None

if platform.machine() == "arm64":
    COMPUTE_TYPE = "default"
    CPU_THREADS = 4
else:
    COMPUTE_TYPE = "int8"
    CPU_THREADS = multiprocessing.cpu_count()

def load_model(model_name=None) -> WhisperModel:
    global _model_instance, CURRENT_MODEL_NAME
    
    if model_name:
        CURRENT_MODEL_NAME = model_name
        
    print(f"\\n[faster-whisper] Loading {CURRENT_MODEL_NAME} ({COMPUTE_TYPE}) on CPU with {CPU_THREADS} threads...", flush=True)

    _model_instance = WhisperModel(
        CURRENT_MODEL_NAME,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=1,
        download_root=os.path.expanduser("~/.cache/parakeet-flow/models"),
    )

    print(f"[faster-whisper] ✅ Model loaded.", flush=True)
    return _model_instance

def get_current_model():
    return _model_instance

def transcribe(model: WhisperModel, audio_array: np.ndarray) -> str:
    """
    Transcribe a float32 numpy array (16kHz, mono, normalized -1..1).

    Returns the transcribed text string, or empty string if nothing detected.
    """
    segments, info = model.transcribe(
        audio_array,
        language="en",           # English only — removes language detection overhead
        beam_size=5,             # good balance of speed vs accuracy
        best_of=5,
        vad_filter=True,         # skip silent segments early — big speed win for short clips
        vad_parameters=dict(
            min_silence_duration_ms=300,
            threshold=0.5,
        ),
        condition_on_previous_text=False,  # stateless — better for short clips
        word_timestamps=False,
    )

    words = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            words.append(text)

    return " ".join(words).strip()
