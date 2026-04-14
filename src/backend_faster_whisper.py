"""
Backend: faster-whisper + distil-large-v3 (INT8, CPU)
=====================================================
Optimized for Intel i7-9750H (6 Cores)
--------------------------------------
Uses CTranslate2 with:
  - Intel MKL for optimized matrix math
  - INT8 quantization (AVX2/VNNI support)
  - Physical core affinity (6 threads) to avoid hyperthreading lag
"""

import os
import platform
import numpy as np
from src import log
try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - test environments may not have the package
    class WhisperModel:  # type: ignore[override]
        pass

# Default model on startup
CURRENT_MODEL_NAME = "deepdml/faster-whisper-large-v3-turbo-ct2"

if platform.machine() == "arm64":
    COMPUTE_TYPE = "default"
    CPU_THREADS = 4
else:
    COMPUTE_TYPE = "int8"
    # OPTIMIZATION: On Intel, using physical cores only is 20-30% faster for AI
    # i7-9750H has 6 physical cores.
    CPU_THREADS = 6

def load_model(model_name=None) -> WhisperModel:
    global CURRENT_MODEL_NAME
    
    if model_name:
        CURRENT_MODEL_NAME = model_name
        
    log.info(f"\n[faster-whisper] 🚀 Optimizing for Intel i7 (Threads: {CPU_THREADS}, Mode: {COMPUTE_TYPE})")
    log.info(f"[faster-whisper] Loading {CURRENT_MODEL_NAME}...")

    model = WhisperModel(
        CURRENT_MODEL_NAME,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=1,
        download_root=os.path.expanduser("~/.cache/parakeet-flow/models"),
    )

    log.info(f"[faster-whisper] ✅ Model loaded.")
    return model
def transcribe(model: WhisperModel, audio_array: np.ndarray) -> str:
    """
    Transcribe a float32 numpy array (16kHz, mono, normalized -1..1).
    """
    segments, info = model.transcribe(
        audio_array,
        language="en",
        beam_size=1, # OPTIMIZATION: Greedy decoding (1) is much faster than beam search (5)
        best_of=1,   # OPTIMIZATION: Greedy decoding (1) is much faster than best_of (5)
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            threshold=0.5,
        ),
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    words = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            words.append(text)

    return " ".join(words).strip()
