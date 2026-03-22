"""
Backend: NVIDIA Parakeet-TDT (via sherpa-onnx)
==============================================
Highly optimized TDT (Token-and-Duration Transducer) architecture.
Runs significantly faster than Whisper on CPU.
"""

import os
import multiprocessing
import numpy as np
import sherpa_onnx

# Default model on startup if this backend is forced
CURRENT_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
_model_instance = None

def load_model(model_name=None) -> sherpa_onnx.OfflineRecognizer:
    global _model_instance, CURRENT_MODEL_NAME
    
    if model_name:
        # Strip 'nemo-' prefix if user passed it, as we add it in the path
        CURRENT_MODEL_NAME = model_name.replace("nemo-", "")
        
    model_dir = os.path.expanduser(f"~/.cache/parakeet-flow/models/sherpa-onnx-nemo-{CURRENT_MODEL_NAME}-int8")
    
    if not os.path.exists(model_dir):
         raise RuntimeError(f"Model directory not found: {model_dir}")

    print(f"\n[sherpa-onnx] Loading {CURRENT_MODEL_NAME} (INT8) from {model_dir}...", flush=True)

    # Use all available cores for max performance on Intel i7
    num_threads = multiprocessing.cpu_count()

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
        model_type="nemo_transducer", # CRITICAL for Parakeet-TDT
        debug=False
    )

    _model_instance = recognizer
    print(f"[sherpa-onnx] ✅ Model loaded.", flush=True)
    return _model_instance

def get_current_model():
    return _model_instance

def transcribe(model: sherpa_onnx.OfflineRecognizer, audio_array: np.ndarray) -> str:
    """
    Transcribe a float32 numpy array (16kHz, mono, normalized -1..1).
    """
    stream = model.create_stream()
    stream.accept_waveform(16000, audio_array)
    model.decode_stream(stream)
    return stream.result.text.strip()
