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
from src.utils.env_utils import get_integer_from_environment

try:
    import sherpa_onnx
    _SHERPA_ONNX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on platform wheels
    sherpa_onnx = None  # type: ignore[assignment]
    _SHERPA_ONNX_IMPORT_ERROR = exc

# Default model on startup if this backend is forced
CURRENT_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"


def get_model_folder_path_and_download_link(model_name_to_check: str) -> tuple[str, str]:
    """
    This function takes the name of the speech recognition model and gives you two things:
    1. The folder path on your computer where the model should be saved.
    2. The internet link to download the model if you do not have it.
    
    It removes the "nemo-" text from the name to make the folder name match what is on the internet.
    """
    clean_model_name = model_name_to_check.replace("nemo-", "")
    is_moonshine_model = "moonshine" in clean_model_name

    # We make the folder name based on if it is a moonshine model or a normal nemo model
    if is_moonshine_model:
        folder_name_on_computer = f"sherpa-onnx-{clean_model_name}-en-int8"
    else:
        folder_name_on_computer = f"sherpa-onnx-nemo-{clean_model_name}-int8"

    base_cache_folder_path = os.path.expanduser("~/.cache/parakeet-flow/models")
    full_model_folder_path = os.path.join(base_cache_folder_path, folder_name_on_computer)
    
    internet_download_link = (
        f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{folder_name_on_computer}.tar.bz2"
    )

    return full_model_folder_path, internet_download_link


def download_model_file_and_unzip_it(internet_download_link: str, destination_folder_path: str):
    """
    This function downloads the compressed model file from the internet and unzips it into a folder.
    
    Steps:
    1. Find the file name from the internet link.
    2. Make the folder on your computer if it does not exist.
    3. Download the file from the internet to your computer.
    4. Unzip the file so the computer can use it.
    5. Delete the compressed file to save space on your computer.
    """
    import tarfile
    import urllib.request

    compressed_file_name = internet_download_link.split("/")[-1]
    full_path_to_compressed_file = os.path.join(destination_folder_path, compressed_file_name)

    # Make the folder if it does not exist
    os.makedirs(destination_folder_path, exist_ok=True)

    log.info(f"⬇️ Downloading {compressed_file_name} (this may take a minute)...")
    try:
        # Download the file
        urllib.request.urlretrieve(internet_download_link, full_path_to_compressed_file)

        log.info(f"📦 Extracting {compressed_file_name}...")
        # Unzip the file
        with tarfile.open(full_path_to_compressed_file, "r:bz2") as tar_file_object:
            tar_file_object.extractall(path=destination_folder_path)

        log.info("✅ Done.")
    finally:
        # Delete the compressed file
        if os.path.exists(full_path_to_compressed_file):
            os.remove(full_path_to_compressed_file)


def load_speech_recognition_model_from_disk(requested_model_name=None) -> sherpa_onnx.OfflineRecognizer:
    """
    This function loads the speech recognition model from your computer into memory so it can be used to convert audio to text.
    
    Steps:
    1. Check if the sherpa_onnx library is installed.
    2. Get the exact model name.
    3. Get the folder path and download link for the model.
    4. If the model is not on your computer, download and unzip it.
    5. Find out how many CPU threads to use for speed.
    6. Load the model into memory based on what type of model it is (Moonshine, CTC, or Transducer).
    7. Return the loaded model.
    """
    global CURRENT_MODEL_NAME

    # Step 1: Check if the sherpa_onnx library is installed
    if sherpa_onnx is None:
        error_message_to_show = "sherpa-onnx is unavailable in this environment"
        if _SHERPA_ONNX_IMPORT_ERROR:
            error_message_to_show += f": {_SHERPA_ONNX_IMPORT_ERROR}"
        raise RuntimeError(error_message_to_show)

    # Step 2: Get the exact model name
    target_model_name = requested_model_name or CURRENT_MODEL_NAME
    
    # Step 3: Get the folder path and download link for the model
    model_folder_path, internet_download_link = get_model_folder_path_and_download_link(target_model_name)
    CURRENT_MODEL_NAME = target_model_name.replace("nemo-", "")

    # Step 4: If the model is not on your computer, download and unzip it
    if not os.path.exists(model_folder_path):
        log.info(f"Model not found. Initiating auto-download for {CURRENT_MODEL_NAME}...")
        cache_base_folder = os.path.dirname(model_folder_path)
        download_model_file_and_unzip_it(internet_download_link, cache_base_folder)

    log.info(f"\n[sherpa-onnx] Loading {CURRENT_MODEL_NAME} (INT8) from {model_folder_path}...")

    # Step 5: Find out how many CPU threads to use for speed
    # Use PARAKEET_THREADS env var, or default to 6 if missing or malformed
    number_of_cpu_threads_to_use = get_integer_from_environment("PARAKEET_THREADS", 6)

    log.info(f"[sherpa-onnx] Using {number_of_cpu_threads_to_use} threads")

    is_moonshine_model = "moonshine" in CURRENT_MODEL_NAME
    is_ctc_model = "ctc" in CURRENT_MODEL_NAME

    # Step 6: Load the model into memory based on what type of model it is
    if is_moonshine_model:
        loaded_speech_recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=f"{model_folder_path}/preprocess.onnx",
            encoder=f"{model_folder_path}/encode.int8.onnx",
            uncached_decoder=f"{model_folder_path}/uncached_decode.int8.onnx",
            cached_decoder=f"{model_folder_path}/cached_decode.int8.onnx",
            tokens=f"{model_folder_path}/tokens.txt",
            num_threads=number_of_cpu_threads_to_use,
            debug=False,
        )
    elif is_ctc_model:
        loaded_speech_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=f"{model_folder_path}/model.int8.onnx",
            tokens=f"{model_folder_path}/tokens.txt",
            num_threads=number_of_cpu_threads_to_use,
            sample_rate=16000,
            feature_dim=80,
            debug=False,
        )
    else:
        # Use the from_transducer factory method which is available in the Python API
        loaded_speech_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_folder_path}/encoder.int8.onnx",
            decoder=f"{model_folder_path}/decoder.int8.onnx",
            joiner=f"{model_folder_path}/joiner.int8.onnx",
            tokens=f"{model_folder_path}/tokens.txt",
            num_threads=number_of_cpu_threads_to_use,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",  # CRITICAL for Parakeet-TDT
            debug=False,
        )

    log.info(f"[sherpa-onnx] ✅ Model loaded.")
    
    # Step 7: Return the loaded model
    return loaded_speech_recognizer

def convert_audio_to_text(loaded_speech_recognizer_model: sherpa_onnx.OfflineRecognizer, audio_data_array: np.ndarray) -> str:
    """
    This function takes the audio data and uses the loaded model to figure out what words were spoken.
    It returns the spoken words as a text string.
    
    Input audio must be a float32 numpy array (16kHz, mono, normalized -1 to 1).
    """
    audio_stream = loaded_speech_recognizer_model.create_stream()
    audio_stream.accept_waveform(16000, audio_data_array)
    loaded_speech_recognizer_model.decode_stream(audio_stream)
    
    final_text_string = audio_stream.result.text.strip()
    return final_text_string
