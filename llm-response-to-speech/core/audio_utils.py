"""
Audio Processing Utilities
===========================
Utilities for converting between numpy arrays and WAV format.
"""
import io
import wave
import numpy as np
from typing import Tuple


def numpy_to_wav(
    audio: np.ndarray,
    sample_rate: int = 22050,
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """
    Convert numpy array to WAV bytes.

    Args:
        audio: Audio data as numpy array (float32, -1.0 to 1.0)
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample (2 = 16-bit)

    Returns:
        WAV file as bytes
    """
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV in memory
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return wav_buffer.getvalue()


def wav_to_numpy(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Convert WAV bytes to numpy array.

    Args:
        wav_bytes: WAV file as bytes

    Returns:
        Tuple of (audio array, sample_rate)
    """
    with io.BytesIO(wav_bytes) as wav_buffer:
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

            # Convert to numpy
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32767.0

            return audio_float32, sample_rate


def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalize audio to target RMS level.

    Args:
        audio: Audio data as numpy array
        target_rms: Target RMS amplitude

    Returns:
        Normalized audio
    """
    current_rms = calculate_rms(audio)
    if current_rms > 0:
        scaling_factor = target_rms / current_rms
        return audio * scaling_factor
    return audio


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS amplitude of audio.

    Args:
        audio: Audio data as numpy array

    Returns:
        RMS amplitude
    """
    return float(np.sqrt(np.mean(audio ** 2)))


def get_wav_info(wav_bytes: bytes) -> dict:
    """
    Get information about WAV audio.

    Args:
        wav_bytes: WAV file as bytes

    Returns:
        Dictionary with audio info
    """
    with io.BytesIO(wav_bytes) as wav_file:
        with wave.open(wav_file, 'rb') as wf:
            return {
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'sample_rate': wf.getframerate(),
                'nframes': wf.getnframes(),
                'duration': wf.getnframes() / wf.getframerate(),
                'bytes': len(wav_bytes),
            }
