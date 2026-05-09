"""Pure audio analysis helpers used by the Ear runtime.

This module contains stateless helpers that operate on audio bytes or audio
sample arrays. Keeping them here makes the behavior easier to test without
constructing the full microphone runtime.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from src import log


def get_rms(audio_block_bytes: bytes) -> float:
    """Measure the root-mean-square loudness of one PCM-16 audio block.

    The input is raw little-endian 16-bit mono audio bytes. The function turns
    those bytes into signed sample values, normalizes them to the `-1.0..1.0`
    range, and returns the RMS value as one stable loudness number.
    """

    sample_count = len(audio_block_bytes) // 2
    pcm_samples = struct.unpack(f"{sample_count}h", audio_block_bytes[: sample_count * 2])
    if not pcm_samples:
        return 0.0

    sum_of_squares = sum((sample / 32768.0) ** 2 for sample in pcm_samples)
    return math.sqrt(sum_of_squares / len(pcm_samples))


def boost_audio_chunk(audio_chunk_bytes: bytes, gain_multiplier: float) -> bytes:
    """Apply a safe gain multiplier to PCM-16 audio bytes.

    The output stays in signed 16-bit range by clipping after the gain is
    applied. Odd trailing bytes are dropped because PCM-16 samples must always
    use two bytes per sample.
    """

    if not audio_chunk_bytes:
        return audio_chunk_bytes

    if len(audio_chunk_bytes) % 2:
        audio_chunk_bytes = audio_chunk_bytes[:-1]

    pcm_samples = np.frombuffer(audio_chunk_bytes, dtype=np.int16).astype(np.float32)
    boosted_samples = (pcm_samples * gain_multiplier).clip(-32768, 32767).astype(np.int16)
    return boosted_samples.tobytes()


def analyze_frequency_bands(
    audio_samples: np.ndarray,
    *,
    sample_rate: int,
) -> dict[str, float]:
    """Split one audio sample array into normalized bass, mid, and treble bands.

    The result always contains the three keys `bass`, `mid`, and `treble`.
    Values are normalized to roughly sum to 1.0 when signal energy exists.
    When the signal is silent or FFT processing fails, the function returns the
    historical balanced fallback used by Ear.
    """

    try:
        samples_float = audio_samples.astype(np.float32) / 32768.0
        windowed_samples = samples_float * np.hanning(len(samples_float))
        fft_result = np.fft.fft(windowed_samples)
        fft_magnitude = np.abs(fft_result[: len(fft_result) // 2])
        frequency_bins = np.fft.fftfreq(len(windowed_samples), 1.0 / sample_rate)[
            : len(fft_magnitude)
        ]

        bass_energy = np.sum(fft_magnitude[(frequency_bins >= 20) & (frequency_bins < 250)])
        mid_energy = np.sum(fft_magnitude[(frequency_bins >= 250) & (frequency_bins < 4000)])
        treble_energy = np.sum(
            fft_magnitude[(frequency_bins >= 4000) & (frequency_bins < 8000)]
        )

        total_energy = bass_energy + mid_energy + treble_energy
        if total_energy > 0:
            return {
                "bass": bass_energy / total_energy,
                "mid": mid_energy / total_energy,
                "treble": treble_energy / total_energy,
            }

        return {"bass": 0.33, "mid": 0.33, "treble": 0.34}
    except Exception as error:
        log.info(f"[Ear] ⚠️ Frequency analysis failed: {error}")
        return {"bass": 0.33, "mid": 0.33, "treble": 0.34}
