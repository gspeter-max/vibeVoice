"""Test audio utility functions."""
import numpy as np
import sys
sys.path.insert(0, '.')

from core.audio_utils import (
    numpy_to_wav,
    wav_to_numpy,
    normalize_audio,
    calculate_rms,
)


def test_numpy_to_wav_conversion():
    """Test numpy array to WAV conversion."""
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)).astype(np.float32)
    wav_bytes = numpy_to_wav(audio, sample_rate=22050)
    assert len(wav_bytes) > 0
    assert isinstance(wav_bytes, bytes)
    print("✓ test_numpy_to_wav_conversion passed")


def test_wav_roundtrip():
    """Test WAV conversion roundtrip."""
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)).astype(np.float32)
    wav_bytes = numpy_to_wav(audio)
    recovered_audio, sample_rate = wav_to_numpy(wav_bytes)
    assert sample_rate == 22050
    assert len(recovered_audio) == 22050
    assert np.allclose(audio, recovered_audio, atol=0.01)
    print("✓ test_wav_roundtrip passed")


def test_normalize_audio():
    """Test audio normalization."""
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)).astype(np.float32) * 0.1
    normalized = normalize_audio(audio)
    assert calculate_rms(normalized) > calculate_rms(audio)
    print("✓ test_normalize_audio passed")


if __name__ == "__main__":
    test_numpy_to_wav_conversion()
    test_wav_roundtrip()
    test_normalize_audio()
    print("\n✅ All audio utility tests passed!")
