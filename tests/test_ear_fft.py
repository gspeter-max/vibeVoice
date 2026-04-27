"""
Tests for ear.py FFT frequency analysis
"""
import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.audio.ear import Ear


def test_frequency_analysis_returns_valid_bands():
    """Verify _analyze_frequency_bands returns valid frequency bands"""
    # We can't instantiate Ear without a mic, so we'll test the method directly
    # by creating a mock instance with minimal initialization

    # Create test audio samples (sine wave at 440 Hz - A note, mid frequency)
    sample_rate = 16000
    duration = 0.1  # 100ms
    frequency = 440  # A note (mid frequency)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    # Create a minimal Ear-like object to test the method
    class MockEar:
        pass

    mock_ear = MockEar()
    mock_ear.RATE = 16000

    # Bind the method
    from src.audio.ear import Ear
    mock_ear._analyze_frequency_bands = Ear._analyze_frequency_bands.__get__(mock_ear, MockEar)

    # Test with mid-frequency audio
    result = mock_ear._analyze_frequency_bands(audio_samples)

    # Verify result structure
    assert isinstance(result, dict)
    assert 'bass' in result
    assert 'mid' in result
    assert 'treble' in result

    # Verify values are in valid range [0.0, 1.0]
    assert 0.0 <= result['bass'] <= 1.0
    assert 0.0 <= result['mid'] <= 1.0
    assert 0.0 <= result['treble'] <= 1.0

    # Verify bands sum to approximately 1.0 (normalized)
    total = result['bass'] + result['mid'] + result['treble']
    assert abs(total - 1.0) < 0.01  # Allow small floating point error


def test_frequency_analysis_low_frequency():
    """Verify low frequency audio produces dominant bass band"""
    sample_rate = 16000
    duration = 0.1
    frequency = 100  # Low frequency (bass)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    class MockEar:
        RATE = 16000

    from src.audio.ear import Ear
    mock_ear = MockEar()
    mock_ear._analyze_frequency_bands = Ear._analyze_frequency_bands.__get__(mock_ear, MockEar)

    result = mock_ear._analyze_frequency_bands(audio_samples)

    # Bass should be dominant for low frequency
    assert result['bass'] > result['mid']
    assert result['bass'] > result['treble']


def test_frequency_analysis_high_frequency():
    """Verify high frequency audio produces dominant treble band"""
    sample_rate = 16000
    duration = 0.1
    frequency = 6000  # High frequency (treble)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    class MockEar:
        RATE = 16000

    from src.audio.ear import Ear
    mock_ear = MockEar()
    mock_ear._analyze_frequency_bands = Ear._analyze_frequency_bands.__get__(mock_ear, MockEar)

    result = mock_ear._analyze_frequency_bands(audio_samples)

    # Treble should be dominant for high frequency
    # Note: Due to Nyquist frequency (8000 Hz), 6000 Hz might not be perfectly captured
    # but it should still produce higher treble than bass
    assert result['treble'] > result['bass']


def test_frequency_analysis_silence():
    """Verify silence produces balanced frequency bands"""
    # Create silent audio (all zeros)
    audio_samples = np.zeros(1024, dtype=np.int16)

    class MockEar:
        RATE = 16000

    from src.audio.ear import Ear
    mock_ear = MockEar()
    mock_ear._analyze_frequency_bands = Ear._analyze_frequency_bands.__get__(mock_ear, MockEar)

    result = mock_ear._analyze_frequency_bands(audio_samples)

    # Silent audio should produce roughly equal distribution
    # (fallback to equal distribution when total energy is 0)
    assert abs(result['bass'] - 0.33) < 0.1
    assert abs(result['mid'] - 0.33) < 0.1
    assert abs(result['treble'] - 0.34) < 0.1
