"""Tests for Ear runtime frequency analysis helpers."""
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.audio.ear_runtime.analysis import analyze_frequency_bands, get_rms


def test_frequency_analysis_returns_valid_bands():
    """Verify frequency analysis returns valid normalized bands."""
    sample_rate = 16000
    duration = 0.1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    result = analyze_frequency_bands(audio_samples, sample_rate=sample_rate)

    assert isinstance(result, dict)
    assert 'bass' in result
    assert 'mid' in result
    assert 'treble' in result

    assert 0.0 <= result['bass'] <= 1.0
    assert 0.0 <= result['mid'] <= 1.0
    assert 0.0 <= result['treble'] <= 1.0

    total = result['bass'] + result['mid'] + result['treble']
    assert abs(total - 1.0) < 0.01


def test_analysis_module_frequency_helper_matches_expected_shape():
    sample_rate = 16000
    duration = 0.1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    result = analyze_frequency_bands(audio_samples, sample_rate=sample_rate)

    assert set(result.keys()) == {"bass", "mid", "treble"}
    assert 0.0 <= result["bass"] <= 1.0
    assert 0.0 <= result["mid"] <= 1.0
    assert 0.0 <= result["treble"] <= 1.0


def test_analysis_module_get_rms_returns_positive_value_for_signal():
    audio_bytes = (np.array([1000, -1000, 1000, -1000], dtype=np.int16)).tobytes()
    assert get_rms(audio_bytes) > 0.0


def test_frequency_analysis_low_frequency():
    """Verify low frequency audio produces dominant bass band"""
    sample_rate = 16000
    duration = 0.1
    frequency = 100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    result = analyze_frequency_bands(audio_samples, sample_rate=sample_rate)

    assert result['bass'] > result['mid']
    assert result['bass'] > result['treble']


def test_frequency_analysis_high_frequency():
    """Verify high frequency audio produces dominant treble band"""
    sample_rate = 16000
    duration = 0.1
    frequency = 6000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    result = analyze_frequency_bands(audio_samples, sample_rate=sample_rate)

    assert result['treble'] > result['bass']


def test_frequency_analysis_silence():
    """Verify silence produces balanced frequency bands"""
    audio_samples = np.zeros(1024, dtype=np.int16)
    result = analyze_frequency_bands(audio_samples, sample_rate=16000)

    assert abs(result['bass'] - 0.33) < 0.1
    assert abs(result['mid'] - 0.33) < 0.1
    assert abs(result['treble'] - 0.34) < 0.1
