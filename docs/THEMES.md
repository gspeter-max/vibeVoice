# HUD Theme Documentation

## Overview

The Parakeet Flow HUD features a single theme with **dark gray borders** and **frequency-based colorful waveform bars** that respond to voice input frequency content.

## Visual Design

- **Border:** Solid gray outline (`QColor(90, 90, 95)`)
- **Background:** Dark gray (`QColor(16, 16, 18)`)
- **Bars:** Dynamic frequency-based colors that respond to voice input

## Waveform Bar Colors (Frequency-Based)

The waveform bars display **colors based on audio frequency spectrum**:

- **Bass (low frequencies, 20-250 Hz)** → Warm colors (Red/Orange)
  - Examples: Low hum, deep vowels, bass guitar
  - Hue range: 0.0-0.12 (red to orange)

- **Mid (medium frequencies, 250-4000 Hz)** → Neutral colors (Green/Yellow)
  - Examples: Speech intelligibility range, most vocals
  - Hue range: 0.28-0.40 (green to yellow)

- **Treble (high frequencies, 4000-8000 Hz)** → Cool colors (Blue/Purple)
  - Examples: Consonants, high-pitched sounds, cymbals
  - Hue range: 0.58-0.78 (blue to purple)

### Frequency Analysis

The system uses **Fast Fourier Transform (FFT)** to analyze audio in real-time:

1. **Audio capture:** 16kHz sample rate, 1024-sample chunks
2. **FFT computation:** Converts time-domain audio to frequency-domain
3. **Band extraction:** Splits spectrum into bass/mid/treble bands
4. **Color mapping:** Dominant frequency band determines bar color

### Voice Reactivity

The bars respond to **both frequency content and volume**:

- **Frequency dominance:** Determines color (bass=red, mid=green, treble=blue)
- **Volume intensity:** Affects saturation (quiet=pastel, loud=vibrant)
- **Bar height:** Affects brightness (taller=brighter)

## Implementation

### Frequency Analysis (ear.py)

The `_analyze_frequency_bands()` method performs FFT analysis:

```python
def _analyze_frequency_bands(self, audio_samples: np.ndarray) -> dict:
    # Apply windowing function
    window = np.hanning(len(samples_float))

    # Compute FFT
    fft_result = np.fft.fft(windowed)
    fft_magnitude = np.abs(fft_result[:len(fft_result)//2])

    # Extract frequency bands
    bass_energy = np.sum(fft_magnitude[bass_mask])
    mid_energy = np.sum(fft_magnitude[mid_mask])
    treble_energy = np.sum(fft_magnitude[treble_mask])

    # Normalize to 0.0-1.0
    return {'bass': bass_norm, 'mid': mid_norm, 'treble': treble_norm}
```

### Color Generation (ThemeManager.get_bar_color())

Takes frequency bands and maps them to colors:

```python
def get_bar_color(self, bar_index, total_bars, voice_intensity,
                 bar_height_factor, frequency_bands):
    bass = frequency_bands['bass']
    mid = frequency_bands['mid']
    treble = frequency_bands['treble']

    # Find dominant frequency
    if bass >= mid and bass >= treble:
        base_hue = 0.0 + (bass * 0.12)  # Warm red/orange
    elif mid >= bass and mid >= treble:
        base_hue = 0.28 + (mid * 0.12)  # Green/yellow
    else:
        base_hue = 0.58 + (treble * 0.20)  # Blue/purple

    # Create color
    color = QColor.fromHsvF(hue, saturation, brightness, 1.0)
    return color
```

### Data Flow

```
ear.py (audio capture)
  → _audio_callback()
    → _analyze_frequency_bands() (FFT analysis)
      → UDP send: "vol:RMS,bass:BASS,mid:MID,treble:TREBLE"
        → hud.py (VolumeListener)
          → _on_frequency_bands()
            → self._frequency_bands = freq_bands
              → paintEvent()
                → ThemeManager.get_bar_color(frequency_bands=self._frequency_bands)
                  → Dynamic frequency-based colors!
```

## Performance

- FFT analysis: ~1-2% CPU per chunk
- Colors calculated at 60 FPS
- No performance impact on transcription quality
- Real-time frequency visualization with minimal latency

## Testing

See `tests/test_ear_fft.py` for FFT analysis tests and `tests/test_theme_manager.py` for color mapping tests.

## Examples

### Speaking with deep voice
- Frequency: Bass dominant
- Color: Red/Orange
- Visual: Warm, energetic bars

### Speaking with high-pitched voice
- Frequency: Treble dominant
- Color: Blue/Purple
- Visual: Cool, calm bars

### Normal speech
- Frequency: Mid dominant
- Color: Green/Yellow
- Visual: Balanced, neutral bars

### Whistling
- Frequency: Treble dominant
- Color: Blue/Purple
- Visual: Cool bars with high treble energy
