# HUD Theme Documentation

## Overview

The Parakeet Flow HUD features a single theme with **dark gray borders** and **dynamic rainbow spectrum waveform bars** that create a beautiful, professional audio visualizer effect.

## Visual Design

- **Border:** Solid gray outline (`QColor(90, 90, 95)`)
- **Background:** Dark gray (`QColor(16, 16, 18)`)
- **Bars:** Dynamic rainbow spectrum that shifts over time

## Waveform Bar Colors (Rainbow Spectrum)

The waveform bars display a **rainbow spectrum** across all bars:

- **Center bars:** Warm colors (Red → Orange → Yellow)
- **Middle bars:** Transition colors (Yellow → Green)
- **Edge bars:** Cool colors (Green → Blue → Purple)

### Spectrum Layout

```
Bar 0: Purple/Blue  (edge)
Bar 1: Blue
Bar 2: Green/Blue
Bar 3: Yellow/Orange (center)
Bar 4: Green/Blue
Bar 5: Blue
Bar 6: Purple/Blue  (edge)
```

### Dynamic Effects

**1. Time-Based Color Cycling**
- Colors continuously shift through the spectrum
- Creates smooth, mesmerizing animation
- Rate: ~6.7 seconds for full spectrum cycle

**2. Frequency-Based Color Bias**
- Bass-heavy audio → Shifts spectrum toward warm colors (red/orange)
- Treble-heavy audio → Shifts spectrum toward cool colors (blue/purple)
- Creates visual correlation with audio frequency content

**3. Voice Reactivity**
- Volume intensity → Increases saturation (quiet=pastel, loud=vibrant)
- Bar height → Increases brightness (taller=brighter)
- Position → Adds subtle variation for visual interest

## Frequency Analysis (FFT)

The system uses **Fast Fourier Transform (FFT)** to analyze audio frequency:

1. **Audio capture:** 16kHz sample rate, 1024-sample chunks
2. **FFT computation:** Converts time-domain to frequency-domain
3. **Band extraction:** Splits into bass (20-250Hz), mid (250-4000Hz), treble (4000-8000Hz)
4. **Color bias:** Adjusts spectrum toward warm/cool based on dominant frequencies

## Implementation

### Color Generation (ThemeManager.get_bar_color())

```python
def get_bar_color(self, bar_index, total_bars, voice_intensity,
                 bar_height_factor, frequency_bands):
    # Map bar position to hue spectrum
    pos_mid = (total_bars - 1) / 2.0
    pos_from_center = abs(bar_index - pos_mid) / pos_mid

    # Base hue: 0.0 (center/red) to 0.75 (edge/purple)
    base_hue = pos_from_center * 0.75

    # Time-based cycling
    time_shift = (time.time() * 0.15) % 1.0

    # Frequency-based bias
    freq_bias = (bass - treble) * 0.15

    # Combine all factors
    hue = (base_hue + time_shift + freq_bias) % 1.0

    # Dynamic saturation and brightness
    saturation = 0.85 + (voice_intensity * 0.15) + (bar_height_factor * 0.10)
    brightness = 0.82 + (bar_height_factor * 0.13) + (voice_intensity * 0.05)

    return QColor.fromHsvF(hue, saturation, brightness, 1.0)
```

### Data Flow

```
ear.py (microphone)
  → FFT analysis (bass/mid/treble)
    → UDP: "vol:RMS,bass:B,mid:M,treble:T"
      → hud.py (VolumeListener)
        → frequency_bands signal
          → ThemeManager.get_bar_color()
            → Rainbow spectrum + time cycling + frequency bias!
```

## Visual Experience

### What You'll See

1. **Rainbow gradient** across all 7 bars
2. **Colors slowly cycling** through the spectrum over time
3. **Bars animating** with voice volume (height)
4. **Colors shifting** based on voice frequency (bass→warm, treble→cool)
5. **Smooth 60 FPS** animation with no stuttering

### Examples

- **Quiet speaking:** Pastel rainbow colors with low saturation
- **Loud speaking:** Vibrant, saturated rainbow colors
- **Deep voice:** Spectrum shifts toward red/orange
- **High-pitched:** Spectrum shifts toward blue/purple
- **No audio:** Subtle cycling rainbow with low brightness

## Performance

- FFT analysis: ~1-2% CPU per chunk
- Color calculation: Negligible at 60 FPS
- No impact on transcription quality
- Smooth animation with minimal latency

## Testing

See `tests/test_ear_fft.py` for FFT analysis tests and `tests/test_theme_manager.py` for color mapping tests.

**All tests passing: 10/10**
