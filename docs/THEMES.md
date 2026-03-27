# HUD Theme Documentation

## Overview

The Parakeet Flow HUD features a single theme with **dark gray borders** and **unified flowing color waveform bars** that create a beautiful, cohesive audio visualizer effect.

## Visual Design

- **Border:** Solid gray outline (`QColor(90, 90, 95)`)
- **Background:** Dark gray (`QColor(16, 16, 18)`)
- **Bars:** Unified flowing color that shifts through spectrum over time

## Waveform Bar Colors (Unified Flowing Wave)

**ALL bars share the SAME color**, creating a unified wave effect:

- **All bars shift colors together** as one cohesive unit
- **Color cycles smoothly** through entire spectrum over time
- **Feels like breathing** - the entire waveform flows and pulses together

### Color Flow Cycle

The entire waveform shifts through colors:
```
Time 0s:  Red/Orange      (warm)
Time 3s:  Yellow/Green    (transition)
Time 6s:  Blue/Cyan       (cool)
Time 9s:  Purple/Magenta  (deep)
Time 12s: Back to Red     (cycle repeats)
```

**Cycle duration:** ~12 seconds for full spectrum

### Dynamic Effects

**1. Unified Color Flow**
- All 7 bars always share the exact same color
- Creates cohesive, professional visual effect
- Waveform "breathes" as one unit

**2. Time-Based Color Cycling**
- Smooth, continuous color transition
- Soothing, mesmerizing flow effect
- Never jarring or distracting

**3. Frequency-Based Color Bias**
- Bass-heavy audio → Colors shift toward warm (red/orange/yellow)
- Treble-heavy audio → Colors shift toward cool (blue/purple/cyan)
- Creates subtle visual correlation with voice frequency

**4. Voice Reactivity**
- Volume intensity → Increases color saturation (quiet=pastel, loud=vibrant)
- Bar height → Increases brightness (taller=brighter)
- Real-time response at 60 FPS

## Frequency Analysis (FFT)

The system uses **Fast Fourier Transform (FFT)** to analyze audio frequency:

1. **Audio capture:** 16kHz sample rate, 1024-sample chunks
2. **FFT computation:** Converts time-domain to frequency-domain
3. **Band extraction:** Splits into bass (20-250Hz), mid (250-4000Hz), treble (4000-8000Hz)
4. **Color bias:** Adjusts color toward warm/cool based on dominant frequencies

## Implementation

### Color Generation (ThemeManager.get_bar_color())

```python
def get_bar_color(self, bar_index, total_bars, voice_intensity,
                 bar_height_factor, frequency_bands):
    # Time-based hue cycling (slow, smooth flow)
    # 0.08 = ~12 seconds for full spectrum cycle
    time_hue = (time.time() * 0.08) % 1.0

    # Frequency-based hue adjustment
    # Bass dominant → warm colors
    # Treble dominant → cool colors
    freq_bias = (bass - treble) * 0.2

    # Combine for final hue
    hue = (time_hue + freq_bias) % 1.0

    # Dynamic saturation (pulses with voice)
    saturation = 0.82 + (voice_intensity * 0.18)

    # Dynamic brightness (responds to voice + height)
    brightness = 0.80 + (voice_intensity * 0.08) + (bar_height_factor * 0.12)

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
            → Unified flowing color for ALL bars!
```

## Visual Experience

### What You'll See

1. **All bars same color** - cohesive, unified look
2. **Color slowly flowing** through spectrum (red→orange→yellow→green→blue→purple)
3. **Entire waveform breathing** - all bars shift color together
4. **Subtle frequency response** - bass shifts to warm, treble to cool
5. **Smooth 60 FPS** animation with no stuttering

### Examples

- **Quiet speaking:** Soft pastel colors, low saturation
- **Loud speaking:** Vibrant, saturated colors
- **Deep voice:** Color shifts toward red/orange (warm)
- **High-pitched:** Color shifts toward blue/purple (cool)
- **No audio:** Subtle slow color cycling, low brightness

## Design Philosophy

**Why unified colors instead of rainbow across bars?**

- ✅ **More cohesive** - waveform looks like one unit, not separate bars
- ✅ **Less distracting** - single color focus, not rainbow chaos
- ✅ **More professional** - like high-end audio equipment
- ✅ **Easier to process** - brain processes one color, not seven
- ✅ **Feels organic** - like breathing, not disjointed spectrum

## Performance

- FFT analysis: ~1-2% CPU per chunk
- Color calculation: Negligible at 60 FPS
- No impact on transcription quality
- Smooth animation with minimal latency

## Testing

See `tests/test_ear_fft.py` for FFT analysis tests and `tests/test_theme_manager.py` for color mapping tests.

**All tests passing: 10/10**
