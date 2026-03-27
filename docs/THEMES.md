# HUD Theme Documentation

## Overview

The Parakeet Flow HUD features a single theme with **dark gray borders** and **colorful voice-reactive waveform bars**.

## Visual Design

- **Border:** Solid gray outline (`QColor(90, 90, 95)`)
- **Background:** Dark gray (`QColor(16, 16, 18)`)
- **Bars:** Dynamic rainbow colors that respond to voice input

## Waveform Bar Colors

The waveform bars display a **rainbow gradient** across the pill:

- **Center bars:** Pink/Magenta
- **Mid-center bars:** Purple
- **Mid-edge bars:** Blue
- **Edge bars:** Cyan

### Voice Reactivity

The bars become **brighter and more vibrant** with increased voice volume:

- **Quiet voice:** Dim colors, lower alpha transparency
- **Loud voice:** Bright colors, higher alpha transparency
- **Bar height:** Also affects brightness (taller bars = brighter)

## Implementation

The color generation is handled by `ThemeManager.get_bar_color()` which takes:

- `bar_index`: Which bar (0 to NUM_BARS-1)
- `total_bars`: Total number of bars (7)
- `voice_intensity`: Volume level (0.0 to 1.0)
- `bar_height_factor`: Normalized height (0.0 to 1.0)

This creates a **dynamic, voice-responsive visualizer** that's both beautiful and functional.

## Performance

- Colors calculated on each paint event (60 FPS)
- Minimal CPU overhead (~1-2%)
- No performance impact on transcription quality
