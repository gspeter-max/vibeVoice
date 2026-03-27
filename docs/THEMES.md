# HUD Themes Documentation

## Overview

The Parakeet Flow HUD supports 4 visual themes that can be selected at startup via the `start.sh` script.

## Theme Options

### Theme 0: Original (Solid Gray)
The default production theme with a solid gray border and dark background.
- **Border:** Solid gray `QColor(90, 90, 95, 200)`
- **Background:** Solid dark gray `QColor(16, 16, 18, alpha)`
- **Animation:** None
- **Use Case:** Production use, minimal visual distraction

### Theme 1: Rainbow Gradient
A diagonal linear gradient with vibrant pink→purple→blue→cyan→yellow colors.
- **Border:** 5-stop diagonal gradient (2.5px width)
- **Background:** Subtle vertical gradient
- **Animation:** None
- **Colors:** Pink, Purple, Blue, Cyan, Yellow
- **Use Case:** High visibility, energetic aesthetic

### Theme 2: Radial Glow
A radial gradient emanating from the center with yellow→magenta→blue.
- **Border:** Radial gradient (2.5px width)
- **Background:** Subtle vertical gradient
- **Animation:** None
- **Colors:** Yellow center, Magenta mid, Blue edge
- **Use Case:** Glowing/pulsing aesthetic

### Theme 3: Animated Aurora
An animated rainbow gradient that slowly rotates through hues.
- **Border:** 6-stop animated linear gradient (2.5px width)
- **Background:** Subtle vertical gradient
- **Animation:** Continuous hue rotation (0.2% per frame)
- **Colors:** Full HSV spectrum
- **Use Case:** Dynamic/animated aesthetic, visual interest

## How to Select a Theme

### Interactive Menu (Recommended)
1. Run `bash start.sh`
2. Select theme 0-3 when prompted
3. The selected theme applies to all HUD instances

### Direct Environment Variable
```bash
HUD_THEME=1 python src/hud.py --demo
HUD_THEME=2 bash start.sh
```

### Per-Session Selection
Each time you run `start.sh`, you can select a different theme. The choice is not persisted.

## Implementation Details

### Theme Manager Architecture
- **File:** `src/theme_manager.py`
- **Class:** `ThemeManager`
- **Methods:**
  - `create_border_pen(x, y, w, h, hue_offset)` — Returns themed QPen
  - `create_background_brush(x, y, h, alpha)` — Returns themed QBrush
  - `requires_animation()` — Returns True for THEME_ANIMATED

### Performance Considerations
- Gradients are recreated on each paint event
- THEME_ANIMATED adds ~2% CPU overhead for hue rotation
- Theme selection happens at startup, zero runtime cost for switching

## Troubleshooting

### Theme Not Applying
1. Verify `HUD_THEME` environment variable is set: `echo $HUD_THEME`
2. Check hud.py logs for theme initialization
3. Ensure theme_manager.py is in src/

### Animation Not Working
1. Verify you selected Theme 3 (Animated Aurora)
2. Check that `_tick()` is running (look for timer logs)
3. Ensure `requires_animation()` returns True

## Future Enhancements
Possible additions:
- Theme persistence across sessions
- Custom color scheme configuration
- Gradient direction configuration
- Animation speed control
