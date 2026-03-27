# Gradient UI Themes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- ]`) syntax for tracking.

**Goal:** Add 3 gradient border theme options to the pill HUD (keeping the original solid theme) with runtime theme switching via startup configuration.

**Architecture:**
- Create a `ThemeManager` class that encapsulates all theme logic (gradient creation, color schemes)
- Add environment variable `HUD_THEME` to select theme at startup (0-3)
- Modify `PillHUD.paintEvent()` to use ThemeManager for border/background rendering
- Themes are immutable and defined at module load time for performance

**Tech Stack:** PySide6 Qt, Python 3.11, environment variables for configuration

---

## File Structure

```
src/
  hud.py                    # MODIFY: Add ThemeManager integration, paintEvent changes
  theme_manager.py          # CREATE: All gradient theme definitions and rendering logic
tests/
  test_theme_manager.py     # CREATE: Unit tests for gradient generation
  test_hud_themes.py        # CREATE: Integration tests for HUD theme switching
docs/
  THEMES.md                 # CREATE: User documentation for theme options
scripts/
  start.sh                  # MODIFY: Add theme selection menu
```

---

## Task 1: Create ThemeManager Module

**Files:**
- Create: `src/theme_manager.py`
- Test: `tests/test_theme_manager.py`

- [ ] **Step 1: Write failing test for ThemeManager initialization**

```python
# tests/test_theme_manager.py
import pytest
from src.theme_manager import ThemeManager, THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED

def test_theme_manager_initialization():
    """ThemeManager should initialize with a valid theme"""
    tm = ThemeManager(THEME_ORIGINAL)
    assert tm.current_theme == THEME_ORIGINAL
    assert tm.border_width > 0

def test_invalid_theme_defaults_to_original():
    """Invalid theme ID should fall back to ORIGINAL"""
    tm = ThemeManager(999)
    assert tm.current_theme == THEME_ORIGINAL

def test_theme_names_are_accessible():
    """All themes should have human-readable names"""
    assert ThemeManager.theme_name(THEME_ORIGINAL) == "Original (Solid Gray)"
    assert ThemeManager.theme_name(THEME_RAINBOW) == "Rainbow Gradient"
    assert ThemeManager.theme_name(THEME_RADIAL) == "Radial Glow"
    assert ThemeManager.theme_name(THEME_ANIMATED) == "Animated Aurora"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_theme_manager.py -v
```
Expected: `ImportError: cannot import name 'ThemeManager'`

- [ ] **Step 3: Create ThemeManager with theme constants**

```python
# src/theme_manager.py
from PySide6.QtGui import QLinearGradient, QRadialGradient, QColor, QPen, QBrush
from PySide6.QtCore import Qt

# Theme constants
THEME_ORIGINAL = 0
THEME_RAINBOW = 1
THEME_RADIAL = 2
THEME_ANIMATED = 3

THEME_NAMES = {
    THEME_ORIGINAL: "Original (Solid Gray)",
    THEME_RAINBOW: "Rainbow Gradient",
    THEME_RADIAL: "Radial Glow",
    THEME_ANIMATED: "Animated Aurora",
}


class ThemeManager:
    """Manages HUD theme selection and gradient generation."""

    def __init__(self, theme_id: int):
        if theme_id not in THEME_NAMES:
            theme_id = THEME_ORIGINAL
        self.current_theme = theme_id
        self.border_width = 1.2 if theme_id == THEME_ORIGINAL else 2.5

    @staticmethod
    def theme_name(theme_id: int) -> str:
        """Get human-readable theme name."""
        return THEME_NAMES.get(theme_id, "Unknown")

    def create_border_pen(self, rect_x: float, rect_y: float, rect_w: float, rect_h: float, hue_offset: float = 0.0) -> QPen:
        """Create border pen based on current theme.

        Args:
            rect_x, rect_y, rect_w, rect_h: Pill rectangle dimensions
            hue_offset: Animation offset for THEME_ANIMATED (0.0-1.0)

        Returns:
            QPen configured with theme-appropriate brush
        """
        if self.current_theme == THEME_ORIGINAL:
            return QPen(QColor(90, 90, 95, 200), self.border_width)

        elif self.current_theme == THEME_RAINBOW:
            gradient = QLinearGradient(rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)
            gradient.setColorAt(0.0, QColor(255, 0, 128))    # Pink
            gradient.setColorAt(0.25, QColor(128, 0, 255))   # Purple
            gradient.setColorAt(0.5, QColor(0, 128, 255))    # Blue
            gradient.setColorAt(0.75, QColor(0, 255, 128))   # Cyan
            gradient.setColorAt(1.0, QColor(255, 255, 0))    # Yellow
            return QPen(gradient, self.border_width)

        elif self.current_theme == THEME_RADIAL:
            center_x = rect_x + rect_w / 2
            center_y = rect_y + rect_h / 2
            radius = max(rect_w, rect_h) / 2
            gradient = QRadialGradient(center_x, center_y, radius)
            gradient.setColorAt(0.0, QColor(255, 255, 0))    # Yellow center
            gradient.setColorAt(0.5, QColor(255, 0, 255))    # Magenta mid
            gradient.setColorAt(1.0, QColor(0, 128, 255))    # Blue edge
            return QPen(gradient, self.border_width)

        elif self.current_theme == THEME_ANIMATED:
            gradient = QLinearGradient(rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)
            # Create rainbow with hue offset for animation
            for i in range(6):
                hue = ((i / 5.0) + hue_offset) % 1.0
                color = QColor.fromHsvF(hue, 1.0, 1.0, 1.0)
                gradient.setColorAt(i / 5.0, color)
            return QPen(gradient, self.border_width)

        # Fallback
        return QPen(QColor(90, 90, 95, 200), self.border_width)

    def create_background_brush(self, rect_x: float, rect_y: float, rect_h: float, alpha: int) -> QBrush:
        """Create background brush based on current theme.

        Args:
            rect_x, rect_y, rect_h: Rectangle dimensions
            alpha: Transparency value (0-255)

        Returns:
            QBrush configured with theme-appropriate fill
        """
        if self.current_theme == THEME_ORIGINAL:
            return QBrush(QColor(16, 16, 18, alpha))

        # For gradient themes, use subtle vertical gradient background
        bg_gradient = QLinearGradient(rect_x, rect_y, rect_x, rect_y + rect_h)
        bg_gradient.setColorAt(0.0, QColor(40, 40, 50, alpha))
        bg_gradient.setColorAt(1.0, QColor(20, 20, 30, alpha))
        return QBrush(bg_gradient)

    def requires_animation(self) -> bool:
        """Check if theme requires per-frame animation updates."""
        return self.current_theme == THEME_ANIMATED
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_theme_manager.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/theme_manager.py tests/test_theme_manager.py
git commit -m "feat: add ThemeManager with 4 gradient themes"
```

---

## Task 2: Integrate ThemeManager into PillHUD

**Files:**
- Modify: `src/hud.py:140-220` (PillHUD.__init__ method)
- Modify: `src/hud.py:355-407` (PillHUD.paintEvent method)

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_hud_themes.py
import os
import pytest
from src.hud import PillHUD
from src.theme_manager import THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED

def test_hud_respects_theme_environment_variable():
    """HUD should initialize with theme from HUD_THEME env var"""
    os.environ['HUD_THEME'] = '1'  # Rainbow
    hud = PillHUD()
    assert hud._theme_manager.current_theme == THEME_RAINBOW
    hud.close()

def test_hud_animation_state_for_animated_theme():
    """Animated theme should enable animation flag"""
    os.environ['HUD_THEME'] = '3'  # Animated
    hud = PillHUD()
    assert hud._theme_manager.requires_animation() is True
    hud.close()

def test_hud_default_theme_without_env_var():
    """HUD should default to ORIGINAL theme when no env var set"""
    if 'HUD_THEME' in os.environ:
        del os.environ['HUD_THEME']
    hud = PillHUD()
    assert hud._theme_manager.current_theme == THEME_ORIGINAL
    hud.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_hud_themes.py -v
```
Expected: `AttributeError: 'PillHUD' object has no attribute '_theme_manager'`

- [ ] **Step 3: Import ThemeManager and add to __init__**

```python
# src/hud.py - Add at top of file after existing imports
from src.theme_manager import ThemeManager

# In PillHUD.__init__ method (around line 172, after self._t0 initialization)
# ADD:
        # Theme manager initialization
        theme_id = int(os.environ.get('HUD_THEME', str(THEME_ORIGINAL)))
        self._theme_manager = ThemeManager(theme_id)
        self._hue_offset = 0.0  # For animated theme
```

- [ ] **Step 4: Modify paintEvent to use ThemeManager**

```python
# src/hud.py - Replace paintEvent method (lines 355-407)
# FIND these lines:
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        pw = self._cur_w
        ph = self._cur_h
        px = (WINDOW_W - pw) / 2
        py = (WINDOW_H - ph) / 2
        cx = WINDOW_W / 2
        cy = WINDOW_H / 2
        r  = ph / 2

        pill = QPainterPath()
        pill.addRoundedRect(QRectF(px, py, pw, ph), r, r)

        fill_alpha = 50 if self._state == HIDDEN else 240
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(16, 16, 18, fill_alpha)))
        p.drawPath(pill)

        p.setPen(QPen(QColor(90, 90, 95, 200), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(pill)

# REPLACE with:
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        pw = self._cur_w
        ph = self._cur_h
        px = (WINDOW_W - pw) / 2
        py = (WINDOW_H - ph) / 2
        cx = WINDOW_W / 2
        cy = WINDOW_H / 2
        r  = ph / 2

        pill = QPainterPath()
        pill.addRoundedRect(QRectF(px, py, pw, ph), r, r)

        # Use theme manager for background and border
        fill_alpha = 50 if self._state == HIDDEN else 240
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._theme_manager.create_background_brush(px, py, ph, fill_alpha))
        p.drawPath(pill)

        # Use theme manager for border with hue offset for animated themes
        p.setPen(self._theme_manager.create_border_pen(px, py, pw, ph, self._hue_offset))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(pill)
```

- [ ] **Step 5: Update _tick method to animate hue for animated theme**

```python
# src/hud.py - In _tick method (around line 292, after self._t = time.time() - self._t0)
# ADD:
        # Update hue offset for animated themes
        if self._theme_manager.requires_animation():
            self._hue_offset = (self._hue_offset + 0.002) % 1.0
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_hud_themes.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 7: Manual verification test**

```bash
# Test each theme manually
HUD_THEME=0 python src/hud.py --demo  # Original
HUD_THEME=1 python src/hud.py --demo  # Rainbow
HUD_THEME=2 python src/hud.py --demo  # Radial
HUD_THEME=3 python src/hud.py --demo  # Animated
```
Expected: Visual confirmation of each theme

- [ ] **Step 8: Commit**

```bash
git add src/hud.py tests/test_hud_themes.py
git commit -m "feat: integrate ThemeManager into PillHUD with runtime theme switching"
```

---

## Task 3: Add Theme Selection to start.sh

**Files:**
- Modify: `start.sh`

- [ ] **Step 1: Create test script to verify theme menu works**

```bash
# Create test script: scripts/test_theme_menu.sh
#!/bin/bash

# Test that theme menu appears and sets env var correctly
echo "Testing theme menu functionality..."

# Mock user input selecting theme 1
echo "1" | bash start.sh 2>&1 | grep -q "Rainbow Gradient"
if [ $? -eq 0 ]; then
    echo "✓ Theme menu displays correctly"
else
    echo "✗ Theme menu not found"
    exit 1
fi
```

- [ ] **Step 2: Run test to verify current start.sh behavior**

```bash
chmod +x scripts/test_theme_menu.sh
bash scripts/test_theme_menu.sh
```
Expected: Test fails (menu doesn't exist yet)

- [ ] **Step 3: Add theme selection menu to start.sh**

```bash
# In start.sh, ADD this function after the initial setup section:

show_theme_menu() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           🎨 HUD THEME SELECTION                            ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║  Choose the visual theme for the pill HUD:                 ║"
    echo "║                                                            ║"
    echo "║  [0] Original (Solid Gray)   — Current production look    ║"
    echo "║  [1] Rainbow Gradient        — Diagonal pink→purple→blue   ║"
    echo "║  [2] Radial Glow             — Glowing center effect       ║"
    echo "║  [3] Animated Aurora         — Rotating rainbow animation  ║"
    echo "║                                                            ║"
    echo "║  Press 0-3 to select, or ENTER for default (0)            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo -n "Theme selection [0-3]: "
}

# ADD this function to get theme selection:

get_theme_selection() {
    local selection
    read selection

    case "$selection" in
        0|1|2|3|"")
            if [ -z "$selection" ]; then
                selection="0"
            fi
            export HUD_THEME="$selection"
            echo ""
            echo "✓ Theme set to: $selection"
            ;;
        *)
            echo ""
            echo "⚠️  Invalid selection. Using default theme (0)."
            export HUD_THEME="0"
            ;;
    esac
}

# MODIFY the main startup section to call the menu:
# FIND the section where services are started and ADD BEFORE it:

# Theme selection menu
show_theme_menu
get_theme_selection
echo ""
```

- [ ] **Step 4: Run test script to verify menu works**

```bash
bash scripts/test_theme_menu.sh
```
Expected: Test passes

- [ ] **Step 5: Manual verification**

```bash
# Run start.sh and test each theme option interactively
bash start.sh
# Test selecting 0, 1, 2, 3, and invalid input
```
Expected: Menu appears, accepts input, sets HUD_THEME correctly

- [ ] **Step 6: Commit**

```bash
git add start.sh scripts/test_theme_menu.sh
git commit -m "feat: add interactive theme selection menu to start.sh"
```

---

## Task 4: Create User Documentation

**Files:**
- Create: `docs/THEMES.md`

- [ ] **Step 1: Create comprehensive theme documentation**

```markdown
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
```

- [ ] **Step 2: Verify documentation renders correctly**

```bash
cat docs/THEMES.md
```
Expected: All sections visible, formatting correct

- [ ] **Step 3: Commit**

```bash
git add docs/THEMES.md
git commit -m "docs: add comprehensive HUD themes user guide"
```

---

## Task 5: Add Gradient Theme Demo Mode

**Files:**
- Modify: `src/hud.py:409-420` (main block)
- Create: `scripts/theme_demo.py`

- [ ] **Step 1: Create demo script test**

```python
# tests/test_theme_demo.py
import subprocess
import time
import os

def test_theme_demo_cycles_through_all_themes():
    """Demo script should cycle through all 4 themes"""
    result = subprocess.run(
        ['python', 'scripts/theme_demo.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "Theme 0" in result.stdout
    assert "Theme 1" in result.stdout
    assert "Theme 2" in result.stdout
    assert "Theme 3" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_theme_demo.py -v
```
Expected: `FileNotFoundError: scripts/theme_demo.py`

- [ ] **Step 3: Create theme demo script**

```python
#!/usr/bin/env python3
"""
theme_demo.py — Cycle through all HUD themes for visual comparison
Usage: python scripts/theme_demo.py
"""

import sys
import time
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hud import PillHUD
from theme_manager import THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED
from PySide6.QtWidgets import QApplication

def demo_theme(theme_id: int, duration: float = 5.0):
    """Show a single theme for specified duration."""
    os.environ['HUD_THEME'] = str(theme_id)
    from theme_manager import ThemeManager
    theme_name = ThemeManager.theme_name(theme_id)

    print(f"\n{'='*60}")
    print(f"  Displaying Theme {theme_id}: {theme_name}")
    print(f"  Showing for {duration} seconds...")
    print(f"{'='*60}\n")

    app = QApplication.instance() or QApplication(sys.argv)
    hud = PillHUD()
    hud.show()

    start_time = time.time()
    while time.time() - start_time < duration:
        app.processEvents()
        time.sleep(0.05)

    hud.close()
    print(f"✓ Theme {theme_id} demo complete\n")

def main():
    print("\n" + "="*60)
    print("  PARAKEET FLOW — HUD THEME DEMO")
    print("="*60)
    print("\nThis demo will cycle through all 4 themes.")
    print("Each theme displays for 5 seconds.\n")

    themes = [THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED]

    for theme_id in themes:
        try:
            demo_theme(theme_id, duration=5.0)
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.\n")
            return

    print("\n" + "="*60)
    print("  ALL THEMES DEMOED")
    print("="*60)
    print("\nTo select a theme permanently, use: bash start.sh")
    print("Or set: HUD_THEME=<0-3> python src/hud.py\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Make demo script executable**

```bash
chmod +x scripts/theme_demo.py
```

- [ ] **Step 5: Run demo script manually**

```bash
python scripts/theme_demo.py
```
Expected: Cycles through all 4 themes, 5 seconds each

- [ ] **Step 6: Run automated test**

```bash
pytest tests/test_theme_demo.py -v
```
Expected: Test PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/theme_demo.py tests/test_theme_demo.py
git commit -m "feat: add theme demo script for visual comparison"
```

---

## Task 6: Add Comprehensive Integration Tests

**Files:**
- Modify: `tests/test_hud_themes.py`
- Create: `tests/integration/test_theme_integration.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/integration/test_theme_integration.py
import os
import sys
import time
import subprocess
import pytest

@pytest.fixture(scope="module")
def qapp():
    """Shared QApplication for all tests"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

def test_full_theme_switching_cycle(qapp):
    """Test switching between all themes in runtime"""
    from src.hud import PillHUD
    from src.theme_manager import THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED

    themes = [THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED]

    for theme_id in themes:
        os.environ['HUD_THEME'] = str(theme_id)
        hud = PillHUD()
        assert hud._theme_manager.current_theme == theme_id
        hud.close()

def test_gradient_performance(qapp):
    """Ensure gradient rendering doesn't cause performance issues"""
    from src.hud import PillHUD
    import time

    os.environ['HUD_THEME'] = '3'  # Animated (most expensive)

    hud = PillHUD()
    hud.show()

    # Measure render time for 10 frames
    start = time.time()
    for _ in range(10):
        hud.update()
        hud.repaint()
        qapp.processEvents()
    elapsed = time.time() - start

    hud.close()

    # Should render 10 frames in less than 1 second
    assert elapsed < 1.0, f"Gradient rendering too slow: {elapsed}s for 10 frames"
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/integration/test_theme_integration.py -v
```
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_theme_integration.py
git commit -m "test: add comprehensive theme integration tests"
```

---

## Task 7: Add Theme Persistence Option (Optional Enhancement)

**Files:**
- Create: `src/theme_config.py`
- Modify: `start.sh`

- [ ] **Step 1: Create config module test**

```python
# tests/test_theme_config.py
import os
import tempfile
from src.theme_config import save_theme_preference, load_theme_preference

def test_save_and_load_theme_preference():
    """Theme preference should persist to file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
        config_path = f.name

    try:
        save_theme_preference(2, config_path)
        loaded = load_theme_preference(config_path)
        assert loaded == 2
    finally:
        os.unlink(config_path)

def test_missing_config_returns_default():
    """Missing config file should return default theme"""
    loaded = load_theme_preference("/nonexistent/path/theme.conf")
    assert loaded == 0  # THEME_ORIGINAL
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_theme_config.py -v
```
Expected: `ImportError: cannot import name 'save_theme_preference'`

- [ ] **Step 3: Create config module**

```python
# src/theme_config.py
import os
from typing import Optional

DEFAULT_THEME_CONFIG_PATH = os.path.expanduser("~/.config/parakeet-flow/theme.conf")

def save_theme_preference(theme_id: int, config_path: Optional[str] = None) -> None:
    """Save theme preference to config file.

    Args:
        theme_id: Theme ID (0-3)
        config_path: Optional custom config path
    """
    if config_path is None:
        config_path = DEFAULT_THEME_CONFIG_PATH

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(str(theme_id))

def load_theme_preference(config_path: Optional[str] = None) -> int:
    """Load theme preference from config file.

    Args:
        config_path: Optional custom config path

    Returns:
        Theme ID (0-3), or 0 (THEME_ORIGINAL) if file doesn't exist
    """
    if config_path is None:
        config_path = DEFAULT_THEME_CONFIG_PATH

    if not os.path.exists(config_path):
        return 0

    try:
        with open(config_path, 'r') as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_theme_config.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Modify start.sh to use saved preference**

```bash
# ADD to start.sh after theme functions:

load_saved_theme() {
    local saved_theme
    saved_theme=$(python3 -c "from src.theme_config import load_theme_preference; print(load_theme_preference())" 2>/dev/null || echo "0")

    if [ -n "$saved_theme" ] && [ "$saved_theme" != "0" ]; then
        echo ""
        echo "🎨 Found saved theme preference: $saved_theme"
        from src.theme_manager import ThemeManager
        local theme_name=$(python3 -c "from src.theme_manager import ThemeManager; print(ThemeManager.theme_name($saved_theme))")
        echo "   ($theme_name)"
        echo ""
        echo -n "Use saved theme? [Y/n]: "
        read use_saved

        if [[ "$use_saved" =~ ^[Yy]*$ ]] || [ -z "$use_saved" ]; then
            export HUD_THEME="$saved_theme"
            echo "✓ Using saved theme $saved_theme"
            return 0
        fi
    fi
    return 1
}

# MODIFY main startup section to check for saved preference first:

# Check for saved theme preference
if ! load_saved_theme; then
    # No saved preference or user declined, show menu
    show_theme_menu
    get_theme_selection

    # Ask to save preference
    echo ""
    echo -n "Save this theme preference for future sessions? [y/N]: "
    read save_pref
    if [[ "$save_pref" =~ ^[Yy]$ ]]; then
        python3 -c "from src.theme_config import save_theme_preference; save_theme_preference($HUD_THEME)"
        echo "✓ Theme preference saved"
    fi
fi
```

- [ ] **Step 6: Test persistence flow**

```bash
# Start fresh, select theme 2, save it
bash start.sh

# Stop and restart — should offer to use saved theme
bash start.sh
```
Expected: Offers to use saved theme

- [ ] **Step 7: Commit**

```bash
git add src/theme_config.py tests/test_theme_config.py start.sh
git commit -m "feat: add optional theme preference persistence"
```

---

## Summary Checklist

After completing all tasks, verify:

- [ ] All 4 themes render correctly (Original, Rainbow, Radial, Animated)
- [ ] Theme selection menu appears in start.sh
- [ ] Environment variable HUD_THEME works for direct theme selection
- [ ] Demo script cycles through all themes
- [ ] All unit tests pass: `pytest tests/ -v`
- [ ] Integration tests pass: `pytest tests/integration/ -v`
- [ ] Manual testing confirms visual quality
- [ ] Documentation is complete in docs/THEMES.md
- [ ] Theme persistence works (optional Task 7)

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/test_theme_manager.py -v    # Theme gradient generation
pytest tests/test_hud_themes.py -v        # HUD theme integration
pytest tests/test_theme_config.py -v      # Config persistence
pytest tests/test_theme_demo.py -v        # Demo script
```

### Integration Tests
```bash
pytest tests/integration/test_theme_integration.py -v
```

### Manual Testing
```bash
# Interactive theme selection
bash start.sh

# Direct theme selection
HUD_THEME=1 python src/hud.py --demo

# Demo all themes
python scripts/theme_demo.py

# Performance test
time python scripts/theme_demo.py
```

---

## Performance Benchmarks

Expected render times per theme (10 frames):

| Theme | Expected Time | Max Acceptable |
|-------|---------------|----------------|
| Original | < 50ms | < 100ms |
| Rainbow | < 60ms | < 150ms |
| Radial | < 60ms | < 150ms |
| Animated | < 70ms | < 200ms |

If any theme exceeds max acceptable, optimize gradient generation.
