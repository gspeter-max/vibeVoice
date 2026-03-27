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
