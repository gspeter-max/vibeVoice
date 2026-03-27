import pytest
from PySide6.QtGui import QPen, QBrush, QColor
from PySide6.QtCore import Qt
from theme_manager import ThemeManager, THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED


def test_theme_manager_initialization():
    """Verify ThemeManager initializes with correct theme and border_width > 0"""
    # Test with THEME_ORIGINAL
    tm = ThemeManager(THEME_ORIGINAL)
    assert tm.current_theme == THEME_ORIGINAL
    assert tm.border_width > 0
    assert tm.border_width == 1.2

    # Test with THEME_RAINBOW (gradient theme)
    tm_rainbow = ThemeManager(THEME_RAINBOW)
    assert tm_rainbow.current_theme == THEME_RAINBOW
    assert tm_rainbow.border_width > 0
    assert tm_rainbow.border_width == 2.5


def test_invalid_theme_defaults_to_original():
    """Verify invalid theme ID (999) falls back to THEME_ORIGINAL"""
    tm = ThemeManager(999)  # Invalid theme ID
    assert tm.current_theme == THEME_ORIGINAL


def test_theme_names_are_accessible():
    """Verify all 4 theme names are accessible"""
    assert ThemeManager.theme_name(THEME_ORIGINAL) == "Original (Solid Gray)"
    assert ThemeManager.theme_name(THEME_RAINBOW) == "Rainbow Gradient"
    assert ThemeManager.theme_name(THEME_RADIAL) == "Radial Glow"
    assert ThemeManager.theme_name(THEME_ANIMATED) == "Animated Aurora"
