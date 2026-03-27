import pytest
from PySide6.QtGui import QColor
from theme_manager import ThemeManager, THEME_ORIGINAL


def test_theme_manager_initialization():
    """Verify ThemeManager initializes correctly"""
    tm = ThemeManager(THEME_ORIGINAL)
    assert tm.current_theme == THEME_ORIGINAL
    assert tm.border_width == 1.2


def test_theme_name():
    """Verify theme name is accessible"""
    assert ThemeManager.theme_name(THEME_ORIGINAL) == "Original (Dark with Colorful Bars)"


def test_get_bar_color_returns_valid_color():
    """Verify get_bar_color returns valid QColor"""
    tm = ThemeManager(THEME_ORIGINAL)

    # Test center bar with no voice
    color = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.0, bar_height_factor=0.5)
    assert isinstance(color, QColor)
    assert color.alpha() > 0
    assert color.red() >= 0 and color.red() <= 255
    assert color.green() >= 0 and color.green() <= 255
    assert color.blue() >= 0 and color.blue() <= 255

    # Test edge bar with full voice
    color = tm.get_bar_color(bar_index=0, total_bars=7, voice_intensity=1.0, bar_height_factor=1.0)
    assert isinstance(color, QColor)
    assert color.alpha() > 0


def test_bar_colors_different_by_position():
    """Verify bars at different positions get different colors"""
    tm = ThemeManager(THEME_ORIGINAL)

    color_center = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.5, bar_height_factor=0.5)
    color_edge = tm.get_bar_color(bar_index=0, total_bars=7, voice_intensity=0.5, bar_height_factor=0.5)

    # Colors should be different
    assert (color_center.red(), color_center.green(), color_center.blue()) != \
           (color_edge.red(), color_edge.green(), color_edge.blue())


def test_bar_colors_always_vibrant():
    """Verify bars are always vibrant regardless of voice"""
    tm = ThemeManager(THEME_ORIGINAL)

    # Test with no voice
    color_quiet = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.0, bar_height_factor=0.5)

    # Should be fully opaque and vibrant
    assert color_quiet.alpha() == 255
    assert color_quiet.red() + color_quiet.green() + color_quiet.blue() > 400  # Vibrant colors
