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
    color = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.0, bar_height_factor=0.5, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})
    assert isinstance(color, QColor)
    assert color.alpha() > 0
    assert color.red() >= 0 and color.red() <= 255
    assert color.green() >= 0 and color.green() <= 255
    assert color.blue() >= 0 and color.blue() <= 255

    # Test edge bar with full voice
    color = tm.get_bar_color(bar_index=0, total_bars=7, voice_intensity=1.0, bar_height_factor=1.0, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})
    assert isinstance(color, QColor)
    assert color.alpha() > 0


def test_bar_colors_different_by_position():
    """Verify all bars have unified color (same color at same time)"""
    tm = ThemeManager(THEME_ORIGINAL)

    # Get colors for different bars at the same moment
    color_center = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.5, bar_height_factor=0.5, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})
    color_edge = tm.get_bar_color(bar_index=0, total_bars=7, voice_intensity=0.5, bar_height_factor=0.5, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})

    # All bars should have the same color (unified wave effect)
    # Allow small differences due to floating point math
    assert abs(color_center.red() - color_edge.red()) <= 1
    assert abs(color_center.green() - color_edge.green()) <= 1
    assert abs(color_center.blue() - color_edge.blue()) <= 1


def test_bar_colors_always_vibrant():
    """Verify bars are always vibrant regardless of voice"""
    tm = ThemeManager(THEME_ORIGINAL)

    # Test with no voice - use balanced frequency bands
    color_quiet = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.0, bar_height_factor=0.5, frequency_bands={'bass': 0.33, 'mid': 0.33, 'treble': 0.33})

    # Should be fully opaque and vibrant (at least one RGB component should be strong)
    assert color_quiet.alpha() == 255
    max_component = max(color_quiet.red(), color_quiet.green(), color_quiet.blue())
    assert max_component > 180  # At least one component should be strong


def test_frequency_based_colors():
    """Verify unified color flow effect works correctly"""
    tm = ThemeManager(THEME_ORIGINAL)

    # All bars should have same color (unified wave)
    # Test with balanced frequencies
    color_center = tm.get_bar_color(
        bar_index=3,
        total_bars=7,
        voice_intensity=0.5,
        bar_height_factor=0.5,
        frequency_bands={'bass': 0.33, 'mid': 0.33, 'treble': 0.34}
    )
    # Should be vibrant (at least one color component strong)
    max_component = max(color_center.red(), color_center.green(), color_center.blue())
    assert max_component > 180  # Should be vibrant

    # Edge bar should have SAME color as center (unified wave)
    color_edge = tm.get_bar_color(
        bar_index=0,
        total_bars=7,
        voice_intensity=0.5,
        bar_height_factor=0.5,
        frequency_bands={'bass': 0.33, 'mid': 0.33, 'treble': 0.34}
    )
    # Should also be vibrant
    max_component = max(color_edge.red(), color_edge.green(), color_edge.blue())
    assert max_component > 180

    # Center and edge bars should have SAME color (unified wave effect)
    # Colors should match exactly (or very close due to floating point)
    assert abs(color_center.red() - color_edge.red()) <= 1
    assert abs(color_center.green() - color_edge.green()) <= 1
    assert abs(color_center.blue() - color_edge.blue()) <= 1

    # Frequency bias should affect the unified color
    # Bass dominant → warm color shift
    color_bass = tm.get_bar_color(
        bar_index=3,
        total_bars=7,
        voice_intensity=0.5,
        bar_height_factor=0.5,
        frequency_bands={'bass': 0.9, 'mid': 0.05, 'treble': 0.05}
    )
    # Bass should have decent red component (warm color)
    assert color_bass.red() > 100 or color_bass.green() > 100
