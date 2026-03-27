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
    """Verify bars at different positions get different colors"""
    tm = ThemeManager(THEME_ORIGINAL)

    color_center = tm.get_bar_color(bar_index=3, total_bars=7, voice_intensity=0.5, bar_height_factor=0.5, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})
    color_edge = tm.get_bar_color(bar_index=0, total_bars=7, voice_intensity=0.5, bar_height_factor=0.5, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})

    # Colors should be different
    assert (color_center.red(), color_center.green(), color_center.blue()) != \
           (color_edge.red(), color_edge.green(), color_edge.blue())


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
    """Verify frequency-based color mapping works correctly"""
    tm = ThemeManager(THEME_ORIGINAL)

    # Bass dominant -> warm colors (red/orange)
    # Hue 0.0 = red, 0.1 = orange
    color_bass = tm.get_bar_color(
        bar_index=3,
        total_bars=7,
        voice_intensity=0.5,
        bar_height_factor=0.5,
        frequency_bands={'bass': 0.9, 'mid': 0.1, 'treble': 0.1}
    )
    # Should have high red component for warm colors
    assert color_bass.red() > 150  # High red for bass

    # Treble dominant -> cool colors (blue/purple)
    # Hue 0.6-0.7 = blue/purple
    color_treble = tm.get_bar_color(
        bar_index=3,
        total_bars=7,
        voice_intensity=0.5,
        bar_height_factor=0.5,
        frequency_bands={'bass': 0.1, 'mid': 0.1, 'treble': 0.9}
    )
    # Should have high blue component for cool colors
    assert color_treble.blue() > 150  # High blue for treble

    # Mid dominant -> neutral colors (green/yellow)
    # Hue 0.3-0.4 = green/yellow
    color_mid = tm.get_bar_color(
        bar_index=3,
        total_bars=7,
        voice_intensity=0.5,
        bar_height_factor=0.5,
        frequency_bands={'bass': 0.1, 'mid': 0.9, 'treble': 0.1}
    )
    # Should have high green component for mid
    assert color_mid.green() > 150  # High green for mid
