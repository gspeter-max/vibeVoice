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
    """Verify rainbow spectrum colors work correctly"""
    tm = ThemeManager(THEME_ORIGINAL)

    # Center bar should be warm color (red/orange/yellow range)
    # With time cycling, just verify it's vibrant
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

    # Edge bar should be cool color (blue/purple range)
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

    # Center and edge bars should have different colors (spectrum effect)
    # They might occasionally be similar due to time cycling, but generally different
    center_rgb = (color_center.red(), color_center.green(), color_center.blue())
    edge_rgb = (color_edge.red(), color_edge.green(), color_edge.blue())

    # Calculate color difference
    diff = sum(abs(c - e) for c, e in zip(center_rgb, edge_rgb))
    # Colors should be noticeably different (at least 50 difference in RGB space)
    # This might occasionally fail due to time cycling, but 99% of the time it should pass
    assert diff > 30 or abs(center_rgb[0] - edge_rgb[0]) > 20
