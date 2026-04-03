import pytest
from PySide6.QtGui import QColor
from theme_manager import ThemeManager, THEME_ORIGINAL
import hud


def test_theme_manager_initialization():
    """Verify ThemeManager initializes correctly"""
    tm = ThemeManager(THEME_ORIGINAL)
    assert tm.current_theme == THEME_ORIGINAL
    assert tm.border_width == 1.2


def test_theme_name():
    """Verify theme name is accessible"""
    assert ThemeManager.theme_name(THEME_ORIGINAL) == "Original (Dark with Premium White Bars)"


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


def test_bar_colors_are_premium_white():
    """Verify bars stay white instead of cycling through colors."""
    tm = ThemeManager(THEME_ORIGINAL)

    color_quiet = tm.get_bar_color(
        bar_index=3,
        total_bars=21,
        voice_intensity=0.0,
        bar_height_factor=0.2,
        frequency_bands={'bass': 0.33, 'mid': 0.33, 'treble': 0.34}
    )
    color_loud = tm.get_bar_color(
        bar_index=10,
        total_bars=21,
        voice_intensity=1.0,
        bar_height_factor=1.0,
        frequency_bands={'bass': 0.9, 'mid': 0.05, 'treble': 0.05}
    )

    for color in (color_quiet, color_loud):
        assert color.red() == 255
        assert color.green() == 255
        assert color.blue() == 255

    assert color_quiet.alpha() < color_loud.alpha()
    assert color_quiet.alpha() >= 150
    assert color_loud.alpha() <= 255


def test_hud_waveform_is_dense_and_compact():
    """Verify the waveform is compact enough for the menu bar."""
    assert hud.NUM_BARS == 9
    assert hud.STATUS_ITEM_W <= 64
    assert hud.STATUS_ITEM_H <= 22
    assert hud.BAR_W <= 2.0
    assert hud.BAR_GAP <= 2.5
    assert hud.BAR_MAX_H >= 15.0
    assert hud.BAR_MIN_H >= 5.0

    total_waveform_width = hud.NUM_BARS * hud.BAR_W + (hud.NUM_BARS - 1) * hud.BAR_GAP
    assert total_waveform_width < hud.STATUS_ITEM_W


def test_hud_declares_monochrome_bar_mode():
    """Verify the HUD exposes an explicit monochrome runtime mode."""
    assert hud.BAR_COLOR_MODE == "monochrome"


def test_hud_runtime_signature_mentions_monochrome():
    """Verify the HUD startup signature advertises the menu-bar renderer."""
    signature = hud.runtime_signature()
    assert "monochrome" in signature
    assert "anchor=menu-bar" in signature
    assert "bars=9" in signature
    assert "wave=chaotic-zigzag" in signature


def test_theme_manager_returns_exact_white_rgb():
    """Verify bar colors are exact white regardless of frequency data."""
    tm = ThemeManager(THEME_ORIGINAL)
    color = tm.get_bar_color(
        bar_index=12,
        total_bars=14,
        voice_intensity=0.7,
        bar_height_factor=0.9,
        frequency_bands={"bass": 0.8, "mid": 0.1, "treble": 0.1},
    )
    assert (color.red(), color.green(), color.blue()) == (255, 255, 255)


def test_hud_bar_color_for_draw_is_strict_white():
    """Verify HUD draw path uses strict white regardless of intensity/height."""
    c1 = hud.bar_color_for_draw(voice_intensity=0.0, bar_height_factor=0.0)
    c2 = hud.bar_color_for_draw(voice_intensity=0.5, bar_height_factor=0.5)
    c3 = hud.bar_color_for_draw(voice_intensity=1.0, bar_height_factor=1.0)

    for color in (c1, c2, c3):
        assert (color.red(), color.green(), color.blue()) == (255, 255, 255)

    assert c1.alpha() <= c2.alpha() <= c3.alpha()
    assert c1.alpha() >= 248
    assert c3.alpha() >= 254


def test_menu_bar_waveform_layout_is_centered():
    """Verify menu-bar waveform bars are centered within the status slot."""
    layout = hud.compute_menu_bar_waveform_layout(
        status_width=hud.STATUS_ITEM_W,
        status_height=hud.STATUS_ITEM_H,
        num_bars=hud.NUM_BARS,
        bar_width=hud.BAR_W,
        bar_gap=hud.BAR_GAP,
        bar_height=hud.BAR_MAX_H,
    )

    assert len(layout) == hud.NUM_BARS
    first = layout[0]
    last = layout[-1]
    center = hud.STATUS_ITEM_W / 2
    assert abs((first["x"] + last["x"] + last["width"]) / 2 - center) < 0.6
    assert all(abs(item["y"] - (hud.STATUS_ITEM_H - hud.BAR_MAX_H) / 2) < 0.6 for item in layout)
