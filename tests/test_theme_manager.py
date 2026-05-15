from PySide6.QtGui import QColor
from src.ui.theme_manager import ThemeManager, THEME_ORIGINAL

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
    assert color.red() == 255
    assert color.green() == 255
    assert color.blue() == 255

    # Test edge bar with full voice
    color = tm.get_bar_color(bar_index=0, total_bars=7, voice_intensity=1.0, bar_height_factor=1.0, frequency_bands={'bass': 0.3, 'mid': 0.3, 'treble': 0.3})
    assert isinstance(color, QColor)
    assert color.alpha() > 0


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
