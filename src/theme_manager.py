"""
theme_manager.py — Premium monochrome bars for Parakeet Flow HUD
================================================================
Provides theme management for a dark HUD with premium white waveform bars.
"""

from PySide6.QtGui import QColor, QPen, QBrush
from PySide6.QtCore import Qt

# Theme constant
THEME_ORIGINAL = 0

THEME_NAMES = {
    THEME_ORIGINAL: "Original (Dark with Premium White Bars)",
}


class ThemeManager:
    """
    Manages the HUD theme for a premium monochrome waveform.
    """

    def __init__(self, theme_id: int):
        """
        Initialize ThemeManager.

        Args:
            theme_id: Theme ID (only 0 supported)
        """
        # Only theme 0 is supported
        self.current_theme = THEME_ORIGINAL
        self.border_width = 1.2

    @staticmethod
    def theme_name(theme_id: int) -> str:
        """
        Get human-readable name for a theme.

        Args:
            theme_id: Theme ID

        Returns:
            Theme name string, or "Unknown" if invalid
        """
        return THEME_NAMES.get(theme_id, "Unknown")

    def create_border_pen(self, rect_x: float, rect_y: float, rect_w: float, rect_h: float, hue_offset: float = 0.0) -> QPen:
        """
        Create a QPen for the border.

        Args:
            rect_x, rect_y, rect_w, rect_h: Dimensions and position of rectangle
            hue_offset: Unused (for compatibility)

        Returns:
            QPen with solid gray border
        """
        return QPen(QColor(108, 108, 114, 210), self.border_width)

    def create_background_brush(self, rect_x: float, rect_y: float, rect_h: float, alpha: int) -> QBrush:
        """
        Create a QBrush for the background.

        Args:
            rect_x, rect_y, rect_h: Rectangle dimensions
            alpha: Transparency value (0-255)

        Returns:
            QBrush with dark gray background
        """
        return QBrush(QColor(14, 14, 16, alpha))

    def requires_animation(self) -> bool:
        """
        Check if theme requires bar animation.

        Returns:
            True - bars should be animated with voice
        """
        return True

    def get_bar_color(self, bar_index: int, total_bars: int, voice_intensity: float,
                     bar_height_factor: float, frequency_bands: dict = None) -> QColor:
        """
        Get a premium monochrome color for waveform bars.

        All bars stay white. Only alpha changes to add depth without introducing
        rainbow or hue-shifting effects.

        Args:
            bar_index: Index of this bar (0 to total_bars-1) - NOT used for color
            total_bars: Total number of bars - NOT used for color
            voice_intensity: Voice volume level (0.0 to 1.0)
            bar_height_factor: Normalized bar height (0.0 to 1.0)
            frequency_bands: Dict with 'bass', 'mid', 'treble' values (0.0 to 1.0)

        Returns:
            QColor for the bar
        """
        del bar_index, total_bars, frequency_bands

        alpha = int(160 + voice_intensity * 50 + bar_height_factor * 35)
        alpha = max(150, min(255, alpha))
        return QColor(255, 255, 255, alpha)
