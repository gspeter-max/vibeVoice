"""
theme_manager.py — Dynamic colorful bars for Parakeet Flow HUD
================================================================
Provides theme management with dynamic rainbow-colored bars based on voice input.
"""

from PySide6.QtGui import QColor, QPen, QBrush
from PySide6.QtCore import Qt

# Theme constant
THEME_ORIGINAL = 0

THEME_NAMES = {
    THEME_ORIGINAL: "Original (Dark with Colorful Bars)",
}


class ThemeManager:
    """
    Manages HUD theme with dynamic colorful bars based on voice.
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
        Create a QPen for the border (dark gray).

        Args:
            rect_x, rect_y, rect_w, rect_h: Dimensions and position of rectangle
            hue_offset: Unused (for compatibility)

        Returns:
            QPen with solid gray border
        """
        return QPen(QColor(90, 90, 95, 200), self.border_width)

    def create_background_brush(self, rect_x: float, rect_y: float, rect_h: float, alpha: int) -> QBrush:
        """
        Create a QBrush for the background (dark).

        Args:
            rect_x, rect_y, rect_h: Rectangle dimensions
            alpha: Transparency value (0-255)

        Returns:
            QBrush with dark gray background
        """
        return QBrush(QColor(16, 16, 18, alpha))

    def requires_animation(self) -> bool:
        """
        Check if theme requires bar animation.

        Returns:
            True - bars should be animated with voice
        """
        return True

    def get_bar_color(self, bar_index: int, total_bars: int, voice_intensity: float,
                     bar_height_factor: float) -> QColor:
        """
        Get dynamic color for a waveform bar based on position and voice.

        Args:
            bar_index: Index of this bar (0 to total_bars-1)
            total_bars: Total number of bars
            voice_intensity: Voice volume level (0.0 to 1.0)
            bar_height_factor: Normalized bar height (0.0 to 1.0)

        Returns:
            QColor for the bar (rainbow gradient based on position)
        """
        # Rainbow gradient across bars using HSV
        # Map bar position to hue (pink at center to cyan at edges)
        mid = (total_bars - 1) / 2.0
        pos_from_center = abs(bar_index - mid) / mid if mid > 0 else 0

        # Pink/Magenta at center (hue ~0.85) to Cyan at edges (hue ~0.5)
        # Reverse: center should be smaller hue number
        hue = 0.85 - (pos_from_center * 0.35)  # 0.85 (pink) to 0.5 (cyan)

        # Full saturation and brightness
        color = QColor.fromHsvF(hue, 1.0, 1.0, 1.0)
        color.setAlpha(255)

        return color
