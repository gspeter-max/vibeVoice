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
                     bar_height_factor: float, frequency_bands: dict = None) -> QColor:
        """
        Get DYNAMIC color for a waveform bar based on voice intensity and frequency bands.

        Args:
            bar_index: Index of this bar (0 to total_bars-1)
            total_bars: Total number of bars
            voice_intensity: Voice volume level (0.0 to 1.0)
            bar_height_factor: Normalized bar height (0.0 to 1.0)
            frequency_bands: Dict with 'bass', 'mid', 'treble' values (0.0 to 1.0)

        Returns:
            QColor for the bar (frequency-based, voice-reactive colors)
        """
        import time

        # Default frequency bands if not provided
        if frequency_bands is None:
            frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.33}

        bass = frequency_bands.get('bass', 0.33)
        mid = frequency_bands.get('mid', 0.33)
        treble = frequency_bands.get('treble', 0.33)

        # FREQUENCY-BASED COLOR MAPPING
        # Bass (low frequencies) -> Warm colors (red/orange, hue 0.0-0.12)
        # Mid (medium frequencies) -> Neutral colors (green/yellow, hue 0.28-0.40)
        # Treble (high frequencies) -> Cool colors (blue/purple, hue 0.58-0.78)

        # Find dominant frequency band
        if bass >= mid and bass >= treble:
            # Bass dominant -> warm red/orange
            base_hue = 0.0 + (bass * 0.12)  # 0.0 to 0.12
        elif mid >= bass and mid >= treble:
            # Mid dominant -> green/yellow
            base_hue = 0.28 + (mid * 0.12)  # 0.28 to 0.40
        else:
            # Treble dominant -> blue/purple
            base_hue = 0.58 + (treble * 0.20)  # 0.58 to 0.78

        # Add subtle time-based cycling (very subtle, just 5% variation)
        time_hue = (time.time() * 0.1) % 0.05

        # Bar position adds very slight variation
        pos_mid = (total_bars - 1) / 2.0
        pos_from_center = abs(bar_index - pos_mid) / pos_mid if pos_mid > 0 else 0
        pos_hue = pos_from_center * 0.02

        # Combine: frequency dominates, time and position add subtle shifts
        hue = (base_hue + time_hue + pos_hue) % 1.0

        # Saturation increases with voice intensity (quiet = more pastel, loud = vibrant)
        # Minimum saturation of 0.75 even with no voice
        saturation = 0.75 + (voice_intensity * 0.25)
        saturation = min(1.0, saturation)

        # Brightness increases with bar height and voice
        # Minimum brightness of 0.85
        brightness = 0.85 + (bar_height_factor * 0.10) + (voice_intensity * 0.05)
        brightness = min(1.0, brightness)

        # Create the dynamic color
        color = QColor.fromHsvF(hue, saturation, brightness, 1.0)
        color.setAlpha(255)

        return color
