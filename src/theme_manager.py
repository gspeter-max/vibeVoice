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
        Get DYNAMIC color for waveform bars with unified color flow.

        ALL bars share the same color, which shifts over time through the spectrum.
        Creates a flowing wave effect where the entire waveform changes color together.

        Args:
            bar_index: Index of this bar (0 to total_bars-1) - NOT used for color
            total_bars: Total number of bars - NOT used for color
            voice_intensity: Voice volume level (0.0 to 1.0)
            bar_height_factor: Normalized bar height (0.0 to 1.0)
            frequency_bands: Dict with 'bass', 'mid', 'treble' values (0.0 to 1.0)

        Returns:
            QColor for the bar (unified flowing color, same for all bars)
        """
        import time

        # Default frequency bands if not provided
        if frequency_bands is None:
            frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.33}

        bass = frequency_bands.get('bass', 0.33)
        treble = frequency_bands.get('treble', 0.33)

        # UNIFIED COLOR FLOW - ALL BARS SAME COLOR
        # Color cycles through entire spectrum over time
        # Creates smooth, flowing wave effect

        # Time-based hue cycling (slow, smooth flow)
        # 0.08 = ~12 seconds for full spectrum cycle
        time_hue = (time.time() * 0.08) % 1.0

        # Frequency-based hue adjustment
        # Bass dominant → cycle toward warm colors (red/orange/yellow)
        # Treble dominant → cycle toward cool colors (blue/purple)
        freq_bias = (bass - treble) * 0.2  # Range: -0.2 to +0.2

        # Combine time cycling with frequency bias
        hue = (time_hue + freq_bias) % 1.0

        # DYNAMIC SATURATION
        # Pulses with voice intensity
        base_saturation = 0.82
        voice_boost = voice_intensity * 0.18

        saturation = base_saturation + voice_boost
        saturation = min(1.0, max(0.70, saturation))

        # DYNAMIC BRIGHTNESS
        # Responds to both voice and bar height
        base_brightness = 0.80
        voice_boost = voice_intensity * 0.08
        height_boost = bar_height_factor * 0.12

        brightness = base_brightness + voice_boost + height_boost
        brightness = min(1.0, max(0.72, brightness))

        # Create the unified color
        color = QColor.fromHsvF(hue, saturation, brightness, 1.0)
        color.setAlpha(255)

        return color
