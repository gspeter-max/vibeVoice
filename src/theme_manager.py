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
        Get DYNAMIC color for a waveform bar with rainbow spectrum effect.

        Creates a spectrum analyzer where each bar shows a different color
        across the full color spectrum (red → orange → yellow → green → blue → purple).

        Args:
            bar_index: Index of this bar (0 to total_bars-1)
            total_bars: Total number of bars
            voice_intensity: Voice volume level (0.0 to 1.0)
            bar_height_factor: Normalized bar height (0.0 to 1.0)
            frequency_bands: Dict with 'bass', 'mid', 'treble' values (0.0 to 1.0)

        Returns:
            QColor for the bar (dynamic rainbow spectrum, voice-reactive)
        """
        import time

        # Default frequency bands if not provided
        if frequency_bands is None:
            frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.33}

        bass = frequency_bands.get('bass', 0.33)
        mid = frequency_bands.get('mid', 0.33)
        treble = frequency_bands.get('treble', 0.33)

        # RAINBOW SPECTRUM ACROSS BARS
        # Each bar gets a different color across the full spectrum
        # Center bars = warm colors (red/orange/yellow)
        # Edge bars = cool colors (green/blue/purple)

        # Map bar position to hue spectrum
        # Center (bar_index = mid) → 0.0 (red)
        # Edge (bar_index = 0 or max) → 0.7 (purple)
        pos_mid = (total_bars - 1) / 2.0
        pos_from_center = abs(bar_index - pos_mid) / pos_mid if pos_mid > 0 else 0

        # Base hue from position: 0.0 (center/red) to 0.75 (edge/purple)
        # This creates a rainbow gradient from center to edges
        base_hue = pos_from_center * 0.75

        # TIME-BASED COLOR CYCLING
        # Colors shift over time for dynamic effect
        time_shift = (time.time() * 0.15) % 1.0

        # FREQUENCY-BASED HUE SHIFT
        # Bass dominant → shift toward warm colors (red/orange)
        # Treble dominant → shift toward cool colors (blue/purple)
        freq_bias = (bass - treble) * 0.15  # Range: -0.15 to +0.15

        # Combine all hue factors
        hue = (base_hue + time_shift + freq_bias) % 1.0

        # DYNAMIC SATURATION
        # Increases with voice intensity and bar height
        # Also varies slightly by position for visual interest
        base_saturation = 0.85
        voice_boost = voice_intensity * 0.15
        height_boost = bar_height_factor * 0.10
        pos_variation = pos_from_center * 0.05

        saturation = base_saturation + voice_boost + height_boost + pos_variation
        saturation = min(1.0, max(0.70, saturation))

        # DYNAMIC BRIGHTNESS
        # Taller bars = brighter
        # Louder voice = overall brighter
        base_brightness = 0.82
        height_boost = bar_height_factor * 0.13
        voice_boost = voice_intensity * 0.05

        brightness = base_brightness + height_boost + voice_boost
        brightness = min(1.0, max(0.75, brightness))

        # Create the dynamic color
        color = QColor.fromHsvF(hue, saturation, brightness, 1.0)
        color.setAlpha(255)

        return color
