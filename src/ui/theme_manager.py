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
    Orchestrates the visual style and branding of the HUD elements.
    The ThemeManager centralizes all color, pen, and brush definitions
    to ensure a consistent look and feel across the application. It
    provides a premium monochrome aesthetic, focusing on high-contrast
    white elements against a dark background to minimize visual
    distraction while the user is dictated to their computer.
    """

    def __init__(self, theme_id: int):
        """
        Initializes the theme manager with a specific style identifier.
        Currently, the system defaults to a premium monochrome theme,
        but the constructor is designed to support future visual modes.
        It also sets default parameters like border width, ensuring that
        the UI components have a consistent structural appearance from
         the moment the HUD is first rendered on the screen.
        """
        # Only theme 0 is supported
        self.current_theme = THEME_ORIGINAL
        self.border_width = 1.2

    @staticmethod
    def theme_name(theme_id: int) -> str:
        """
        Translates a theme identifier into a human-readable name.
        This is primarily used for logging and developer debugging,
        allowing the system to report exactly which visual style is
        currently active. If an unrecognized ID is provided, it safely
        returns 'Unknown', preventing any crashes when new themes are
        being tested or if configuration files become corrupted.
        """
        return THEME_NAMES.get(theme_id, "Unknown")

    def create_border_pen(self, rect_x: float, rect_y: float, rect_w: float, rect_h: float, hue_offset: float = 0.0) -> QPen:
        """
        Creates a drawing tool for rendering UI component borders.
        It returns a QPen object configured with a specific gray color
        and the manager's default border width. This tool is used by
        the rendering engine to draw the outlines of the HUD elements,
        ensuring they are clearly visible against different background
        windows while maintaining the project's minimalist aesthetic.
        """
        return QPen(QColor(108, 108, 114, 210), self.border_width)

    def create_background_brush(self, rect_x: float, rect_y: float, rect_h: float, alpha: int) -> QBrush:
        """
        Generates a fill tool for rendering the background of HUD elements.
        The resulting QBrush uses a deep dark-gray color with a variable
        transparency level. This allows the HUD to have a sophisticated
        'glass' effect, where the user can still see a hint of the windows
        behind the HUD, making it feel integrated into the macOS desktop.
        """
        return QBrush(QColor(14, 14, 16, alpha))

    def requires_animation(self) -> bool:
        """
        Determines if the active theme should use dynamic animations.
        For the monochrome theme, this always returns True, signaling
        that the waveform bars should move in response to the user's
        voice. This centralized check allows the UI engine to skip
        expensive animation calculations for static themes, though
        no such themes are currently implemented in this version.
        """
        return True

    def get_bar_color(self, bar_index: int, total_bars: int, voice_intensity: float,
                     bar_height_factor: float, frequency_bands: dict = None) -> QColor:
        """
        Calculates the dynamic color for an individual audio waveform bar.
        While all bars remain white to maintain the monochrome theme,
        the function adjusts the transparency based on the volume and
        the current height of each bar. This creates a shimmering effect
        that gives the user immediate visual feedback on the quality
        and intensity of their microphone input as they are speaking.
        """
        del bar_index, total_bars, frequency_bands

        alpha = int(160 + voice_intensity * 50 + bar_height_factor * 35)
        alpha = max(150, min(255, alpha))
        return QColor(255, 255, 255, alpha)
