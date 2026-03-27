"""
ThemeManager Module - Manages visual themes for PillHUD widget.

Provides 4 gradient themes:
- THEME_ORIGINAL: Simple solid gray border (default)
- THEME_RAINBOW: Diagonal rainbow gradient
- THEME_RADIAL: Radial gradient from center
- THEME_ANIMATED: Animated HSV color gradient
"""

from PySide6.QtGui import QPen, QBrush, QColor, QLinearGradient, QRadialGradient
from PySide6.QtCore import Qt

# Theme ID constants
THEME_ORIGINAL = 0
THEME_RAINBOW = 1
THEME_RADIAL = 2
THEME_ANIMATED = 3

# Human-readable theme names
THEME_NAMES = {
    THEME_ORIGINAL: "Original (Solid Gray)",
    THEME_RAINBOW: "Rainbow Gradient",
    THEME_RADIAL: "Radial Glow",
    THEME_ANIMATED: "Animated Aurora",
}


class ThemeManager:
    """Manages visual themes for PillHUD with gradient borders."""

    def __init__(self, theme_id: int):
        """
        Initialize ThemeManager with a specific theme.

        Args:
            theme_id: Theme ID (THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED)
                     Invalid IDs fall back to THEME_ORIGINAL
        """
        # Validate theme_id
        if theme_id not in THEME_NAMES:
            theme_id = THEME_ORIGINAL

        self.current_theme = theme_id

        # Set border width based on theme type
        if theme_id == THEME_ORIGINAL:
            self.border_width = 1.2
        else:
            # Gradient themes need thicker borders
            self.border_width = 2.5

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
        Create a QPen for the border based on current theme.

        Args:
            rect_x: Rectangle X coordinate
            rect_y: Rectangle Y coordinate
            rect_w: Rectangle width
            rect_h: Rectangle height
            hue_offset: Hue offset for animated themes (0.0 to 1.0)

        Returns:
            QPen configured with theme-appropriate brush
        """
        if self.current_theme == THEME_ORIGINAL:
            # Simple solid gray border
            return QPen(QColor(90, 90, 95, 200), self.border_width)

        elif self.current_theme == THEME_RAINBOW:
            # Diagonal rainbow gradient with 5 stops
            gradient = QLinearGradient(rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)

            # Pink at top-left
            gradient.setColorAt(0.0, QColor(255, 105, 180, 255))  # HotPink
            # Purple
            gradient.setColorAt(0.25, QColor(147, 112, 219, 255))  # MediumPurple
            # Blue
            gradient.setColorAt(0.5, QColor(65, 105, 225, 255))  # RoyalBlue
            # Cyan
            gradient.setColorAt(0.75, QColor(0, 191, 255, 255))  # DeepSkyBlue
            # Yellow at bottom-right
            gradient.setColorAt(1.0, QColor(255, 255, 0, 255))  # Yellow

            pen = QPen(gradient)
            pen.setWidthF(self.border_width)
            return pen

        elif self.current_theme == THEME_RADIAL:
            # Radial gradient from center
            center_x = rect_x + rect_w / 2
            center_y = rect_y + rect_h / 2
            radius = max(rect_w, rect_h) / 2

            gradient = QRadialGradient(center_x, center_y, radius)

            # Yellow at center
            gradient.setColorAt(0.0, QColor(255, 255, 0, 255))  # Yellow
            # Magenta in middle
            gradient.setColorAt(0.5, QColor(255, 0, 255, 255))  # Magenta
            # Blue at edges
            gradient.setColorAt(1.0, QColor(0, 0, 255, 255))  # Blue

            pen = QPen(gradient)
            pen.setWidthF(self.border_width)
            return pen

        elif self.current_theme == THEME_ANIMATED:
            # Animated gradient with 6 HSV colors using hue_offset
            gradient = QLinearGradient(rect_x, rect_y, rect_x + rect_w, rect_y)

            # 6 color stops with HSV colors, shifted by hue_offset
            for i in range(6):
                hue = (hue_offset + i / 6.0) % 1.0
                # Convert HSV to RGB with full saturation and value
                color = QColor()
                color.setHsvF(hue, 1.0, 1.0, 1.0)
                position = i / 5.0
                gradient.setColorAt(position, color)

            pen = QPen(gradient)
            pen.setWidthF(self.border_width)
            return pen

        # Fallback (shouldn't reach here)
        pen = QPen(QColor(90, 90, 95, 200))
        pen.setWidthF(self.border_width)
        return pen

    def create_background_brush(self, rect_x: float, rect_y: float, rect_h: float, alpha: int) -> QBrush:
        """
        Create a QBrush for the background based on current theme.

        Args:
            rect_x: Rectangle X coordinate
            rect_y: Rectangle Y coordinate
            rect_h: Rectangle height
            alpha: Alpha value (0-255) for transparency

        Returns:
            QBrush configured with theme-appropriate background
        """
        if self.current_theme == THEME_ORIGINAL:
            # Simple solid dark background
            color = QColor(16, 16, 18, alpha)
            return QBrush(color)

        else:
            # Gradient themes get a vertical gradient background
            gradient = QLinearGradient(rect_x, rect_y, rect_x, rect_y + rect_h)

            # Darker at top
            gradient.setColorAt(0.0, QColor(40, 40, 50, alpha))
            # Even darker at bottom
            gradient.setColorAt(1.0, QColor(20, 20, 30, alpha))

            return QBrush(gradient)

    def requires_animation(self) -> bool:
        """
        Check if current theme requires animation.

        Returns:
            True only for THEME_ANIMATED
        """
        return self.current_theme == THEME_ANIMATED
