"""Tests for ThemeManager integration into PillHUD."""

import os
import pytest
from PySide6.QtWidgets import QApplication
from src.theme_manager import THEME_ORIGINAL, THEME_RAINBOW, THEME_ANIMATED
from src.hud import PillHUD


class TestHUDThemeIntegration:
    """Test ThemeManager integration in PillHUD widget."""

    def setup_method(self):
        """Create QApplication for tests."""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()

    def test_hud_respects_theme_environment_variable(self):
        """Test that PillHUD reads HUD_THEME environment variable."""
        # Set environment variable to Rainbow theme
        os.environ['HUD_THEME'] = str(THEME_RAINBOW)

        # Create PillHUD instance
        hud = PillHUD()

        # Assert theme manager is initialized with correct theme
        assert hasattr(hud, '_theme_manager'), "PillHUD should have _theme_manager attribute"
        assert hud._theme_manager.current_theme == THEME_RAINBOW, \
            f"Expected theme {THEME_RAINBOW}, got {hud._theme_manager.current_theme}"

    def test_hud_animation_state_for_animated_theme(self):
        """Test that animated theme requires animation."""
        # Set environment variable to Animated theme
        os.environ['HUD_THEME'] = str(THEME_ANIMATED)

        # Create PillHUD instance
        hud = PillHUD()

        # Assert theme manager reports animation required
        assert hasattr(hud, '_theme_manager'), "PillHUD should have _theme_manager attribute"
        assert hud._theme_manager.requires_animation() is True, \
            "Animated theme should require animation"

    def test_hud_default_theme_without_env_var(self):
        """Test that PillHUD defaults to THEME_ORIGINAL when HUD_THEME is not set."""
        # Remove HUD_THEME from environment
        if 'HUD_THEME' in os.environ:
            del os.environ['HUD_THEME']

        # Create PillHUD instance
        hud = PillHUD()

        # Assert theme manager defaults to THEME_ORIGINAL
        assert hasattr(hud, '_theme_manager'), "PillHUD should have _theme_manager attribute"
        assert hud._theme_manager.current_theme == THEME_ORIGINAL, \
            f"Expected default theme {THEME_ORIGINAL}, got {hud._theme_manager.current_theme}"
