import os
import sys
import pytest

@pytest.fixture(scope="module")
def qapp():
    """Shared QApplication for all tests"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

def test_full_theme_switching_cycle(qapp):
    """Test switching between all themes in runtime"""
    from src.hud import PillHUD
    from src.theme_manager import THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED

    themes = [THEME_ORIGINAL, THEME_RAINBOW, THEME_RADIAL, THEME_ANIMATED]

    for theme_id in themes:
        os.environ['HUD_THEME'] = str(theme_id)
        hud = PillHUD()
        assert hud._theme_manager.current_theme == theme_id
        hud.close()
