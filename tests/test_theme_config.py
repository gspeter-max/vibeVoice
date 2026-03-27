import os
import tempfile
from src.theme_config import save_theme_preference, load_theme_preference

def test_save_and_load_theme_preference():
    """Theme preference should persist to file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
        config_path = f.name

    try:
        save_theme_preference(2, config_path)
        loaded = load_theme_preference(config_path)
        assert loaded == 2
    finally:
        os.unlink(config_path)

def test_missing_config_returns_default():
    """Missing config file should return default theme"""
    loaded = load_theme_preference("/nonexistent/path/theme.conf")
    assert loaded == 0  # THEME_ORIGINAL
