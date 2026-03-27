import os
from typing import Optional

DEFAULT_THEME_CONFIG_PATH = os.path.expanduser("~/.config/parakeet-flow/theme.conf")

def save_theme_preference(theme_id: int, config_path: Optional[str] = None) -> None:
    """Save theme preference to config file.

    Args:
        theme_id: Theme ID (0-3)
        config_path: Optional custom config path
    """
    if config_path is None:
        config_path = DEFAULT_THEME_CONFIG_PATH

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(str(theme_id))

def load_theme_preference(config_path: Optional[str] = None) -> int:
    """Load theme preference from config file.

    Args:
        config_path: Optional custom config path

    Returns:
        Theme ID (0-3), or 0 (THEME_ORIGINAL) if file doesn't exist
    """
    if config_path is None:
        config_path = DEFAULT_THEME_CONFIG_PATH

    if not os.path.exists(config_path):
        return 0

    try:
        with open(config_path, 'r') as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return 0
