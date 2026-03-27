"""
Shared configuration settings for TTS system.
"""
import yaml
from pathlib import Path


# Load models.yaml
CONFIG_PATH = Path(__file__).parent / "models.yaml"

with open(CONFIG_PATH, 'r') as f:
    MODELS_CONFIG = yaml.safe_load(f)


def get_model_config(model_name: str) -> dict:
    """Get configuration for specific model."""
    return MODELS_CONFIG['models'][model_name]


def get_socket_path(model_name: str) -> str:
    """Get Unix socket path for model."""
    return get_model_config(model_name)['socket']
