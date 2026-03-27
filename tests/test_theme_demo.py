import subprocess
import time
import os

def test_theme_demo_cycles_through_all_themes():
    """Demo script should cycle through all 4 themes"""
    result = subprocess.run(
        ['python', 'scripts/theme_demo.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "Theme 0" in result.stdout
    assert "Theme 1" in result.stdout
    assert "Theme 2" in result.stdout
    assert "Theme 3" in result.stdout
