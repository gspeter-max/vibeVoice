"""Unit tests for src/audio/ear_runtime/menu.py (Terminal Menu UI)."""

from unittest.mock import MagicMock, patch
import pytest

from src.audio.ear_runtime.menu import TerminalMenu


class FakeEarForMenu:
    def __init__(self):
        self.current_model = "parakeet-tdt-0.6b-v3"

    def set_stt_model(self, model_name):
        self.current_model = model_name
        return True


def test_terminal_menu_initialization():
    ear = FakeEarForMenu()
    with patch("sys.stdin.fileno", return_value=0), patch("termios.tcgetattr"):
        menu = TerminalMenu(ear_instance=ear)
        assert menu.ear is ear
