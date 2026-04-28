import logging
from unittest.mock import MagicMock, patch
import pytest

def test_hud_logs_follow_pulse_format(capsys):
    """
    Verify HUD logs only state changes in the arrow format.
    """
    # Mock dependencies as in existing plan
    with patch("src.ui.hud.QTimer"), \
         patch("src.ui.hud.IPCServer"), \
         patch("src.ui.hud.VolumeListener"), \
         patch("src.ui.hud.NSStatusBar"), \
         patch("src.ui.hud.MenuBarWaveformView"):
        import src.ui.hud as hud
        controller = hud.MenuBarWaveformController.__new__(hud.MenuBarWaveformController)
        controller._snd_listen = "dummy.mp3"
        controller._snd_done = "dummy.wav"
        
        # Manually initialize required internal state to safely bypass __init__
        controller._state = hud.HIDDEN
        controller._timer = MagicMock()
        
        controller.show_listening()
        controller.show_done()
        controller._on_volume(0.5) # Should be silent
            
        captured = capsys.readouterr()
        stdout = captured.out
        
        assert "[HUD]   → Listening" in stdout
        assert "[HUD]   → Done" in stdout
        assert "Received volume" not in stdout
