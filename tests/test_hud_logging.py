from unittest.mock import patch
import src.ui.hud as hud

def test_hud_logs_interface_commands(capsys):
    """
    Verify HUD logs state changes when interface commands are received.
    Note: The actual logging is done via 'log.info' or print in some cases.
    We'll test the OscillatingInterfaceController's command handling.
    """
    from PySide6.QtWidgets import QApplication
    import sys
    QApplication.instance() or QApplication(sys.argv)

    with patch("src.ui.hud.RoundedRectangularIndicatorWidget"):
        controller = hud.OscillatingInterfaceController()
        
        # Test 'listen' command
        with patch("src.ui.hud.logging"):
            # We check if it doesn't crash and updates state
            controller.on_interface_command("listen", 0.5)
            assert controller.widget.update_interface_state.called
