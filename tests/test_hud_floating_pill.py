import pytest
from unittest.mock import MagicMock, patch
import src.ui.hud as hud

def test_hud_constants_exist():
    """Verify that the new HUD constants are defined."""
    assert hud.INDICATOR_WIDTH == 100
    assert hud.INDICATOR_HEIGHT == 26
    assert hud.STATE_HIDDEN == "HIDDEN"
    assert hud.STATE_LISTENING == "LISTENING"

def test_widget_initialization():
    """Verify the RoundedRectangularIndicatorWidget initializes with correct flags."""
    # We need a QApplication for UI tests
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    
    widget = hud.RoundedRectangularIndicatorWidget()
    assert widget.width() == hud.INDICATOR_WIDTH
    assert widget.height() == hud.INDICATOR_HEIGHT
    assert widget._interface_state == hud.STATE_HIDDEN

def test_controller_interface_commands():
    """Verify OscillatingInterfaceController routes commands to the widget."""
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    
    with patch("src.ui.hud.RoundedRectangularIndicatorWidget") as MockWidget:
        mock_widget_instance = MockWidget.return_value
        controller = hud.OscillatingInterfaceController()
        
        # Test 'listen' command
        controller.on_interface_command("listen", 0.5)
        mock_widget_instance.update_interface_state.assert_called_with(hud.STATE_LISTENING, 0.5)
        
        # Test 'process' command
        controller.on_interface_command("process")
        mock_widget_instance.update_interface_state.assert_called_with(hud.STATE_PROCESSING)
        
        # Test 'hide' command
        controller.on_interface_command("hide")
        mock_widget_instance.update_interface_state.assert_called_with(hud.STATE_HIDDEN)

def test_widget_state_transitions():
    """Verify widget internal state updates correctly."""
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    
    widget = hud.RoundedRectangularIndicatorWidget()
    widget.update_interface_state(hud.STATE_LISTENING, 0.8)
    assert widget._interface_state == hud.STATE_LISTENING
    assert widget._base_amplitude == 0.8
    
    widget.update_interface_state(hud.STATE_THINKING)
    assert widget._interface_state == hud.STATE_THINKING
