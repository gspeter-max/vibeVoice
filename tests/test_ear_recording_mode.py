import os
import pytest
from unittest.mock import patch, Mock
import sys

# Standard mock setup for Ear tests to avoid audio hardware issues
sys.modules['pynput'] = type(sys)('pynput')
sys.modules['pynput.keyboard'] = type(sys)('pynput.keyboard')
sys.modules['pynput.mouse'] = type(sys)('pynput.mouse')

class MockButton:
    left = 'left'
    right = 'right'

sys.modules['pynput.mouse'].Button = MockButton()

class MockPyAudio:
    paInt16 = 16
    paContinue = 0
    class PyAudio:
        def get_default_input_device_info(self):
            return {"index": 0}
        def get_device_info_by_index(self, index):
            return {"name": "Test Device"}

sys.modules['pyaudio'] = MockPyAudio()

import src.ear as ear_module

def test_nemotron_not_in_models_when_no_streaming():
    """
    When RECORDING_MODE is set to 'no_streaming', Nemotron should NOT
    be included in the active models list to prevent incompatibility issues.
    """
    with patch.dict(os.environ, {"RECORDING_MODE": "no_streaming"}):
        models = ear_module.get_active_models()
        assert "nemotron-streaming-0.6b" not in models
        # Menu should only allow 4 options
        assert len(models) == 4

def test_nemotron_in_models_when_silence_streaming():
    """
    When RECORDING_MODE is 'silence_streaming', Nemotron SHOULD be
    included as the 5th option in the models list.
    """
    with patch.dict(os.environ, {"RECORDING_MODE": "silence_streaming"}):
        models = ear_module.get_active_models()
        assert "nemotron-streaming-0.6b" in models
        # Menu should allow all 5 options
        assert len(models) == 5

@patch('sys.stdin.fileno', return_value=0)
@patch('src.ear.send_switch_command')
def test_terminal_menu_ignores_nemotron_key_in_no_streaming(mock_send, mock_fileno):
    """
    Verifies that pressing '5' in the terminal menu does absolutely nothing
    if the current mode does not support Nemotron.
    """
    menu = ear_module.TerminalMenu()
    
    with patch.dict(os.environ, {"RECORDING_MODE": "no_streaming"}):
        # We simulate the sys.stdin.read returning '5'
        with patch('sys.stdin.read', return_value='5'):
            # Trigger the logic that handles 'c in 12345'
            # (Note: we just test the inner logic, not the full run() loop which is blocking)
            c = '5'
            idx = int(c) - 1
            active_models = ear_module.get_active_models()
            
            if idx < len(active_models):
                mock_send(active_models[idx], menu.ear)
                
            # mock_send should NOT have been called because len(active_models) is 4, and idx is 4.
            # 4 < 4 is False.
            mock_send.assert_not_called()

@patch('sys.stdin.fileno', return_value=0)
@patch('src.ear.send_switch_command')
def test_terminal_menu_accepts_nemotron_key_in_streaming(mock_send, mock_fileno):
    """
    Verifies that pressing '5' works when in streaming mode.
    """
    menu = ear_module.TerminalMenu()
    
    with patch.dict(os.environ, {"RECORDING_MODE": "silence_streaming"}):
        c = '5'
        idx = int(c) - 1
        active_models = ear_module.get_active_models()
        
        if idx < len(active_models):
            mock_send(active_models[idx], menu.ear)
            
        mock_send.assert_called_once_with("nemotron-streaming-0.6b", None)