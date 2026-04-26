import pytest
import os
import signal
from unittest.mock import patch, MagicMock

# We will import the module under test after we write it
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../feedback-loop')))

def test_stop_all_programs_and_exit_script_calls_killpg():
    """
    Ensures that when stop_all_programs_and_exit_script is called, it attempts to kill the process group.
    """
    import verify_startup
    
    mock_process = MagicMock()
    mock_process.pid = 12345
    verify_startup.main_app_process = mock_process
    
    with patch('os.getpgid', return_value=54321) as mock_getpgid:
        with patch('os.killpg') as mock_killpg:
            with patch('sys.exit') as mock_exit:
                verify_startup.stop_all_programs_and_exit_script(0)
                
                mock_getpgid.assert_called_once_with(12345)
                mock_killpg.assert_called_once_with(54321, signal.SIGTERM)
                mock_exit.assert_called_once_with(0)
