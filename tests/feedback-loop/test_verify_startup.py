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

def test_print_last_few_lines_of_error_log_reads_last_lines(tmp_path):
    """
    Ensures print_last_few_lines_of_error_log reads the correct number of lines from the end of a file.
    """
    import verify_startup
    
    test_log_file = tmp_path / "test.log"
    test_log_file.write_text("line1\nline2\nline3\nline4\n")
    
    with patch('builtins.print') as mock_print:
        verify_startup.print_last_few_lines_of_error_log(str(test_log_file), lines_to_read=2)
        
        # Should print the start banner, line3, line4, and the end banner
        assert mock_print.call_count == 4
        mock_print.assert_any_call("line3")
        mock_print.assert_any_call("line4")
