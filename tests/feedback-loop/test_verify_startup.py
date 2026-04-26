import pytest
import os
import signal
from unittest.mock import patch, MagicMock

# We will import the module under test after we write it
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../feedback-loop')))

# -----------------------------------------------------------------------------
# Tests for Task 2: stop_all_programs_and_exit_script
# -----------------------------------------------------------------------------

def test_stop_all_programs_and_exit_script_calls_killpg():
    """
    Ensures that when stop_all_programs_and_exit_script is called normally, 
    it attempts to gracefully kill the process group.
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
                mock_process.wait.assert_called_once_with(timeout=5)
                mock_exit.assert_called_once_with(0)

def test_stop_all_programs_and_exit_script_when_no_process_running():
    """
    Ensures that if there is no main application process running, 
    it just exits without trying to kill anything.
    """
    import verify_startup
    
    verify_startup.main_app_process = None
    
    with patch('os.getpgid') as mock_getpgid:
        with patch('os.killpg') as mock_killpg:
            with patch('sys.exit') as mock_exit:
                verify_startup.stop_all_programs_and_exit_script(1)
                
                mock_getpgid.assert_not_called()
                mock_killpg.assert_not_called()
                mock_exit.assert_called_once_with(1)

def test_stop_all_programs_and_exit_script_when_soft_kill_fails_uses_hard_kill():
    """
    Ensures that if the graceful kill (SIGTERM) or wait fails, 
    it falls back to force killing (SIGKILL).
    """
    import verify_startup
    
    mock_process = MagicMock()
    mock_process.pid = 12345
    # Make wait throw an exception to simulate failure
    mock_process.wait.side_effect = Exception("Process refused to die")
    verify_startup.main_app_process = mock_process
    
    with patch('os.getpgid', return_value=54321):
        with patch('os.killpg') as mock_killpg:
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print'):
                    verify_startup.stop_all_programs_and_exit_script(2)
                
                # Should have been called twice: first with SIGTERM, then with SIGKILL
                assert mock_killpg.call_count == 2
                mock_killpg.assert_any_call(54321, signal.SIGTERM)
                mock_killpg.assert_any_call(54321, signal.SIGKILL)
                mock_exit.assert_called_once_with(2)

def test_stop_safely_when_user_presses_control_c():
    """
    Ensures the signal handler calls the exit script with error code 1.
    """
    import verify_startup
    
    with patch('verify_startup.stop_all_programs_and_exit_script') as mock_stop:
        with patch('builtins.print'):
            verify_startup.stop_safely_when_user_presses_control_c(signal.SIGINT, None)
            
            mock_stop.assert_called_once_with(1)


# -----------------------------------------------------------------------------
# Tests for Task 3: print_last_few_lines_of_error_log
# -----------------------------------------------------------------------------

def test_print_last_few_lines_of_error_log_reads_last_lines(tmp_path):
    """
    Ensures print_last_few_lines_of_error_log reads the correct number of lines from the end of a file.
    """
    import verify_startup
    
    test_log_file = tmp_path / "test.log"
    test_log_file.write_text("line1\nline2\nline3\nline4\n")
    
    with patch('builtins.print') as mock_print:
        verify_startup.print_last_few_lines_of_error_log(str(test_log_file), lines_to_read=2)
        
        assert mock_print.call_count == 4
        mock_print.assert_any_call("\n--- START LOG DUMP: " + str(test_log_file) + " ---")
        mock_print.assert_any_call("line3")
        mock_print.assert_any_call("line4")
        mock_print.assert_any_call("--- END LOG DUMP: " + str(test_log_file) + " ---\n")

def test_print_last_few_lines_of_error_log_when_file_does_not_exist():
    """
    Ensures it handles non-existent files gracefully without crashing.
    """
    import verify_startup
    
    fake_file_path = "/this/file/does/not/exist.log"
    
    with patch('builtins.print') as mock_print:
        verify_startup.print_last_few_lines_of_error_log(fake_file_path)
        
        mock_print.assert_called_once_with(f"\n--- LOG DUMP FAILED: {fake_file_path} not found ---")

def test_print_last_few_lines_of_error_log_when_read_fails(tmp_path):
    """
    Ensures it handles permission errors or read failures gracefully.
    """
    import verify_startup
    
    test_log_file = tmp_path / "unreadable.log"
    test_log_file.write_text("some content")
    
    # Mock builtins.open to throw an exception
    with patch('builtins.open', side_effect=PermissionError("Permission denied")):
        with patch('builtins.print') as mock_print:
            verify_startup.print_last_few_lines_of_error_log(str(test_log_file))
            
            # The exception should be caught and printed
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            assert "LOG DUMP FAILED: Could not read" in output
            assert "Permission denied" in output