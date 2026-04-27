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


# -----------------------------------------------------------------------------
# Tests for Task 4: start_main_application_in_background
# -----------------------------------------------------------------------------

def test_start_main_application_in_background_creates_process_group():
    """
    Ensures start_main_application_in_background sets the correct environment variables and calls Popen with setsid.
    """
    import verify_startup
    
    with patch('subprocess.Popen') as mock_popen:
        with patch('os.makedirs'):
            with patch('builtins.open', MagicMock()):
                verify_startup.start_main_application_in_background()
                
                mock_popen.assert_called_once()
                args, kwargs = mock_popen.call_args
                
                assert kwargs['env']['RECORDING_MODE'] == "silence_streaming"
                assert kwargs['env']['STREAMING_TELEMETRY_ENABLED'] == "0"
                assert kwargs['preexec_fn'] == os.setsid

def test_start_main_application_in_background_handles_os_error():
    """
    Ensures that if os.makedirs throws an OSError, it raises properly.
    """
    import verify_startup
    
    with patch('os.makedirs', side_effect=OSError("Disk full")):
        with pytest.raises(OSError):
            verify_startup.start_main_application_in_background()


# -----------------------------------------------------------------------------
# Tests for Task 5: Check 1 & Check 2 (Brain PID and Socket)
# -----------------------------------------------------------------------------

def test_check_if_brain_program_is_running_from_file_avoids_stale_files(tmp_path):
    """
    Ensures check_brain_pid reads the integer and checks if the process is alive.
    If the process is dead, it should keep waiting and eventually return False.
    """
    import verify_startup
    
    pid_file = tmp_path / "test-brain.pid"
    pid_file.write_text("999999\n")
    
    # Mock no main app crash
    verify_startup.main_app_process = None
    
    with patch('os.kill') as mock_kill:
        # Mock os.kill to raise ProcessLookupError, meaning process is dead
        mock_kill.side_effect = ProcessLookupError
        
        # Test should fail because the process is not alive, even though file exists
        result = verify_startup.check_if_brain_program_is_running_from_file(timeout_seconds=0.1, pid_path=str(pid_file))
        assert result == False

def test_check_if_brain_program_is_running_fails_fast_on_main_crash():
    """
    Ensures that if the main application process crashes unexpectedly, it returns False immediately.
    """
    import verify_startup
    
    mock_process = MagicMock()
    mock_process.poll.return_value = 1 # Not None means it crashed
    verify_startup.main_app_process = mock_process
    
    result = verify_startup.check_if_brain_program_is_running_from_file(timeout_seconds=10)
    assert result == False

def test_check_if_brain_program_is_ready_to_receive_data_timeout():
    """
    Ensures it times out correctly when the socket is not found.
    """
    import verify_startup
    
    verify_startup.main_app_process = None
    
    with patch('os.path.exists', return_value=False):
        result = verify_startup.check_if_brain_program_is_ready_to_receive_data(timeout_seconds=0.1)
        assert result == False


# -----------------------------------------------------------------------------
# Tests for Task 6, 7: HUD Checks and Pings
# -----------------------------------------------------------------------------

def test_check_if_hud_display_program_is_ready_to_receive_data_refused():
    """
    Ensures HUD TCP check returns False if connection is refused constantly.
    """
    import verify_startup
    verify_startup.main_app_process = None
    
    with patch('socket.socket') as mock_socket:
        mock_instance = mock_socket.return_value
        mock_instance.connect.side_effect = ConnectionRefusedError
        
        result = verify_startup.check_if_hud_display_program_is_ready_to_receive_data(timeout_seconds=0.1)
        assert result == False

def test_send_fake_audio_to_brain_and_see_if_it_survives_success():
    """
    Ensures a successful ping returns True.
    """
    import verify_startup
    
    with patch('socket.socket'):
        with patch('time.sleep'):
            with patch('verify_startup.check_if_brain_program_is_running_from_file', return_value=True):
                result = verify_startup.send_fake_audio_to_brain_and_see_if_it_survives()
                assert result == True

def test_send_fake_command_to_hud_and_see_if_it_survives_crashes():
    """
    Ensures if HUD crashes post-ping, it returns False.
    """
    import verify_startup
    
    with patch('socket.socket'):
        with patch('time.sleep'):
            with patch('verify_startup.check_if_hud_display_program_is_running_from_file', return_value=False):
                result = verify_startup.send_fake_command_to_hud_and_see_if_it_survives()
                assert result == False
