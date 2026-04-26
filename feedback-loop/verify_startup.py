#!/usr/bin/env python3
import os
import sys
import time
import socket
import signal
import subprocess

# Global variable to hold the main application process
main_app_process = None

def stop_all_programs_and_exit_script(exit_code: int):
    """
    Cleans up all processes started by this script and exits.
    
    Step-by-step logic:
    1. Check if we have a main application process running.
    2. If it is running, find its process group ID. The process group ID helps us kill the main script and all other small scripts it started.
    3. Send a termination signal to the whole process group.
    4. Wait for up to 5 seconds to let them close safely.
    5. If they do not close, force kill them.
    6. Exit this script with the provided exit code.
    """
    global main_app_process
    print("\n[Cleanup] Stopping all application processes...")
    
    if main_app_process is not None:
        try:
            # Get the process group ID of the main application process
            process_group_id = os.getpgid(main_app_process.pid)
            
            # Send a soft kill signal to the entire process group
            os.killpg(process_group_id, signal.SIGTERM)
            
            # Wait for the main application process to finish
            main_app_process.wait(timeout=5)
        except Exception as error:
            print(f"[Cleanup] Force killing failed with error: {error}")
            try:
                # If soft kill fails, send a hard kill signal to the entire process group
                os.killpg(process_group_id, signal.SIGKILL)
            except Exception:
                pass
                
    print("[Cleanup] Cleanup complete.")
    sys.exit(exit_code)

def stop_safely_when_user_presses_control_c(signal_number, frame):
    """
    Catches the user pressing CTRL+C on the keyboard.
    When this happens, it makes sure to run our cleanup logic safely before stopping.
    """
    print("\n[Interrupt] Caught interrupt signal, starting cleanup...")
    stop_all_programs_and_exit_script(1)

def print_last_few_lines_of_error_log(file_path: str, lines_to_read: int = 30):
    """
    Reads the last few lines of a log file and prints them.
    This helps the developer see exactly why a component failed without searching manually.
    
    Step-by-step logic:
    1. Check if the log file exists. If not, print a failure message.
    2. Open the file and read all the lines inside it.
    3. Take only the last few lines (by default, 30 lines).
    4. Print these lines clearly with borders so they are easy to read.
    """
    if not os.path.exists(file_path):
        print(f"\n--- LOG DUMP FAILED: {file_path} not found ---")
        return
        
    try:
        # Open the file in read mode
        with open(file_path, "r", encoding="utf-8") as file_object:
            all_lines = file_object.readlines()
            
            # Get the last N lines
            last_lines = all_lines[-lines_to_read:]
            
            print(f"\n--- START LOG DUMP: {file_path} ---")
            for line in last_lines:
                # Remove extra blank lines at the end of each log line
                print(line.rstrip())
            print(f"--- END LOG DUMP: {file_path} ---\n")
    except Exception as error:
        print(f"\n--- LOG DUMP FAILED: Could not read {file_path}. Error: {error} ---")

# Tell the system to call our safe stop function when it receives an interrupt or termination signal
signal.signal(signal.SIGINT, stop_safely_when_user_presses_control_c)
signal.signal(signal.SIGTERM, stop_safely_when_user_presses_control_c)

if __name__ == "__main__":
    print("Startup Verifier initialized.")

def start_main_application_in_background():
    """
    Starts the start.sh script in a background process.
    
    Step-by-step logic:
    1. Copy current environment variables.
    2. Set RECORDING_MODE to silence_streaming and STREAMING_TELEMETRY_ENABLED to 0 to avoid user prompts.
    3. Create the 'logs' directory if it doesn't already exist.
    4. Open a subprocess for start.sh, saving its output to logs/startup_verify.log.
    5. Set preexec_fn=os.setsid to start it in a new process group so we can kill all children easily.
    """
    global main_app_process
    
    environment_variables = os.environ.copy()
    environment_variables["RECORDING_MODE"] = "silence_streaming"
    environment_variables["STREAMING_TELEMETRY_ENABLED"] = "0"
    
    startup_log_path = "logs/startup_verify.log"
    os.makedirs("logs", exist_ok=True)
    
    print(f"[Start] Launching start.sh (logs saved to {startup_log_path})")
    log_file_handle = open(startup_log_path, "w")
    
    main_app_process = subprocess.Popen(
        ["./start.sh"],
        env=environment_variables,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    return startup_log_path

def check_if_brain_program_is_running_from_file(timeout_seconds: int = 15, pid_path: str = "/tmp/parakeet-brain.pid") -> bool:
    """
    Waits for the Brain PID file to appear and verifies the process is actually running.
    
    Step-by-step logic:
    1. Wait in a loop for up to 'timeout_seconds'.
    2. Check if the main process crashed. If yes, stop waiting and fail fast.
    3. If the PID file is found, read the PID from it.
    4. Check if the PID is alive using a system check (os.kill(pid, 0)).
    5. If alive, return True. If dead or not found, keep waiting.
    """
    print(f"[Check] Waiting for Brain PID file at {pid_path}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        global main_app_process
        if main_app_process and main_app_process.poll() is not None:
            print("[Error] ❌ main application process crashed unexpectedly.")
            return False

        if os.path.exists(pid_path):
            try:
                with open(pid_path, "r") as pid_file:
                    pid_string = pid_file.read().strip()
                    if pid_string.isdigit():
                        process_id = int(pid_string)
                        os.kill(process_id, 0)
                        print(f"[Check] ✅ Brain PID {process_id} is alive.")
                        return True
            except (ProcessLookupError, FileNotFoundError, ValueError):
                pass
        time.sleep(1)
        
    print("[Error] ❌ Brain PID file never created or process never started.")
    return False

def check_if_brain_program_is_ready_to_receive_data(timeout_seconds: int = 30) -> bool:
    """
    Waits for the Brain UNIX socket to appear, proving it can accept connections.
    """
    socket_path = "/tmp/parakeet.sock"
    print(f"[Check] Waiting for Brain socket at {socket_path}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        global main_app_process
        if main_app_process and main_app_process.poll() is not None:
            print("[Error] ❌ main application process crashed unexpectedly.")
            return False

        if os.path.exists(socket_path):
            print("[Check] ✅ Brain socket found.")
            return True
        time.sleep(1)
        
    print("[Error] ❌ Brain socket never created.")
    return False

def check_if_hud_display_program_is_running_from_file(timeout_seconds: int = 15, pid_path: str = "/tmp/parakeet-hud.pid") -> bool:
    """
    Waits for the HUD PID file to appear and verifies the process is actually running.
    """
    print(f"[Check] Waiting for HUD PID file at {pid_path}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        global main_app_process
        if main_app_process and main_app_process.poll() is not None:
            print("[Error] ❌ main application process crashed unexpectedly.")
            return False

        if os.path.exists(pid_path):
            try:
                with open(pid_path, "r") as pid_file:
                    pid_string = pid_file.read().strip()
                    if pid_string.isdigit():
                        process_id = int(pid_string)
                        os.kill(process_id, 0)
                        print(f"[Check] ✅ HUD PID {process_id} is alive.")
                        return True
            except (ProcessLookupError, FileNotFoundError, ValueError):
                pass 
        time.sleep(1)
        
    print("[Error] ❌ HUD PID file never created or process never started.")
    return False

def check_if_hud_display_program_is_ready_to_receive_data(timeout_seconds: int = 10) -> bool:
    """
    Tries to connect to the HUD TCP port to verify it is listening and accepting requests.
    """
    port = 57234
    print(f"[Check] Waiting for HUD TCP port {port}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        global main_app_process
        if main_app_process and main_app_process.poll() is not None:
            print("[Error] ❌ main application process crashed unexpectedly.")
            return False

        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)
            test_socket.connect(("127.0.0.1", port))
            test_socket.close()
            print(f"[Check] ✅ HUD TCP port {port} is open.")
            return True
        except ConnectionRefusedError:
            time.sleep(1)
        except Exception:
            time.sleep(1)
            
    print(f"[Error] ❌ HUD TCP port {port} never opened.")
    return False

def send_fake_audio_to_brain_and_see_if_it_survives() -> bool:
    """
    Connects to the Brain socket and sends dummy audio data.
    Crucially, it verifies the Brain PID is still alive after receiving data,
    proving it did not crash from a bad response.
    """
    socket_path = "/tmp/parakeet.sock"
    print("[Ping] Pinging Brain socket...")
    try:
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        client_socket.connect(socket_path)
        
        # Send 1 second of dummy 16-bit PCM silence
        dummy_data = b"\x00" * 32000 
        client_socket.sendall(dummy_data)
        client_socket.close()
        
        # Give the process a moment to process the data
        time.sleep(1.0)
        
        if not check_if_brain_program_is_running_from_file(timeout_seconds=1):
            print("[Error] ❌ Brain crashed immediately after receiving audio data.")
            return False
            
        print("[Ping] ✅ Brain accepted data and survived.")
        return True
    except Exception as error:
        print(f"[Error] ❌ Brain ping failed: {error}")
        return False

def send_fake_command_to_hud_and_see_if_it_survives() -> bool:
    """
    Connects to the HUD TCP socket and sends a 'listen' command.
    Verifies the HUD PID is still alive afterward.
    """
    port = 57234
    print("[Ping] Pinging HUD TCP port...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        client_socket.connect(("127.0.0.1", port))
        client_socket.sendall(b"listen\n")
        client_socket.close()
        
        time.sleep(1.0)
        
        if not check_if_hud_display_program_is_running_from_file(timeout_seconds=1):
            print("[Error] ❌ HUD crashed immediately after receiving TCP command.")
            return False
            
        print("[Ping] ✅ HUD accepted command and survived.")
        return True
    except Exception as error:
        print(f"[Error] ❌ HUD ping failed: {error}")
        return False

if __name__ == "__main__":
    print("========================================")
    print("  🚀 Starting App Boot Verification")
    print("========================================")
    
    startup_log_path = start_main_application_in_background()
    
    # Run Checks
    if not check_if_brain_program_is_running_from_file():
        print_last_few_lines_of_error_log(startup_log_path)
        stop_all_programs_and_exit_script(1)
        
    if not check_if_brain_program_is_ready_to_receive_data():
        print_last_few_lines_of_error_log(startup_log_path)
        stop_all_programs_and_exit_script(1)
        
    if not check_if_hud_display_program_is_running_from_file():
        print_last_few_lines_of_error_log("logs/hud.log")
        stop_all_programs_and_exit_script(1)
        
    if not check_if_hud_display_program_is_ready_to_receive_data():
        print_last_few_lines_of_error_log("logs/hud.log")
        stop_all_programs_and_exit_script(1)
        
    # Run Pings
    if not send_fake_audio_to_brain_and_see_if_it_survives():
        print_last_few_lines_of_error_log(startup_log_path)
        stop_all_programs_and_exit_script(1)
        
    if not send_fake_command_to_hud_and_see_if_it_survives():
        print_last_few_lines_of_error_log("logs/hud.log")
        stop_all_programs_and_exit_script(1)
        
    print("\n========================================")
    print("  ✅ SUCCESS: All Components Verified")
    print("========================================")
    stop_all_programs_and_exit_script(0)
