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
