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

# Tell the system to call our safe stop function when it receives an interrupt or termination signal
signal.signal(signal.SIGINT, stop_safely_when_user_presses_control_c)
signal.signal(signal.SIGTERM, stop_safely_when_user_presses_control_c)

if __name__ == "__main__":
    print("Startup Verifier initialized.")
