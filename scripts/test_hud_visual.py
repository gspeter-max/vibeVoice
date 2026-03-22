import subprocess
import sys
import time
import os
import socket

def send_cmd(cmd: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(('127.0.0.1', 57234))
        s.sendall(cmd.encode())
        s.close()
    except Exception as e:
        print(f"Failed to send {cmd}: {e}")

def verify_hud_launch():
    print("🧪 Verifying New Pure-Qt HUD Launch...")
    hud_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hud.py")
    
    # Launch HUD (it now handles its own lifecycle)
    proc = subprocess.Popen(
        [sys.executable, hud_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(1) # wait for IPC server to start
    
    if proc.poll() is not None:
        print(f"❌ HUD crashed immediately with exit code {proc.returncode}")
        sys.exit(1)
        
    print("✅ HUD process started successfully.")
    
    print("👉 Sending 'listen' command...")
    send_cmd("listen")
    time.sleep(3)
    
    print("👉 Sending 'process' command...")
    send_cmd("process")
    time.sleep(3)
    
    print("👉 Sending 'done' command...")
    send_cmd("done")
    
    # Wait for the auto-fade to finish
    time.sleep(2)
    
    print("👉 Sending 'hide' (just in case) and terminating...")
    send_cmd("hide")
    time.sleep(0.5)
    
    try:
        proc.terminate()
    except:
        pass
        
    print("✅ HUD test complete.")

if __name__ == "__main__":
    verify_hud_launch()
