import sys
import time
import subprocess
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

print("Starting HUD...")
hud = subprocess.Popen([sys.executable, "hud.py"])
time.sleep(1)

print("Sending listen...")
send_cmd("listen")
time.sleep(3)

print("Sending process...")
send_cmd("process")
time.sleep(3)

print("Sending done...")
send_cmd("done")
time.sleep(2)

print("Closing...")
hud.terminate()
