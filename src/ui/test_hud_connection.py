import socket
import sys

HUD_HOST, HUD_PORT = "127.0.0.1", 57234

def test_send(cmd):
    try:
        with socket.create_connection((HUD_HOST, HUD_PORT), timeout=1.0) as s:
            s.sendall(cmd.encode())
            print(f"✅ Sent: {cmd}")
    except Exception as e:
        print(f"❌ Failed to send {cmd}: {e}")

if __name__ == "__main__":
    test_send(sys.argv[1] if len(sys.argv) > 1 else "listen")
