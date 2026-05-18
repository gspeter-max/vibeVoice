"""Small manual helper for sending one command to the local HUD socket."""

import socket
import sys

from src.utils.settings import settings

def test_send(cmd="listen"):
    """Send one HUD command to the local socket for manual verification."""
    # NOTE: `cmd` has a default so pytest does not mistake it for a required fixture.
    # When run as a script (__main__), sys.argv[1] overrides this default.
    try:
        with socket.create_connection((settings.hud_host, settings.hud_port), timeout=1.0) as s:
            s.sendall(cmd.encode())
            print(f"✅ Sent: {cmd}")
    except OSError as e:
        print(f"Failed to send {cmd}: {e}")

if __name__ == "__main__":
    test_send(sys.argv[1] if len(sys.argv) > 1 else "listen")
