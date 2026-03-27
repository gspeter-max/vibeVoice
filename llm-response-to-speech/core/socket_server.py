"""
Unix Socket Server Helper
==========================
Helper class for Unix socket operations.
"""
import socket
import struct
from typing import Optional, Tuple


class SocketClient:
    """
    Simple client for connecting to Unix socket TTS services.
    """

    def __init__(self, socket_path: str, timeout: float = 30.0):
        """
        Initialize socket client.

        Args:
            socket_path: Path to Unix socket
            timeout: Connection timeout in seconds
        """
        self.socket_path = socket_path
        self.timeout = timeout

    def send_request(self, text: str) -> bytes:
        """
        Send text to TTS service and receive audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes (WAV format)

        Raises:
            ConnectionError: If connection fails
            RuntimeError: If request fails
        """
        sock = None
        try:
            # Connect to socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)

            # Send text length + data
            text_bytes = text.encode('utf-8')
            length_prefix = struct.pack('>I', len(text_bytes))

            sock.sendall(length_prefix)
            sock.sendall(text_bytes)

            # Receive audio length
            length_data = self._recv_exact(sock, 4)
            audio_length = struct.unpack('>I', length_data)[0]

            # Receive audio data
            audio_data = self._recv_exact(sock, audio_length)

            return audio_data

        except socket.error as e:
            raise ConnectionError(f"Socket error: {e}")
        finally:
            if sock:
                sock.close()

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise socket.error("Connection closed")
            data += chunk
        return data
