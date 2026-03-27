"""
Base TTS Hosting Class
======================
Abstract base class for TTS model hosting using Unix socket IPC.
"""
from abc import ABC, abstractmethod
import os
import signal
import logging
import socket as socket_module
import struct
from pathlib import Path
from typing import Dict, Any, Optional
import psutil


class BaseTTSHost(ABC):
    """
    Abstract base class for TTS model hosting via Unix sockets.

    Provides common functionality:
    - Unix socket server lifecycle
    - Model loading and management
    - Statistics tracking
    - Signal handling for graceful shutdown
    - Logging configuration

    Subclasses must implement:
    - load_model(): Load the TTS model
    - synthesize(): Convert text to audio bytes
    - get_model_info(): Return model metadata
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TTS host.

        Args:
            config: Dictionary with keys:
                - name: Model display name
                - socket: Unix socket path
                - pid_file: PID file path
                - log_file: Log file name
                - type: "api", "local", "cli", or "cloud_api"
                - requires_api_key: bool
                - env_var: API key environment variable (if needed)
        """
        self.name = config["name"]
        self.socket_path = config["socket"]
        self.pid_file = config["pid_file"]
        self.log_file = config["log_file"]
        self.service_type = config["type"]
        self.requires_api_key = config.get("requires_api_key", False)
        self.api_key_env_var = config.get("env_var")

        # State
        self.model = None
        self.server_socket: Optional[socket_module.socket] = None
        self.running = False

        # Statistics
        self.stats = {
            'requests_processed': 0,
            'total_characters': 0,
            'total_audio_seconds': 0.0,
            'errors': 0,
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_path = Path(__file__).parent.parent / self.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @abstractmethod
    def load_model(self):
        """
        Load the TTS model.

        Should set self.model and raise exception if loading fails.
        """
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Input text to synthesize
            voice: Voice name or preset
            **kwargs: Additional model-specific parameters

        Returns:
            Audio data as bytes (WAV format)

        Raises:
            ValueError: If text is invalid
            RuntimeError: If synthesis fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the model.

        Returns:
            Dictionary with model metadata
        """
        pass

    def start(self):
        """
        Start the TTS service.

        Creates Unix socket, writes PID file, starts accept loop.
        Handles signals for graceful shutdown.
        """
        # Check if already running
        if self._is_already_running():
            raise RuntimeError(f"{self.name} is already running (PID file exists)")

        # Write PID file
        self._write_pid_file()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Create Unix socket
        self._create_socket()

        # Load model
        self.logger.info(f"Loading {self.name}...")
        self.load_model()
        self.logger.info(f"{self.name} loaded successfully")

        # Start accept loop
        self.running = True
        self.logger.info(f"{self.name} service started on {self.socket_path}")
        self.logger.info(f"Model info: {self.get_model_info()}")

        try:
            self._accept_loop()
        finally:
            self._cleanup()

    def _is_already_running(self) -> bool:
        """Check if service is already running."""
        if not os.path.exists(self.pid_file):
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process is running
            return psutil.pid_exists(pid)
        except:
            return False

    def _write_pid_file(self):
        """Write current PID to file."""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))

    def _create_socket(self):
        """Create Unix domain socket."""
        # Remove existing socket if present
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        # Create socket
        self.server_socket = socket_module.socket(
            socket_module.AF_UNIX,
            socket_module.SOCK_STREAM
        )
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        os.chmod(self.socket_path, 0o777)

    def _accept_loop(self):
        """Main accept loop for client connections."""
        while self.running:
            try:
                # Set timeout to allow checking running flag
                self.server_socket.settimeout(1.0)

                try:
                    client_socket, _ = self.server_socket.accept()
                except socket_module.timeout:
                    continue

                # Handle client (non-blocking)
                self._handle_client(client_socket)

            except Exception as e:
                if self.running:
                    self.logger.error(f"Error in accept loop: {e}")
                    self.stats['errors'] += 1

    def _handle_client(self, client_socket: socket_module.socket):
        """
        Handle single client request.

        Protocol:
        1. Client sends: 4-byte big-endian length + text data
        2. Server responds: 4-byte big-endian length + audio data (WAV)
        """
        try:
            # Receive text length
            length_data = self._recv_exact(client_socket, 4)
            text_length = struct.unpack('>I', length_data)[0]

            # Receive text data
            text_data = self._recv_exact(client_socket, text_length)
            text = text_data.decode('utf-8')

            self.logger.info(f"Synthesizing: {text[:50]}...")

            # Synthesize
            audio_bytes = self.synthesize(text)

            # Send audio length
            audio_length = len(audio_bytes)
            client_socket.sendall(struct.pack('>I', audio_length))

            # Send audio data
            client_socket.sendall(audio_bytes)

            # Update stats
            self.stats['requests_processed'] += 1
            self.stats['total_characters'] += len(text)

            audio_info = self._get_audio_duration(audio_bytes)
            if audio_info:
                self.stats['total_audio_seconds'] += audio_info['duration']

            self.logger.info(
                f"Sent {audio_length} bytes "
                f"({audio_info['duration']:.2f}s if audio)"
            )

        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
            self.stats['errors'] += 1
        finally:
            client_socket.close()

    def _recv_exact(self, sock: socket_module.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise socket_module.error("Connection closed")
            data += chunk
        return data

    def _get_audio_duration(self, wav_bytes: bytes) -> Optional[dict]:
        """Get audio duration from WAV bytes."""
        try:
            from core.audio_utils import get_wav_info
            return get_wav_info(wav_bytes)
        except:
            return None

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self):
        """Stop the service gracefully."""
        self.running = False

    def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up...")

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)

        # Print stats
        self.logger.info(f"Statistics: {self.stats}")
        self.logger.info(f"{self.name} service stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.stats.copy()
