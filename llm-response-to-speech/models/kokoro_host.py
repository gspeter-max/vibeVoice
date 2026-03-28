"""
Kokoro TTS Host
===============
Lightweight, CPU-based TTS model using Kokoro CLI tool, HTTP API, or mock mode.

Kokoro 82M is perfect for systems with limited resources:
- Model size: 350 MB
- RAM requirement: 4 GB
- CPU-based (no GPU needed)
- Open source (MIT license)

Backends (priority order):
1. HTTP API (Docker) - Model stays loaded, fastest
2. CLI tool - Model loads per request
3. Mock mode - Testing only
"""
import subprocess
import tempfile
import os
import numpy as np
import requests
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, '.')


from core.base_tts import BaseTTSHost
from core.audio_utils import numpy_to_wav
from config.settings import get_model_config


class KokoroTTSHost(BaseTTSHost):
    """
    Kokoro TTS host.

    Model: Kokoro TTS 82M
    Type: CLI wrapper or Mock
    Features: Fast inference, lightweight, no API costs
    """

    def __init__(self):
        config = get_model_config('kokoro')
        super().__init__(config)

        # Check for Kokoro HTTP API (Docker container) - FASTEST
        self.kokoro_api_url = self._check_kokoro_api()

        # Try to find kokoro-tts binary
        self.kokoro_binary = self._find_kokoro_binary()

        # Check for edge-tts availability
        self.has_edge_tts = self._check_edge_tts()

        # Determine mode (priority: HTTP API > CLI > edge-tts > mock)
        if self.kokoro_api_url:
            self.mode = "http-api"
            self.logger.info(f"Using Kokoro HTTP API at {self.kokoro_api_url} (model pre-loaded)")
        elif self.kokoro_binary:
            self.mode = "cli"
            self.logger.warning(f"Using Kokoro CLI (slower - model loads per request)")
        elif self.has_edge_tts:
            self.mode = "edge-tts"
            self.logger.info("Using Microsoft Edge TTS (requires internet)")
        else:
            self.mode = "mock"
            self.logger.warning("No TTS backend available, using MOCK mode")

    def _find_kokoro_binary(self) -> str:
        """Find kokoro-tts executable in PATH."""
        possible_paths = [
            "kokoro-tts",  # In PATH
            os.path.expanduser("~/kokoro-tts/kokoro-tts"),
            "/usr/local/bin/kokoro-tts",
        ]

        for path in possible_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded

        # Try which
        try:
            result = subprocess.run(
                ["which", "kokoro-tts"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        return None

    def _check_kokoro_api(self) -> str:
        """Check if Kokoro HTTP API is available (Docker container)."""
        api_url = "http://localhost:8880/v1/audio/speech"

        try:
            # Quick health check
            response = requests.get(
                "http://localhost:8880/v1/models",
                timeout=2
            )
            if response.status_code == 200:
                return api_url
        except:
            pass

        return None

    def _check_edge_tts(self) -> bool:
        """Check if edge-tts library is available."""
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def load_model(self):
        """
        Verify TTS backend is available.

        Checks in order:
        1. HTTP API (Docker) - Model pre-loaded, fastest
        2. kokoro-tts CLI - Model loads per request
        3. edge-tts - Microsoft online service
        4. Mock mode - Testing only
        """
        if self.mode == "http-api":
            self.model = "kokoro-http-api"
            self.logger.info("Kokoro HTTP API available (model pre-loaded in Docker)")

        elif self.mode == "cli":
            self.logger.info(f"Kokoro binary: {self.kokoro_binary}")

            # Test binary
            try:
                result = subprocess.run(
                    [self.kokoro_binary, "--help"],
                    capture_output=True,
                    timeout=5
                )

                if result.returncode == 0:
                    self.model = self.kokoro_binary
                    self.logger.info("Kokoro TTS CLI verified")
                else:
                    raise RuntimeError("kokoro-tts binary not working")

            except subprocess.TimeoutExpired:
                raise RuntimeError("kokoro-tts timeout")
            except Exception as e:
                raise RuntimeError(f"kokoro-tts verification failed: {e}")

        elif self.mode == "edge-tts":
            self.model = "edge-tts"
            self.logger.info("Microsoft Edge TTS available")

        elif self.mode == "mock":
            self.model = "mock"
            self.logger.warning("Kokoro TTS: Using MOCK mode (beep tones only)")

    def synthesize(
        self,
        text: str,
        voice: str = "am_michael",
        lang: str = "en-us",
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech using available backend.

        Args:
            text: Input text to synthesize
            voice: Voice name (am_michael, af_sarah, etc.)
            lang: Language code
            speed: Speech speed multiplier
            **kwargs: Additional parameters

        Returns:
            Audio data as bytes (WAV format)
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")

        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")

        # Dispatch to appropriate backend
        if self.mode == "http-api":
            return self._synthesize_with_http_api(text, voice)
        elif self.mode == "cli":
            return self._synthesize_real(text, voice, lang, speed)
        elif self.mode == "edge-tts":
            return self._synthesize_with_edge_tts(text, voice)
        else:  # mock mode
            return self._synthesize_mock(text)

    def _synthesize_with_http_api(self, text: str, voice: str = "am_michael") -> bytes:
        """
        Synthesize using Kokoro HTTP API (Docker container).

        This is the FASTEST method because:
        - Model is pre-loaded in memory (Docker container)
        - No subprocess spawning overhead
        - No model loading time
        - Just HTTP request/response

        Args:
            text: Input text to synthesize
            voice: Voice name (am_michael, af_sarah, etc.)

        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            # Map voice names to API format if needed
            # Most Kokoro voices work as-is
            voice_param = voice

            # Make request to Kokoro HTTP API
            response = requests.post(
                self.kokoro_api_url,
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": voice_param,
                    "response_format": "wav"
                },
                timeout=30
            )

            if response.status_code == 200:
                audio_bytes = response.content
                self.logger.info(f"Generated {len(audio_bytes)} bytes via HTTP API (model pre-loaded)")
                return audio_bytes
            else:
                raise RuntimeError(f"HTTP API failed: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            raise RuntimeError("HTTP API request timeout")
        except Exception as e:
            raise RuntimeError(f"HTTP API request failed: {e}")

    def _synthesize_real(
        self,
        text: str,
        voice: str,
        lang: str,
        speed: float
    ) -> bytes:
        """Synthesize using real kokoro-tts binary."""
        try:
            # Create temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
                input_file.write(text)
                input_path = input_file.name

            output_path = input_path.replace('.txt', '.wav')

            # Build command
            cmd = [
                self.kokoro_binary,
                input_path,
                output_path,
                "--voice", voice,
                "--lang", lang,
                "--speed", str(speed)
            ]

            # Run kokoro-tts
            self.logger.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"kokoro-tts failed: {error_msg}")

            # Read output WAV
            with open(output_path, 'rb') as f:
                audio_bytes = f.read()

            # Cleanup temp files
            os.remove(input_path)
            os.remove(output_path)

            return audio_bytes

        except subprocess.TimeoutExpired:
            raise RuntimeError("kokoro-tts synthesis timeout")

    def _synthesize_mock(self, text: str) -> bytes:
        """
        Synthesize mock audio (beep tone) for testing without kokoro-tts.

        Generates a simple sine wave tone whose duration scales with text length.
        This allows testing the Unix socket protocol without needing kokoro-tts.

        Args:
            text: Input text (used to calculate duration)

        Returns:
            Mock audio as WAV bytes
        """
        # Duration scales with text length (0.05 seconds per character)
        duration = len(text) * 0.05
        duration = max(duration, 0.5)  # Minimum 0.5 seconds
        duration = min(duration, 10.0)  # Maximum 10 seconds

        # Generate sine wave tone (440 Hz = A4 note)
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # Hz
        audio = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume

        # Convert to WAV
        return numpy_to_wav(audio.astype(np.float32), sample_rate)

    def get_model_info(self) -> Dict[str, Any]:
        """Return Kokoro model information."""
        mode_names = {
            "http-api": "HTTP API (Docker, model pre-loaded)",
            "cli": "CLI wrapper (model loads per request)",
            "edge-tts": "Microsoft Edge TTS (online)",
            "mock": "Mock mode (beep tones)"
        }

        return {
            'name': 'Kokoro 82M',
            'type': mode_names.get(self.mode, "Unknown"),
            'mode': self.mode,
            'model_size_mb': 350 if self.mode in ["http-api", "cli"] else 0,
            'sample_rate': 22050 if self.mode in ["http-api", "cli", "mock"] else 24000,
            'voices': ['am_michael', 'am_adam', 'af_sky', 'af_sarah'],
            'languages': ['en-us', 'en-gb'],
            'license': 'MIT (Open Source)',
            'api_url': self.kokoro_api_url if self.mode == "http-api" else None,
            'executable': self.kokoro_binary if self.mode == "cli" else None,
            'requires_internet': self.mode == "edge-tts",
        }


if __name__ == "__main__":
    # Test Kokoro host directly
    import argparse

    parser = argparse.ArgumentParser(description="Test Kokoro TTS host")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="test_output.wav", help="Output WAV file")
    parser.add_argument("--voice", default="am_michael", help="Voice name")
    args = parser.parse_args()

    host = KokoroTTSHost()
    host.load_model()

    print(f"Mode: {host.mode}")
    print(f"Model: {host.model}")
    print(f"Synthesizing: {args.text}")

    audio = host.synthesize(args.text, voice=args.voice)

    with open(args.output, 'wb') as f:
        f.write(audio)

    print(f"Saved to {args.output}")
