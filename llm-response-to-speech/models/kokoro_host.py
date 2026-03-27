"""
Kokoro TTS Host
===============
Lightweight, CPU-based TTS model using Kokoro CLI tool or mock mode.

Kokoro 82M is perfect for systems with limited resources:
- Model size: 350 MB
- RAM requirement: 4 GB
- CPU-based (no GPU needed)
- Open source (MIT license)
"""
import subprocess
import tempfile
import os
import numpy as np
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

        # Try to find kokoro-tts binary
        self.kokoro_binary = self._find_kokoro_binary()
        self.use_mock = (self.kokoro_binary is None)

        if self.use_mock:
            self.logger.warning("kokoro-tts not found, using MOCK mode for testing")

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

    def load_model(self):
        """
        Verify Kokoro TTS is installed and models are available.

        For mock mode, just mark as loaded.
        """
        if self.use_mock:
            self.model = "mock"
            self.logger.info("Kokoro TTS: Using MOCK mode for testing")
        else:
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
                    self.logger.info("Kokoro TTS verified")
                else:
                    raise RuntimeError("kokoro-tts binary not working")

            except subprocess.TimeoutExpired:
                raise RuntimeError("kokoro-tts timeout")
            except Exception as e:
                raise RuntimeError(f"kokoro-tts verification failed: {e}")

    def synthesize(
        self,
        text: str,
        voice: str = "af_sarah",
        lang: str = "en-us",
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech using Kokoro CLI tool or MOCK mode.

        Args:
            text: Input text to synthesize
            voice: Voice name (af_sarah, am_adam, etc.)
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

        if self.use_mock:
            return self._synthesize_mock(text)
        else:
            return self._synthesize_real(text, voice, lang, speed)

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
        return {
            'name': 'Kokoro 82M',
            'type': 'CLI wrapper' if not self.use_mock else 'Mock mode',
            'model_size_mb': 350,
            'sample_rate': 22050,
            'voices': ['af_sarah', 'am_adam', 'af_sky', 'am_michael'],
            'languages': ['en-us', 'en-gb'],
            'license': 'MIT (Open Source)',
            'executable': self.kokoro_binary if not self.use_mock else 'None (mock mode)',
        }


if __name__ == "__main__":
    # Test Kokoro host directly
    import argparse

    parser = argparse.ArgumentParser(description="Test Kokoro TTS host")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="test_output.wav", help="Output WAV file")
    parser.add_argument("--voice", default="af_sarah", help="Voice name")
    args = parser.parse_args()

    host = KokoroTTSHost()
    host.load_model()

    print(f"Mode: {'MOCK' if host.use_mock else 'REAL'}")
    print(f"Synthesizing: {args.text}")

    audio = host.synthesize(args.text, voice=args.voice)

    with open(args.output, 'wb') as f:
        f.write(audio)

    print(f"Saved to {args.output}")
