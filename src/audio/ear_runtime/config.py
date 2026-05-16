"""
config.py — Ear Configuration
==============================
Contains all constants and configuration parameters for the Ear module.
This file should have no dependencies on other ear_runtime modules.
"""

import os
import pyaudio
from src.utils.settings import settings

# ── Socket & Network ─────────────────────────────────────────────────────────
SOCKET_PATH = "/tmp/parakeet.sock"
"""Path to the Unix domain socket for communicating with the Brain."""

VOL_PORT = 57235
"""UDP port used for sending volume levels to the HUD."""

# ── Audio Capture ────────────────────────────────────────────────────────────
FORMAT = pyaudio.paInt16
"""Audio format used for PyAudio capture (16-bit PCM)."""

CHANNELS = 1
"""Number of audio channels (Mono)."""

RATE = 16000
"""Sample rate in Hz (16kHz is standard for speech models)."""

CHUNK = 1024
"""Number of audio frames per buffer chunk."""

# ── Recording Modes ──────────────────────────────────────────────────────────
BACKEND = settings.backend
"""The transcription backend to use (e.g., 'parakeet', 'nemotron')."""

RECORDING_MODE = settings.recording_mode
"""
The send style for audio data:
- 'no_streaming': send raw audio and let Brain wait for socket close.
- 'silence_streaming': split speech on silence and send chunks.
"""

NO_STREAMING_MODE = "no_streaming"
SILENCE_STREAMING_MODE = "silence_streaming"

RECORDING_BUTTON_HOLD_THRESHOLD = 0.4
"""Time in seconds to hold the record button before it triggers a continuous recording."""

# ── VAD (Voice Activity Detection) ───────────────────────────────────────────
VAD_MODEL_PATH = os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx")
"""File path to the Silero VAD ONNX model."""

VAD_THRESHOLD = settings.vad_score_threshold
"""Confidence threshold for VAD (0.0 to 1.0)."""

VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT = settings.silence_timeout_seconds
"""Seconds of silence before splitting the audio chunk."""

VAD_ENERGY_THRESHOLD = settings.vad_energy_threshold
"""Energy threshold for VAD to filter out low-volume noise."""

VAD_ENERGY_RATIO = settings.vad_energy_ratio
"""Ratio of energy to help distinguish speech from background hum."""

VAD_STATUS_LOG_INTERVAL = 5.0
"""Interval in seconds for logging VAD engine status."""

RECORDING_LEVEL_LOG_INTERVAL = 0.4
"""Interval in seconds for logging the visual volume meter in the terminal."""

# ── Streaming Parameters ─────────────────────────────────────────────────────
OVERLAP_SECONDS = settings.overlap_seconds
"""Seconds of audio to overlap between consecutive chunks to maintain speech context."""

MIN_CHUNK_SECONDS_REQ_FOR_SPLITING_DUE_TO_SILENCE_STREAMING = settings.minimum_chunk_age_before_silence_split_seconds
"""Minimum duration a chunk must have before it can be split by silence."""

# ── Constants for UI/Terminal ────────────────────────────────────────────────
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"

# ── Internal Constants ───────────────────────────────────────────────────────
_RCMD_VK = 54
"""Virtual key code for Right Command key on macOS."""
