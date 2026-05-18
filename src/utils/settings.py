# src/utils/settings.py
"""
Central configuration for VibeVoice.

All environment variables are declared here as typed fields.
Import `settings` from this module instead of calling os.environ.get() directly.

Priority order (highest wins):
  1. Real os.environ (set by shell / start.sh)
  2. .env file
  3. Default values defined below
"""

import os
from pathlib import Path

import pyaudio
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VibeVoiceSettings(BaseSettings):
    """
    Typed configuration loaded from environment variables and the .env file.
    Field names map to env vars of the same name in UPPERCASE.
    """

    # ── API Keys ───────────────────────────────────────────────────────────
    groq_api_key: str = ""
    cerebras_api_key: str = ""

    # ── Provider & Model ───────────────────────────────────────────────────
    vibevoice_provider_index: int = 0
    """Index into the PROVIDERS list in llm_router.py (0=Groq, 1=Cerebras)."""


    # ── Speech Models ─────────────────────────────────────────────────────
    available_stt_models: list[str] = Field(
        default_factory=lambda: [
            "fast-conformer-ctc-en-24500",
            "moonshine-base",
            "parakeet-tdt-0.6b-v2",
            "parakeet-tdt-0.6b-v3",
        ]
    )
    stt_model: str = "parakeet-tdt-0.6b-v3"
    """Speech-to-text model name loaded on startup."""

    # ── Hardware ───────────────────────────────────────────────────────────
    vibevoice_mic_index: str = ""
    """PyAudio device index for the selected microphone. Empty = use system default."""

    audio_format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024

    # ── Recording ─────────────────────────────────────────────────────────
    recording_mode: str = "silence_streaming"
    """'silence_streaming' (split on silence) or 'no_streaming' (send full recording at once)."""

    no_streaming_mode: str = "no_streaming"
    silence_streaming_mode: str = "silence_streaming"
    recording_button_hold_threshold: float = 0.4

    backend: str = "parakeet"
    """Transcription backend: 'parakeet' or 'nemotron'."""

    # ── IPC / Network ─────────────────────────────────────────────────────
    socket_path: str = "/tmp/parakeet.sock"
    """Unix socket path for Ear → Brain audio streaming."""

    hud_host: str = "127.0.0.1"
    """Hostname the Brain uses to send state updates to the HUD."""

    hud_port: int = 57234
    """TCP port the HUD listens on for state commands from the Brain."""

    vol_port: int = 57235
    """UDP port used by the Ear to send live volume levels to the HUD."""

    # ── Telemetry ─────────────────────────────────────────────────────────
    streaming_telemetry_enabled: bool = False
    """When True, each session writes a detailed JSON log to streaming_telemetry_dir."""

    streaming_telemetry_dir: Path = Path("logs/streaming_sessions")
    """Directory where telemetry JSON files are saved."""

    # ── Streaming / VAD Constants ──────────────────────────────────────────
    vad_model_path: str = Field(
        default_factory=lambda: os.path.expanduser(
            "~/.cache/parakeet-flow/vad/silero_vad.onnx"
        ),
    )
    """Filesystem path to the Silero VAD ONNX model (override via VAD_MODEL_PATH)."""

    vad_score_threshold: float = 0.50
    vad_energy_threshold: float = 0.05
    vad_energy_ratio: float = 2.5
    silence_timeout_seconds: float = 0.8
    overlap_seconds: float = 3.5
    minimum_chunk_age_before_silence_split_seconds: float = 8.0
    semantic_overlapping_threshold: float = 0.70
    vad_status_log_interval: float = 5.0
    recording_level_log_interval: float = 0.4

    # ── UI & Input ─────────────────────────────────────────────────────────
    ansi_blue: str = "\033[94m"
    ansi_reset: str = "\033[0m"
    right_cmd_vk: int = 54
    nemotron_model: str = "nemotron-streaming-0.6b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown keys in .env (e.g. VAD thresholds)
    )

    @property
    def normalized_recording_mode(self) -> str:
        """Return the recording mode in normalized lowercase form."""
        return self.recording_mode.strip().lower()

    @property
    def is_no_streaming_mode(self) -> bool:
        """Return `True` when audio should be sent as one full recording."""
        return self.normalized_recording_mode == self.no_streaming_mode

    @property
    def is_silence_streaming_mode(self) -> bool:
        """Return `True` when audio should be split on silence."""
        return self.normalized_recording_mode == self.silence_streaming_mode

    @property
    def active_stt_models(self) -> list[str]:
        """Return the STT models valid for the current recording mode."""
        models = list(self.available_stt_models)
        if self.is_silence_streaming_mode:
            models.append(self.nemotron_model)
        return models


# Single shared instance — imported by all modules that need config.
settings = VibeVoiceSettings()
