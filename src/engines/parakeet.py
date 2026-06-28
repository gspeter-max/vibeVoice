from dataclasses import dataclass

import numpy as np

from src.backend.backend_parakeet import load_model, transcribe
from src.interfaces import TranscriptionEngine


@dataclass
class ParakeetEngine:
    """
    The implementation for Parakeet, Conformer, and Moonshine models.
    These models are "stateless", meaning they don't remember the insert_transcript.
    They transcribe whatever chunk of audio you give them, right now.
    """

    model_name: str

    def __post_init__(self):
        self.tts_model = load_model(self.model_name)

    def is_stateful(self) -> bool:
        """Parakeet models do not remember insert_transcript audio chunks."""
        return False

    def transcribe_chunk(self, audio_samples: np.ndarray | bytes) -> str:
        """
        Passes the audio to the Sherpa-ONNX backend and returns the text string.
        """
        if isinstance(audio_samples, bytes):
            audio_samples = np.frombuffer(audio_samples, dtype=np.float32)

        text = transcribe(self.tts_model, audio_samples)
        return text.strip()

    def clear_internal_memory(self) -> None:
        """
        Stateless models have no memory to clear.
        We do nothing, and that is perfectly fine.
        """
