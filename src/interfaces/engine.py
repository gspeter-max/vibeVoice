from typing import Protocol

import numpy as np


class TranscriptionEngine(Protocol):
    """The Rulebook for all AI transcription models."""

    def is_stateful(self) -> bool:
        """Returns True if the model remembers previous audio chunks (like Nemotron)."""
        ...

    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """Takes raw audio and turns them into text."""
        ...

    def clear_internal_memory(self) -> None:
        """Tells a stateful model to forget everything and start fresh."""
        ...
