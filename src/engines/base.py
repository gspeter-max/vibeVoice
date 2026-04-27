from abc import ABC, abstractmethod
import numpy as np

class TranscriptionEngine(ABC):
    """
    The Rulebook for all AI transcription models.
    Any model we add to the app MUST follow these rules.
    This stops the Brain from having to guess how each model works.
    """

    @abstractmethod
    def is_stateful(self) -> bool:
        """
        Returns True if the model remembers previous audio chunks (like Nemotron).
        Returns False if the model treats every chunk as a brand new recording (like Parakeet).
        """
        pass

    @abstractmethod
    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """
        Takes raw audio numbers (floats between -1.0 and 1.0) and turns them into text.
        If the model is stateful, this returns the FULL transcript so far.
        If the model is stateless, this returns ONLY the text for this specific chunk.
        """
        pass

    def clear_internal_memory(self) -> None:
        """
        Tells a stateful model to forget everything and start fresh.
        Stateless models can just ignore this.
        """
        pass
