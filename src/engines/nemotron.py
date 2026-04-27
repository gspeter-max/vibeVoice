import numpy as np
from src.engines.base import TranscriptionEngine

try:
    from src.streaming.nemotron import NemotronStreamingEngine as LegacyNemotron
except ImportError:
    LegacyNemotron = None

class NemotronEngine(TranscriptionEngine):
    """
    The implementation for the Nemotron streaming model.
    This model is "stateful", meaning it keeps an internal buffer of the audio.
    When you give it a new chunk, it returns the transcript for the ENTIRE recording so far.
    """
    def __init__(self):
        self._engine = LegacyNemotron() if LegacyNemotron else None

    def is_stateful(self) -> bool:
        """Nemotron absolutely remembers past chunks."""
        return True

    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """
        Adds the audio to Nemotron's internal buffer and gets the full, cumulative text back.
        """
        if not self._engine:
            return ""
        
        text = self._engine.add_audio_chunk_and_get_text(audio_samples)
        return text.strip()

    def clear_internal_memory(self) -> None:
        """
        Forces Nemotron to wipe its internal buffer.
        This must be called when the user finishes a recording session,
        so the next recording doesn't start with the old words.
        """
        if self._engine:
            self._engine.clear_internal_memory()
