import numpy as np
from src.engines.base import TranscriptionEngine

# We import the old backend logic to do the heavy lifting,
# but we wrap it in our clean new class.
try:
    import src.backend.backend_parakeet as legacy_backend
except ImportError:
    legacy_backend = None

class ParakeetEngine(TranscriptionEngine):
    """
    The implementation for Parakeet, Conformer, and Moonshine models.
    These models are "stateless", meaning they don't remember the past.
    They transcribe whatever chunk of audio you give them, right now.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._loaded_model = None
        
        # Load the model from disk into memory when the class is created
        if legacy_backend:
            self._loaded_model = legacy_backend.load_speech_recognition_model_from_disk(self.model_name)

    def is_stateful(self) -> bool:
        """Parakeet models do not remember past audio chunks."""
        return False

    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """
        Passes the audio to the Sherpa-ONNX backend and returns the text string.
        """
        if not legacy_backend or not self._loaded_model:
            return ""
        
        text = legacy_backend.convert_audio_to_text(self._loaded_model, audio_samples)
        return text.strip()

    def clear_internal_memory(self) -> None:
        """
        Stateless models have no memory to clear.
        We do nothing, and that is perfectly fine.
        """
        pass
