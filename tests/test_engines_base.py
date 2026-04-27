import pytest
import numpy as np
from src.engines.base import TranscriptionEngine

def test_engine_interface_requires_is_stateful_method():
    class BadEngine(TranscriptionEngine):
        pass
    
    with pytest.raises(TypeError) as exc_info:
        # Should fail because it doesn't implement abstract methods
        engine = BadEngine()
    
    error_message = str(exc_info.value)
    assert "Can't instantiate abstract class" in error_message
    assert "is_stateful" in error_message
    assert "transcribe_chunk" in error_message

def test_engine_interface_default_methods():
    class GoodEngine(TranscriptionEngine):
        def is_stateful(self) -> bool:
            return False
            
        def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
            return "test"
            
    engine = GoodEngine()
    
    # clear_internal_memory should be safe to call even if not overridden
    engine.clear_internal_memory()
