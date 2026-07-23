import numpy as np
from unittest.mock import MagicMock
from src.models.nemotron_engine import NemotronEngine

def test_nemotron_engine_is_stateful():
    engine = NemotronEngine()
    assert engine.is_stateful() is True

def test_nemotron_engine_transcribes_and_clears_memory():
    engine = NemotronEngine()
    engine._engine = MagicMock()
    engine._engine.add_audio_chunk_and_get_text.return_value = "hello cumulative world"
    
    fake_audio = np.zeros(1024, dtype=np.float32)
    result = engine.transcribe_chunk(fake_audio)
    
    assert result == "hello cumulative world"
    engine._engine.add_audio_chunk_and_get_text.assert_called_once_with(fake_audio)
    
    engine.clear_internal_memory()
    engine._engine.clear_internal_memory.assert_called_once()

def test_nemotron_engine_transcribes_bytes():
    engine = NemotronEngine()
    engine._engine = MagicMock()
    engine._engine.add_audio_chunk_and_get_text.return_value = "hello cumulative world"
    
    fake_audio = np.zeros(1024, dtype=np.float32)
    fake_audio_bytes = fake_audio.tobytes()
    result = engine.transcribe_chunk(fake_audio_bytes)
    
    assert result == "hello cumulative world"
    args, kwargs = engine._engine.add_audio_chunk_and_get_text.call_args
    passed_array = args[0]
    assert isinstance(passed_array, np.ndarray)
    assert passed_array.dtype == np.float32
    assert np.allclose(passed_array, fake_audio)
