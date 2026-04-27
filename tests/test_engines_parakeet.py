import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.engines.parakeet import ParakeetEngine

def test_parakeet_engine_is_stateless():
    with patch("src.backend.backend_parakeet.load_speech_recognition_model_from_disk"):
        engine = ParakeetEngine(model_name="dummy_model")
        assert engine.is_stateful() is False

def test_parakeet_engine_transcribes_audio():
    with patch("src.backend.backend_parakeet.load_speech_recognition_model_from_disk") as mock_load, \
         patch("src.backend.backend_parakeet.convert_audio_to_text") as mock_convert:
        
        mock_convert.return_value = "hello world"
        engine = ParakeetEngine(model_name="parakeet-tdt-0.6b-v3")
        
        # Give it some fake audio
        fake_audio = np.zeros(1024, dtype=np.float32)
        result = engine.transcribe_chunk(fake_audio)
        
        assert result == "hello world"
        mock_convert.assert_called_once()
