import numpy as np
from unittest.mock import patch
from src.engines.parakeet import ParakeetEngine

def test_parakeet_engine_is_stateless():
    with patch("src.engines.parakeet.load_model"):
        engine = ParakeetEngine(model_name="dummy_model")
        assert engine.is_stateful() is False

def test_parakeet_engine_transcribes_audio():
    with patch("src.engines.parakeet.load_model"), \
         patch("src.engines.parakeet.transcribe") as mock_transcribe:
        
        mock_transcribe.return_value = "hello world"
        engine = ParakeetEngine(model_name="parakeet-tdt-0.6b-v3")
        
        # Give it some fake audio
        fake_audio = np.zeros(1024, dtype=np.float32)
        result = engine.transcribe_chunk(fake_audio)
        
        assert result == "hello world"
        mock_transcribe.assert_called_once_with(engine.tts_model, fake_audio)

def test_parakeet_engine_transcribes_bytes():
    with patch("src.engines.parakeet.load_model"), \
         patch("src.engines.parakeet.transcribe") as mock_transcribe:
        
        mock_transcribe.return_value = "hello world"
        engine = ParakeetEngine(model_name="parakeet-tdt-0.6b-v3")
        
        # Give it some fake audio as bytes
        fake_audio = np.zeros(1024, dtype=np.float32)
        fake_audio_bytes = fake_audio.tobytes()
        result = engine.transcribe_chunk(fake_audio_bytes)
        
        assert result == "hello world"
        args, kwargs = mock_transcribe.call_args
        passed_model, passed_array = args
        assert isinstance(passed_array, np.ndarray)
        assert passed_array.dtype == np.float32
        assert np.allclose(passed_array, fake_audio)
