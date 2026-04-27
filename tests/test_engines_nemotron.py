import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.engines.nemotron import NemotronEngine

def test_nemotron_engine_is_stateful():
    with patch("src.engines.nemotron.LegacyNemotron") as mock_legacy:
        engine = NemotronEngine()
        assert engine.is_stateful() is True

def test_nemotron_engine_transcribes_and_clears_memory():
    with patch("src.engines.nemotron.LegacyNemotron") as mock_legacy_class:
        mock_instance = MagicMock()
        mock_instance.add_audio_chunk_and_get_text.return_value = "hello cumulative world"
        mock_legacy_class.return_value = mock_instance
        
        engine = NemotronEngine()
        
        fake_audio = np.zeros(1024, dtype=np.float32)
        result = engine.transcribe_chunk(fake_audio)
        
        assert result == "hello cumulative world"
        mock_instance.add_audio_chunk_and_get_text.assert_called_once_with(fake_audio)
        
        engine.clear_internal_memory()
        mock_instance.clear_internal_memory.assert_called_once()
