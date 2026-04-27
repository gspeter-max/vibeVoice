import pytest
import numpy as np
import os
from src.streaming.nemotron import NemotronStreamingEngine

def test_nemotron_engine_initialization():
    """Verify that the Nemotron engine can be initialized and finds its models."""
    engine = NemotronStreamingEngine()
    assert engine.encoder_model is not None
    assert engine.decoder_model is not None
    assert len(engine.vocabulary_tokens) > 0
    assert engine.full_text_result == ""

def test_nemotron_engine_clear_memory():
    """Verify that clearing memory works."""
    engine = NemotronStreamingEngine()
    engine.full_text_result = "some old text"
    engine.clear_internal_memory()
    assert engine.full_text_result == ""
    assert engine.last_written_token[0, 0] == 0

def test_nemotron_engine_add_audio_zeros():
    """Verify that adding zero sound data doesn't crash."""
    engine = NemotronStreamingEngine()
    # 1.12s of zeros at 16kHz
    sound_data = np.zeros(int(1.12 * 16000), dtype=np.float32)
    text_result = engine.add_audio_chunk_and_get_text(sound_data)
    assert isinstance(text_result, str)
