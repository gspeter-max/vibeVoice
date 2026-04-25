import pytest
import numpy as np
import os
from src.streaming.nemotron import NemotronStreamingBackend

def test_nemotron_backend_initialization():
    """Verify that the Nemotron backend can be initialized and finds its models."""
    # This test assumes models are in models/nemotron-0.6b-onnx/
    backend = NemotronStreamingBackend()
    assert backend.encoder is not None
    assert backend.decoder is not None
    assert len(backend.tokens) > 0
    assert backend.transcript == ""

def test_nemotron_backend_reset():
    """Verify that reset clears the state."""
    backend = NemotronStreamingBackend()
    backend.transcript = "some old text"
    backend.reset()
    assert backend.transcript == ""
    assert backend.last_token[0, 0] == 0

def test_nemotron_backend_transcribe_zeros():
    """Verify that transcribing zeros doesn't crash."""
    backend = NemotronStreamingBackend()
    # 1.12s of zeros at 16kHz
    audio = np.zeros(int(1.12 * 16000), dtype=np.float32)
    transcript = backend.transcribe_chunk(audio)
    assert isinstance(transcript, str)
