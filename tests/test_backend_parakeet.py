import pytest

import src.backend.backend_parakeet as backend_parakeet


def test_load_model_raises_clear_error_when_sherpa_onnx_is_missing(monkeypatch):
    monkeypatch.setattr(backend_parakeet, "sherpa_onnx", None)
    monkeypatch.setattr(backend_parakeet, "_SHERPA_ONNX_IMPORT_ERROR", None)

    with pytest.raises(RuntimeError, match="sherpa-onnx is unavailable in this environment"):
        backend_parakeet.load_speech_recognition_model_from_disk("nemo-parakeet-tdt-0.6b-v2")
