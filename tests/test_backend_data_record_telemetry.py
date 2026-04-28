import pytest
from unittest.mock import MagicMock

def test_telemetry_independent_of_brain():
    """
    Verify that telemetry logic can be imported completely independently of brain.py.
    This guarantees that the data recording system does not rely on execution logic.
    """
    import sys
    if "src.backend.brain" in sys.modules:
        del sys.modules["src.backend.brain"]
    import src.backend.data_record.telemetry as telemetry
    assert "src.backend.brain" not in sys.modules

def test_telemetry_disabled_fast_return(monkeypatch):
    """
    Verify that if telemetry is disabled, getting a recorder returns None instantly.
    """
    import src.backend.data_record.telemetry as telemetry
    monkeypatch.setattr(telemetry, "STREAMING_TELEMETRY_ENABLED", False)
    # Should return None immediately without errors
    assert telemetry._telemetry_recorder_for_session("test_sess") is None

def test_model_name_for_telemetry_missing_engine():
    """
    Verify that when the engine is not loaded (None), we get None back.
    """
    import src.backend.data_record.telemetry as telemetry
    import src.backend.state as state
    state.backend_info["engine"] = None
    assert telemetry._model_name_for_telemetry() is None