from src.backend.state import BackendState, SessionStates


def test_telemetry_independent_of_brain():
    """
    Verify that telemetry logic can be imported completely independently of brain.py.
    This guarantees that the data recording system does not rely on execution logic.
    """
    import sys

    if "src.backend.brain" in sys.modules:
        del sys.modules["src.backend.brain"]
    assert "src.backend.brain" not in sys.modules


def test_telemetry_disabled_fast_return(monkeypatch):
    """
    Verify that if telemetry is disabled, getting a recorder returns None instantly.
    """
    import src.backend.data_record.telemetry as telemetry

    monkeypatch.setattr(telemetry.settings, "streaming_telemetry_enabled", False)
    state = SessionStates()
    assert telemetry.build_recorder("test_sess", state) is None


def test_state_backend_get_engine_return_None():
    """
    Verify that when the engine is not loaded (None), we get None back.
    """
    import src.backend.data_record.telemetry as telemetry

    state = SessionStates(backend=BackendState(engine=None))
    engine = state.backend.get_engine()
    assert engine == None

