import sys
import pytest

def test_state_independent_of_brain():
    """
    Verify that state can be imported without brain.py.
    This ensures that the state.py module does not have any circular dependencies
    or accidental imports of the brain module.
    """
    if "src.backend.brain" in sys.modules:
        del sys.modules["src.backend.brain"]
    
    import src.backend.state as state
    assert "src.backend.brain" not in sys.modules
    
    # Verify global locks and store
    assert hasattr(state, "session_store")
    assert hasattr(state, "session_store_lock")
    assert hasattr(state, "backend_info")
    assert hasattr(state, "backend_lock")

def test_session_state_creation():
    """
    Verify state initialization edge cases.
    We test creating a session state with a missing engine (None)
    and ensure the recording state handles missing indexes correctly.
    """
    import src.backend.state as state
    # Create with None engine
    session = state.SessionState(engine=None)
    
    # Test getting recording state handles missing index
    rec = session.get_or_create_recording(1)
    assert rec.received_count == 0
    assert 1 in session.recordings
