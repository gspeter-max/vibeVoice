import os
import pytest
from unittest.mock import patch
from src.utils.env_utils import get_integer_from_environment, get_float_from_environment

def test_get_integer_from_environment_returns_fallback_when_missing():
    """Verifies that a missing environment variable returns the provided default integer."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_integer_from_environment("MISSING_KEY", 42) == 42

def test_get_integer_from_environment_returns_fallback_when_empty():
    """Verifies that an empty string in the environment returns the default integer."""
    with patch.dict(os.environ, {"EMPTY_KEY": ""}):
        assert get_integer_from_environment("EMPTY_KEY", 42) == 42

def test_get_integer_from_environment_returns_fallback_when_whitespace_only():
    """Verifies that a string containing only spaces returns the default integer."""
    with patch.dict(os.environ, {"SPACE_KEY": "   "}):
        assert get_integer_from_environment("SPACE_KEY", 42) == 42

def test_get_integer_from_environment_returns_fallback_when_invalid_string():
    """Verifies that a non-numeric string returns the default integer instead of crashing."""
    with patch.dict(os.environ, {"BAD_KEY": "abc"}):
        assert get_integer_from_environment("BAD_KEY", 42) == 42

def test_get_integer_from_environment_returns_parsed_value_when_valid():
    """Verifies that a valid numeric string is correctly converted to an integer."""
    with patch.dict(os.environ, {"GOOD_KEY": "10"}):
        assert get_integer_from_environment("GOOD_KEY", 42) == 10

def test_get_integer_from_environment_strips_whitespace_before_parsing():
    """Verifies that leading/trailing spaces are removed before attempting to parse."""
    with patch.dict(os.environ, {"PADDED_KEY": "  15  "}):
        assert get_integer_from_environment("PADDED_KEY", 42) == 15

def test_get_float_from_environment_returns_fallback_when_missing():
    """Verifies that a missing environment variable returns the provided default float."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_float_from_environment("MISSING_KEY", 3.14) == 3.14

def test_get_float_from_environment_returns_fallback_when_empty():
    """Verifies that an empty string in the environment returns the default float."""
    with patch.dict(os.environ, {"EMPTY_KEY": ""}):
        assert get_float_from_environment("EMPTY_KEY", 3.14) == 3.14

def test_get_float_from_environment_returns_fallback_when_invalid_string():
    """Verifies that a malformed decimal string returns the default float."""
    with patch.dict(os.environ, {"BAD_KEY": "abc"}):
        assert get_float_from_environment("BAD_KEY", 3.14) == 3.14

def test_get_float_from_environment_returns_parsed_value_when_valid():
    """Verifies that a valid decimal string is correctly converted to a float."""
    with patch.dict(os.environ, {"GOOD_KEY": "2.5"}):
        assert get_float_from_environment("GOOD_KEY", 3.14) == 2.5
