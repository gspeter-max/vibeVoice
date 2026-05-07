# tests/test_env_manager.py
import os
from unittest.mock import patch, MagicMock, mock_open
import pytest
from src.utils.env_manager import check_and_ask_for_api_key

@patch("src.utils.env_manager.Prompt.ask")
def test_check_and_ask_for_api_key_when_missing(mock_ask):
    """
    Test that the function asks for the API key if it is missing 
    and writes it to the .env file.
    """
    # 1. Setup: API key is not in environment
    mock_ask.return_value = "secret_key_123"
    
    # 2. Mock the file opening and environment
    m = mock_open()
    with patch.dict("os.environ", {}, clear=False):
        # Ensure the key is NOT there
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
            
        with patch("builtins.open", m):
            with patch("os.path.exists", return_value=False):
                # 3. Call the function
                check_and_ask_for_api_key("Groq", "GROQ_API_KEY")
        
    # 4. Check that it asked the user
    mock_ask.assert_called_once()
    
    # 5. Check that it tried to write to .env
    # It should open in 'a' (append) mode
    m.assert_called_with(".env", "a")
    handle = m()
    handle.write.assert_called()
    
    # Check that the written string contains the key and value
    written_data = "".join(call.args[0] for call in handle.write.call_args_list)
    assert "GROQ_API_KEY=secret_key_123" in written_data

def test_check_and_ask_for_api_key_when_exists():
    """
    Test that the function does nothing if the API key is already there.
    """
    # 1. Setup: API key is already in environment
    with patch.dict("os.environ", {"GROQ_API_KEY": "already_here"}):
        with patch("src.utils.env_manager.Prompt.ask") as mock_ask:
            check_and_ask_for_api_key("Groq", "GROQ_API_KEY")
            
            # 2. Check that it did NOT ask the user
            mock_ask.assert_not_called()
