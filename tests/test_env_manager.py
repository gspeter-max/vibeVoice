# tests/test_env_manager.py
import os
from unittest.mock import patch, mock_open
from src.utils.env_manager import check_and_ask_for_api_key

@patch("src.utils.env_manager.is_interactive", return_value=True)
@patch("src.utils.env_manager.Prompt.ask")
@patch("dotenv.load_dotenv")  # Patch __init__ export — that is what 'from dotenv import load_dotenv' resolves to
def test_check_and_ask_for_api_key_when_missing(mock_load_dotenv, mock_ask, mock_is_interactive):
    """
    Test that the function asks for the API key if it is missing 
    and writes it to the .env file.
    """
    mock_ask.return_value = "secret_key_123"
    
    m = mock_open()
    with patch.dict("os.environ", {}, clear=False):
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
            
        with patch("builtins.open", m):
            with patch("os.path.exists", return_value=False):
                check_and_ask_for_api_key("Groq", "GROQ_API_KEY")
        
    mock_ask.assert_called_once()
    m.assert_called_with(".env", "w")

    file_handle = m.return_value.__enter__.return_value
    file_handle.writelines.assert_called()

    written_lines = file_handle.writelines.call_args[0][0]
    written_data = "".join(written_lines)
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
