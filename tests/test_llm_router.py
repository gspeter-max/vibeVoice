# tests/test_llm_router.py
import httpx
from unittest.mock import patch, MagicMock
from src.text_refiner.llm_router import refine_text_with_fallbacks

def make_mock_http_error():
    """Helper to make a fake HTTP Error like a rate limit (429)"""
    request = httpx.Request("POST", "http://test")
    return httpx.HTTPStatusError("Rate Limit", request=request, response=httpx.Response(429, request=request))

@patch("src.text_refiner.llm_router.check_and_ask_for_api_key")
@patch("src.text_refiner.llm_router.call_openai_compatible_api")
@patch("src.text_refiner.llm_router.os.environ.get")
def test_router_rotates_on_failure(mock_env_get, mock_call_api, mock_check_key):
    """Test that the router switches to Cerebras for the NEXT call if Groq fails."""
    import src.text_refiner.llm_router as router
    router.current_provider_index = 0  # Reset state
    mock_env_get.return_value = "fake-key"

    # 1. First call fails on Groq
    mock_call_api.side_effect = make_mock_http_error()
    result1 = router.refine_text_with_fallbacks("hello")

    assert result1.lower() == "hello"  # Returns raw text immediately
    assert router.current_provider_index == 1  # Pointer moved to Cerebras
    
    # Verify first call used Groq config
    # router.PROVIDERS[0] is Groq
    args, kwargs = mock_call_api.call_args_list[0]
    assert kwargs["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert kwargs["model"] == "llama-3.3-70b-versatile"

    # 2. Second call uses Cerebras
    mock_call_api.side_effect = None
    mock_call_api.return_value = "Cleaned by Cerebras"
    result2 = router.refine_text_with_fallbacks("world")

    assert result2 == "Cleaned by Cerebras"
    assert mock_call_api.call_count == 2
    
    # Verify second call used Cerebras config
    args, kwargs = mock_call_api.call_args_list[1]
    assert kwargs["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert kwargs["model"] == "llama3.1-8b"


@patch("src.text_refiner.llm_router.check_and_ask_for_api_key")
@patch("src.text_refiner.llm_router.call_openai_compatible_api")
@patch("src.text_refiner.llm_router.os.environ.get")
def test_router_full_rotation(mock_env_get, mock_call_api, mock_check_key):
    """Test that it loops back to Groq (index 0) after both providers fail."""
    import src.text_refiner.llm_router as router
    router.current_provider_index = 0
    mock_env_get.return_value = "fake-key"

    mock_call_api.side_effect = make_mock_http_error()

    # Fail both providers once each to complete a full rotation back to index 0
    router.refine_text_with_fallbacks("1")  # Groq fails → index moves to 1 (Cerebras)
    router.refine_text_with_fallbacks("2")  # Cerebras fails → index wraps to 0 (Groq)

    # Should be back at 0 (Groq) after a full rotation through both providers
    assert router.current_provider_index == 0

def test_router_empty_input_returns_empty():
    """Test that empty strings return immediately without calling APIs."""
    result = refine_text_with_fallbacks("")
    assert result == ""
