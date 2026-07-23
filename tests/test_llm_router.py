# tests/test_llm_router.py
from unittest.mock import MagicMock, patch

import httpx

from src.text_refiner.llm_router import refine


def make_mock_http_error():
    """Helper to make a fake HTTP Error like a rate limit (429)"""
    request = httpx.Request("POST", "http://test")
    return httpx.HTTPStatusError(
        "Rate Limit", request=request, response=httpx.Response(429, request=request)
    )


def test_router_rotates_on_failure():
    """Test that the router switches to Cerebras for the NEXT call if Groq fails."""
    import src.text_refiner.llm_router as router

    with patch("src.text_refiner.llm_router.check_and_ask_for_api_key") as mock_check_key, \
         patch("src.text_refiner.llm_router.call_api") as mock_call_api, \
         patch("src.text_refiner.llm_router.os.environ.get") as mock_env_get:

        router.provider_idx = 0  # Reset state
        mock_env_get.return_value = "fake-key"

        # 1. First call fails on Groq
        mock_call_api.side_effect = make_mock_http_error()
        result1 = router.refine("hello")

        assert result1.lower() == "hello"  # Returns raw text immediately
        assert router.provider_idx == 1  # Pointer moved to Cerebras

        # Verify first call used Groq config
        args, kwargs = mock_call_api.call_args_list[0]
        assert kwargs["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert kwargs["model"] == "llama-3.3-70b-versatile"

        # 2. Second call uses Cerebras
        mock_call_api.side_effect = None
        mock_call_api.return_value = "Cleaned by Cerebras"
        result2 = router.refine("world")

        assert result2 == "Cleaned by Cerebras"
        assert mock_call_api.call_count == 2

        # Verify second call used Cerebras config
        args, kwargs = mock_call_api.call_args_list[1]
        assert kwargs["url"] == "https://api.cerebras.ai/v1/chat/completions"
        assert kwargs["model"] == "llama3.1-8b"


def test_router_full_rotation():
    """Test that it loops back to Groq (index 0) after all providers fail."""
    import src.text_refiner.llm_router as router

    with patch("src.text_refiner.llm_router.check_and_ask_for_api_key") as mock_check_key, \
         patch("src.text_refiner.llm_router.call_api") as mock_call_api, \
         patch("src.text_refiner.llm_router.os.environ.get") as mock_env_get:

        router.provider_idx = 0
        mock_env_get.return_value = "fake-key"

        mock_call_api.side_effect = make_mock_http_error()

        # Fail all providers once each to complete a full rotation back to index 0
        router.refine("1")  # Groq fails → index 1 (Cerebras)
        router.refine("2")  # Cerebras fails → index 2 (OpenGateway)
        router.refine("3")  # OpenGateway fails → index 0 (Groq)

        # Should be back at 0 (Groq) after a full rotation through all providers
        assert router.provider_idx == 0


def test_router_empty_input_returns_empty():
    """Test that empty strings return immediately without calling APIs."""
    result = refine("")
    assert result == ""
