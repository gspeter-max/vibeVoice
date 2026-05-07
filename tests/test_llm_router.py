# tests/test_llm_router.py
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.text_refiner.llm_router import refine_text_with_fallbacks

def make_mock_http_error():
    """Helper to make a fake HTTP Error like a rate limit (429)"""
    request = httpx.Request("POST", "http://test")
    return httpx.HTTPStatusError("Rate Limit", request=request, response=httpx.Response(429, request=request))

@patch("src.text_refiner.llm_router.check_and_ask_for_api_key")
@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_rotates_on_failure(mock_together, mock_cerebras, mock_groq, mock_check_key):
    """Test that the router switches to Cerebras for the NEXT call if Groq fails."""
    import src.text_refiner.llm_router as router
    router.current_provider_index = 0 # Reset state
    
    # We must inject the mocks into the PROVIDERS list because it was built at module load time
    router.PROVIDERS[0]["call"] = mock_groq
    router.PROVIDERS[1]["call"] = mock_cerebras
    router.PROVIDERS[2]["call"] = mock_together
    
    # 1. First call fails on Groq
    mock_groq.side_effect = make_mock_http_error()
    result1 = router.refine_text_with_fallbacks("hello")
    
    assert result1.lower() == "hello" # Returns raw text immediately
    assert router.current_provider_index == 1 # Pointer moved to Cerebras
    
    # 2. Second call uses Cerebras
    mock_cerebras.return_value = "Cleaned by Cerebras"
    result2 = router.refine_text_with_fallbacks("world")
    
    assert result2 == "Cleaned by Cerebras"
    mock_cerebras.assert_called_once()
    mock_groq.assert_called_once() # Called only in the first attempt
    mock_together.assert_not_called()

@patch("src.text_refiner.llm_router.check_and_ask_for_api_key")
@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_full_rotation(mock_together, mock_cerebras, mock_groq, mock_check_key):
    """Test that it loops back to the start after all providers fail."""
    import src.text_refiner.llm_router as router
    router.current_provider_index = 0
    
    router.PROVIDERS[0]["call"] = mock_groq
    router.PROVIDERS[1]["call"] = mock_cerebras
    router.PROVIDERS[2]["call"] = mock_together
    
    mock_groq.side_effect = make_mock_http_error()
    mock_cerebras.side_effect = make_mock_http_error()
    mock_together.side_effect = make_mock_http_error()
    
    # Fail 3 times
    router.refine_text_with_fallbacks("1")
    router.refine_text_with_fallbacks("2")
    router.refine_text_with_fallbacks("3")
    
    # Should be back at 0 (Groq)
    assert router.current_provider_index == 0

def test_router_empty_input_returns_empty():
    """Test that empty strings return immediately without calling APIs."""
    from src.text_refiner.llm_router import refine_text_with_fallbacks
    result = refine_text_with_fallbacks("")
    assert result == ""
