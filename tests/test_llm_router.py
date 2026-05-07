# tests/test_llm_router.py
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.text_refiner.llm_router import refine_text_with_fallbacks

def make_mock_http_error():
    """Helper to make a fake HTTP Error like a rate limit (429)"""
    request = httpx.Request("POST", "http://test")
    return httpx.HTTPStatusError("Rate Limit", request=request, response=httpx.Response(429, request=request))

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_uses_groq_first(mock_together, mock_cerebras, mock_groq):
    """Test that the router tries Groq first and stops if successful."""
    mock_groq.return_value = "Cleaned by Groq"
    
    result = refine_text_with_fallbacks("hello")
    
    assert result == "Cleaned by Groq"
    mock_groq.assert_called_once()
    mock_cerebras.assert_not_called()
    mock_together.assert_not_called()

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_falls_back_to_cerebras_on_rate_limit(mock_together, mock_cerebras, mock_groq):
    """Test that the router tries Cerebras if Groq hits a rate limit."""
    mock_groq.side_effect = make_mock_http_error()
    mock_cerebras.return_value = "Cleaned by Cerebras"
    
    result = refine_text_with_fallbacks("hello")
    
    assert result == "Cleaned by Cerebras"
    mock_groq.assert_called_once()
    mock_cerebras.assert_called_once()
    mock_together.assert_not_called()

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_falls_back_to_together_on_timeout(mock_together, mock_cerebras, mock_groq):
    """Test that the router tries Together AI if both Groq and Cerebras fail."""
    mock_groq.side_effect = httpx.TimeoutException("Timeout")
    mock_cerebras.side_effect = httpx.TimeoutException("Timeout")
    mock_together.return_value = "Cleaned by Together"
    
    result = refine_text_with_fallbacks("hello")
    
    assert result == "Cleaned by Together"
    mock_groq.assert_called_once()
    mock_cerebras.assert_called_once()
    mock_together.assert_called_once()

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_returns_original_if_all_fail(mock_together, mock_cerebras, mock_groq):
    """Test that the router returns the exact original text if everything fails."""
    mock_groq.side_effect = make_mock_http_error()
    mock_cerebras.side_effect = make_mock_http_error()
    mock_together.side_effect = make_mock_http_error()
    
    result = refine_text_with_fallbacks("hello world")
    
    assert result == "hello world"

def test_router_empty_input_returns_empty():
    """Test that empty strings return immediately without calling APIs."""
    result = refine_text_with_fallbacks("")
    assert result == ""
