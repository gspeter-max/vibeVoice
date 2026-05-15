import pytest
import httpx
from unittest.mock import Mock
from src.text_refiner.providers.generic_openai_provider import call_openai_compatible_api

def test_call_openai_compatible_api_success():
    mock_client = Mock(spec=httpx.Client)
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Cleaned text"}}
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response

    result = call_openai_compatible_api(
        client=mock_client,
        api_key="test-key",
        url="https://api.test.com/v1",
        model="test-model",
        raw_text="dirty text"
    )

    assert result == "Cleaned text"
    # Verify request
    args, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "test-model"
    assert "dirty text" in str(kwargs["json"]["messages"])

def test_call_openai_compatible_api_http_error():
    mock_client = Mock(spec=httpx.Client)
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Error", request=Mock(), response=Mock(status_code=401)
    )
    mock_client.post.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError):
        call_openai_compatible_api(
            client=mock_client,
            api_key="wrong-key",
            url="https://api.test.com/v1",
            model="test-model",
            raw_text="text"
        )
