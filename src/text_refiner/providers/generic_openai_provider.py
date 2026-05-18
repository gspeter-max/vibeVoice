"""
This module provides a generic function to call any OpenAI-compatible API.
It is used to consolidate logic for providers like Groq, Cerebras, and NVIDIA.
"""
import httpx
from src.text_refiner.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION, refine_user_prompt

def call_openai_compatible_api(
    client: httpx.Client,
    api_key: str,
    url: str,
    model: str,
    raw_text: str
) -> str:
    """
    Send raw text to an OpenAI-compatible API to fix grammar and spelling.

    Args:
        client: The shared HTTP client used for network requests.
        api_key: The secret API key (password) for the provider.
        url: The full API endpoint URL (e.g., https://api.groq.com/openai/v1/chat/completions).
        model: The specific AI model name to use (e.g., llama-3.3-70b-versatile).
        raw_text: The spoken text that needs cleaning.

    Returns:
        The cleaned text returned by the AI model.

    Raises:
        httpx.HTTPError: If the internet connection fails or the server returns an error.
        ValueError: If the API response has an unexpected structure.
    """

    # 1. Setup the login headers
    #    We use the standard Bearer token authentication used by OpenAI-compatible APIs.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 2. Create the message package (JSON body)
    #    We follow the OpenAI Chat Completions schema.
    message_package = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": refine_user_prompt(raw_text)}
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    # 3. Send the package to the provider and wait for the reply
    #    The router provides a shared global_http_client with a 4s timeout.
    response = client.post(url, headers=headers, json=message_package)

    # 4. Check if the request was successful
    #    If not, this will raise an httpx.HTTPStatusError which the router will catch.
    response.raise_for_status()

    # 5. Extract the cleaned text from the JSON response
    #    The response follows the standard OpenAI structure: choices[0].message.content
    reply_data = response.json()
    try:
        cleaned_text = reply_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected API response structure: {e}") from e

    return cleaned_text
