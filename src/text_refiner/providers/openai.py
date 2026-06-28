"""Call OpenAI-compatible APIs."""

import httpx

from src.text_refiner.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION, refine_user_prompt


def call_api(
    client: httpx.Client,
    key: str,
    url: str,
    model: str,
    text: str,
    timeout: float = 4.0,
) -> str:
    """Send text to an OpenAI-compatible API to refine grammar and spelling."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": refine_user_prompt(text)},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    response = client.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected API response structure: {e}") from e
