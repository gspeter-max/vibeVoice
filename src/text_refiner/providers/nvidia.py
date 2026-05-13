"""
This file handles sending text to the NVIDIA NIM API.
We use the Nemotron Super 49B V1 model — NVIDIA's instruction-tuned Llama 3.3.
NVIDIA NIM is OpenAI-compatible, so we use the same httpx pattern
as our other providers — no new SDK dependency needed.
"""
import os
import httpx
from src.text_refiner.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION, refine_user_prompt


def call_nvidia(client: httpx.Client, raw_text: str) -> str:
    """
    Send the raw text to NVIDIA NIM (Nemotron Super 49B V1) to fix grammar and spelling.

    Nemotron Super 49B is NVIDIA's instruction-tuned version of Llama 3.3.
    It follows instructions precisely — ideal for our grammar cleanup task.

    Args:
        client: The shared HTTP client used across all providers.
        raw_text: The spoken text that may have grammar or spelling mistakes.

    Returns:
        The cleaned text from the AI model.

    Raises:
        ValueError: If the NVIDIA_API_KEY is missing from environment.
        httpx.HTTPError: If the network request fails or times out.
    """
    # 1. Read the NVIDIA API key from the environment
    #    The key is stored as NVIDIA_API_KEY in the .env file
    api_key = os.environ.get("NVIDIA_API_KEY")

    # 2. Set the NVIDIA NIM endpoint and authorization header
    #    NVIDIA NIM is fully OpenAI-compatible at this base URL
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 3. Build the request body for Nemotron Super 49B V1
    #    - stream: False → we want one full response, not chunks
    message_package = {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": refine_user_prompt(raw_text)},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": False,
    }

    # 4. Send the request and raise an error if the server returns a bad status
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()

    # 5. Parse the response and return the cleaned text content
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]
