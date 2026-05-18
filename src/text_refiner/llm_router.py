# src/text_refiner/llm_router.py
"""
This file is the main door to all AI text cleaners.
It holds one internet connection open to save time.
It tries different AI companies in order so we never fail.
"""
import os
import threading
import httpx
from src.text_refiner.providers.generic_openai_provider import call_openai_compatible_api

from src import log
from src.utils.env_manager import check_and_ask_for_api_key

# 1. Configuration for our Providers
# Fallback order: Groq → Cerebras.
# If Groq fails, we rotate to Cerebras automatically.
PROVIDERS = [
    {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "description": "Fastest Performance",
        "feature": "Ultra-low latency"
    },
    {
        "name": "Cerebras",
        "env_var": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-8b",
        "description": "Fast Backup",
        "feature": "Llama 3.1 8B on Cerebras hardware"
    },
]

# 2. State to remember who is the current leader.
# We start with Groq (index 0).
current_provider_index = 0
# Lock to protect current_provider_index from concurrent read/modify/write races.
_provider_lock = threading.Lock()
# 3. We make one internet client for the whole app.
# We use a 4.0s timeout to ensure we stay under the 5.0s user limit.
global_http_client = httpx.Client(timeout=4.0)

def set_primary_provider(index: int) -> None:
    """
    Sets which AI provider we should try to use first.

    Args:
        index: The position in the PROVIDERS list (0 or 1).
    """
    global current_provider_index
    if 0 <= index < len(PROVIDERS):
        with _provider_lock:
            current_provider_index = index
        log.info(f"LLM Router: Primary provider set to {PROVIDERS[index]['name']}")

def refine_text_with_fallbacks(raw_text: str) -> str:
    """
    Clean the text using the current leader provider.
    If the leader is slow or broken, we return raw text INSTANTLY
     and switch leaders for the next time.

    Args:
        raw_text: The original spoken words.

    Returns:
        The cleaned words, or the original words if the current AI is broken.
    """
    global current_provider_index

    # 1. Do not do work if the text is empty
    if not raw_text or not raw_text.strip():
        return raw_text

    # 2. Get the current provider info (read index under lock for thread safety)
    with _provider_lock:
        provider = PROVIDERS[current_provider_index]
    provider_name = provider["name"]
    provider_env_var = provider["env_var"]

    # 3. Try to clean the text
    try:
        # Guarantee that we have an API key before calling.
        # check_and_ask_for_api_key ensures the key is present in os.environ.
        check_and_ask_for_api_key(provider_name, provider_env_var)

        log.info(f"LLM Router: Using {provider_name} for cleanup.")

        # We read directly from os.environ to ensure we get the latest key
        # if it was added during runtime.
        api_key = os.environ.get(provider_env_var, "")

        return call_openai_compatible_api(
            client=global_http_client,
            api_key=api_key,
            url=provider["url"],
            model=provider["model"],
            raw_text=raw_text
        )
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException, ValueError) as error:
        # 4. If it fails (timeout, network, or key), we rotate to the next provider
        with _provider_lock:
            current_provider_index = (current_provider_index + 1) % len(PROVIDERS)
            next_provider = PROVIDERS[current_provider_index]["name"]

        log.warning(
            f"LLM Router: {provider_name} failed. "
            f"Rotating to {next_provider}. Error: {error}"
        )

        # 5. Return raw text immediately so the user gets their text without delay
        return raw_text
