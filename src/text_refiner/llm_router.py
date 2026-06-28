"""AI text cleanup router using multiple fallback providers."""

import os
import threading
from typing import Any

import httpx

from src import log
from src.text_refiner.providers.openai import call_api
from src.utils.env_manager import check_and_ask_for_api_key

PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
    },
    {
        "name": "Cerebras",
        "env_var": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-8b",
    },
    {
        "name": "OpenGateway",
        "env_var": "OPENGATEWAY_API_KEY",
        "url": "https://opengateway.gitlawb.com/v1/chat/completions",
        "model": "mimo-v2.5-pro",
        "timeout": 15.0,
    },
]

provider_idx = 0
provider_lock = threading.Lock()
http_client = httpx.Client(timeout=4.0)


def set_provider(idx: int) -> None:
    """Set the primary provider index."""
    global provider_idx
    if 0 <= idx < len(PROVIDERS):
        with provider_lock:
            provider_idx = idx
        log.info(f"LLM Router: Primary provider set to {PROVIDERS[idx]['name']}")


def refine(text: str) -> str:
    """Clean text using the active provider with automatic fallback on failure."""
    global provider_idx

    if not text or not text.strip():
        return text

    with provider_lock:
        provider = PROVIDERS[provider_idx]
    name = provider["name"]
    env_var = provider["env_var"]

    try:
        check_and_ask_for_api_key(name, env_var)
        log.info(f"LLM Router: Using {name} for cleanup.")
        key = os.environ.get(env_var, "")

        return call_api(
            client=http_client,
            key=key,
            url=provider["url"],
            model=provider["model"],
            text=text,
            timeout=provider.get("timeout", 4.0),
        )
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException, ValueError) as error:
        with provider_lock:
            provider_idx = (provider_idx + 1) % len(PROVIDERS)
            next_provider = PROVIDERS[provider_idx]["name"]

        log.warning(f"LLM Router: {name} failed. Rotating to {next_provider}. Error: {error}")
        return text
