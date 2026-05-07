# src/text_refiner/llm_router.py
"""
This file is the main door to all AI text cleaners.
It holds one internet connection open to save time.
It tries different AI companies in order so we never fail.
"""
import httpx
from typing import Optional
from src.text_refiner.providers.groq import call_groq
from src.text_refiner.providers.cerebras import call_cerebras
from src.text_refiner.providers.together import call_together

from src import log
from src.utils.env_manager import check_and_ask_for_api_key

# 1. Configuration for our Providers
# We list them in order: Groq, Cerebras, Together AI.
PROVIDERS = [
    {
        "name": "Groq", 
        "call": call_groq, 
        "env_var": "GROQ_API_KEY",
        "description": "Fastest Performance",
        "feature": "Ultra-low latency"
    },
    {
        "name": "Cerebras", 
        "call": call_cerebras, 
        "env_var": "CEREBRAS_API_KEY",
        "description": "High-quality Alternative",
        "feature": "Llama-3.1 70B support"
    },
    {
        "name": "Together AI", 
        "call": call_together, 
        "env_var": "TOGETHER_API_KEY",
        "description": "Versatile Models",
        "feature": "Large context window"
    }
]

# 2. State to remember who is the current leader.
# We start with Groq (index 0).
current_provider_index = 0

def set_primary_provider(index: int) -> None:
    """
    Sets which AI provider we should try to use first.
    
    Args:
        index: The position in the PROVIDERS list (0, 1, or 2).
    """
    global current_provider_index
    if 0 <= index < len(PROVIDERS):
        current_provider_index = index
        log.info(f"LLM Router: Primary provider set to {PROVIDERS[index]['name']}")

# 3. We make one internet client for the whole app. 
# We use a 4.0s timeout to ensure we stay under the 5.0s user limit.
global_http_client = httpx.Client(timeout=4.0)

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
        
    # 2. Get the current provider info
    provider = PROVIDERS[current_provider_index]
    provider_name = provider["name"]
    provider_function = provider["call"]
    provider_env_var = provider["env_var"]

    # 3. Try to clean the text
    try:
        # Guarantee that we have an API key before calling
        check_and_ask_for_api_key(provider_name, provider_env_var)
        
        log.info(f"LLM Router: Using {provider_name} for cleanup.")
        return provider_function(global_http_client, raw_text)
    except Exception as error:
        # 4. If it fails (timeout, network, or key), we rotate to the next provider
        current_provider_index = (current_provider_index + 1) % len(PROVIDERS)
        next_provider = PROVIDERS[current_provider_index]["name"]
        
        log.warning(
            f"LLM Router: {provider_name} failed. "
            f"Rotating to {next_provider}. Error: {error}"
        )
        
        # 5. Return raw text immediately so the user gets their text without delay
        return raw_text
