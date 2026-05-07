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

# 1. We make one internet client for the whole app. 
# This is much faster than making a new one every time.
global_http_client = httpx.Client(timeout=5.0)

def refine_text_with_fallbacks(raw_text: str) -> str:
    """
    Clean the text by trying Groq, then Cerebras, then Together AI.
    If all of them break, just return the text with mistakes.
    
    Args:
        raw_text: The original spoken words.
        
    Returns:
        The cleaned words, or the original words if all AI computers are broken.
    """
    # 1. Do not do work if the text is empty
    if not raw_text or not raw_text.strip():
        return raw_text
        
    # 2. Try the primary AI: Groq
    try:
        return call_groq(global_http_client, raw_text)
    except httpx.HTTPError as error:
        log.warning(f"Groq network error: {error}")
    except ValueError as error:
        log.warning(f"Groq missing key: {error}")
        
    # 3. Try the first backup: Cerebras
    try:
        return call_cerebras(global_http_client, raw_text)
    except httpx.HTTPError as error:
        log.warning(f"Cerebras network error: {error}")
    except ValueError as error:
        log.warning(f"Cerebras missing key: {error}")
        
    # 4. Try the second backup: Together AI
    try:
        return call_together(global_http_client, raw_text)
    except httpx.HTTPError as error:
        log.warning(f"Together AI network error: {error}")
    except ValueError as error:
        log.warning(f"Together AI missing key: {error}")
        
    # 5. If everything failed, give back the original text so we don't lose data
    log.warning("All AI cleaners failed. Returning original text.")
    return raw_text
