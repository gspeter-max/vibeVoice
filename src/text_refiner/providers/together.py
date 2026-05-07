"""
This file handles sending text to the Together AI API.
We use this as our second backup if both Groq and Cerebras are not working.
"""
import os
import httpx
from src.text_refiner.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION

def call_together(client: httpx.Client, raw_text: str) -> str:
    """
    Send the raw text to Together AI to fix grammar and spelling.
    
    Args:
        client: The HTTP client for the internet.
        raw_text: The original text.
        
    Returns:
        Cleaned text.
    """
    # 1. Get the secret API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing TOGETHER_API_KEY")
        
    # 2. Setup address and headers
    url = "https://api.together.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 3. Create the message package
    message_package = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": raw_text}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    
    # 4. Send and wait for reply
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()
    
    # 5. Extract text from reply
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]
