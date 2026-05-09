"""
This file handles sending text to the Cerebras API.
We use this as our first backup if Groq is not working.
"""
import os
import httpx
from src.text_refiner.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION , refine_user_prompt 

def call_cerebras(client: httpx.Client, raw_text: str) -> str:
    """
    Send the raw text to Cerebras to fix grammar and spelling.
    
    Args:
        client: The HTTP client for the internet.
        raw_text: The original text.
        
    Returns:
        Cleaned text.
    """
    # 1. Get the secret API key
    api_key = os.environ.get("CEREBRAS_API_KEY")
        
    # 2. Setup address and headers
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 3. Create the message package
    message_package = {
        "model": "llama-3.3-70b",
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": refine_user_prompt(raw_text)}
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    
    # 4. Send and wait for reply
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()
    
    # 5. Extract text from reply
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]
