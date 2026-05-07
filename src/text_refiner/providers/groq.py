"""
This file handles sending text to the Groq API.
We use the Llama 3.3 70B model because it is very smart and fast.
"""
import os
import httpx
from src.text_refiner.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION, refine_user_prompt

def call_groq(client: httpx.Client, raw_text: str) -> str:
    """
    Send the raw text to Groq to fix grammar and spelling.
    
    Args:
        client: The HTTP client we use to connect to the internet.
        raw_text: The spoken text that might have mistakes.
        
    Returns:
        The cleaned text from the AI.
        
    Raises:
        ValueError: If the API key is missing.
        httpx.HTTPError: If the internet connection fails.
    """
    # 1. Get the secret password (API key) for Groq
    api_key = os.environ.get("GROQ_API_KEY")
        
    # 2. Setup the web address and login headers
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 3. Create the message package for the AI
    message_package = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": refine_user_prompt(raw_text)}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    
    # 4. Send the package and wait for the reply
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()
    
    # 5. Open the reply box and get the text message
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]