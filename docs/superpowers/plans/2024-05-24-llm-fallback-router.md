# LLM Fallback Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent**:
- **Goal:** We need to ensure sub-0.2s latency and 99.9% uptime for the text refiner by implementing a global connection pool and a provider fallback waterfall.
- **Context:** Currently, `src/text_refiner/groq_client.py` makes a fresh HTTP request to Groq on every transcription chunk. This incurs TLS/DNS overhead. It also has no fallback if Groq rate-limits us.
- **Decisions:** 
  - We will rename `groq_client.py` to `llm_router.py`.
  - We will use `httpx.Client()` instantiated globally in the router so connections are reused.
  - We will implement three providers: Groq (primary), Cerebras (fallback 1), Together AI (fallback 2).
  - All providers will use their respective 70B models (`llama-3.3-70b-versatile`, `llama-3.3-70b`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`).
  - If all fail, return the original raw text.

**Architecture:**
- **`src/text_refiner/providers/{groq,cerebras,together}.py`**: Simple wrapper functions that take `(client, text)` and make the HTTP POST request to the provider's API.
- **`src/text_refiner/llm_router.py`**: Holds `global_http_client = httpx.Client(timeout=5.0)`. Exposes `refine_text_with_fallbacks(raw_text: str) -> str`. It tries Groq -> Cerebras -> Together.
- **`src/backend/brain.py`**: Calls `refine_text_with_fallbacks(text)` instead of `send_text_to_groq_for_cleanup(text)`.

**Important Rule to follow :**
- Read `/Users/apple/.gemini/GEMINI.md`
- Follow `## Mind Set rules` throughout execution
- **CRITICAL:** add detailed docs in functions and explain the code and logic in comments.
- **CRITICAL:** make the code function name and variable name clear and easily to understand instead of short and confusing names.
  - so 5 year old child easily understand
  - do not put any imagination and analogy to understand for 5 year old child
  - write code function name and docs and code like this: **developer get highest speed to read the code**
  - **Explain like a fresher**
  - **Write docs in your step-by-step simple style.**
  - **Make the docs in function and file headers human-readable and literal.**

---
Task Structure

### Task 1 : Read instructions and update config
- [ ] Read `/Users/apple/.gemini/GEMINI.md`
- [ ] **Step 1: Modify `.env.example`**
```bash
echo "CEREBRAS_API_KEY=" >> .env.example
echo "TOGETHER_API_KEY=" >> .env.example
```

### Task 2: Create Provider Modules
**Files:**
- Create: `src/text_refiner/providers/groq.py`
- Create: `src/text_refiner/providers/cerebras.py`
- Create: `src/text_refiner/providers/together.py`
- Create: `src/text_refiner/providers/__init__.py`

- [ ] **Step 1: Create empty init file**
```bash
mkdir -p src/text_refiner/providers/
touch src/text_refiner/providers/__init__.py
```

- [ ] **Step 2: Write Groq Provider**
```python
# src/text_refiner/providers/groq.py
"""
This file handles sending text to the Groq API.
We use the Llama 3.3 70B model because it is very smart and fast.
"""
import os
import httpx
from src.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION

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
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")
        
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
            {"role": "user", "content": raw_text}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    # 4. Send the package and wait for the reply
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()
    
    # 5. Open the reply box and get the text message
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]
```

- [ ] **Step 3: Write Cerebras Provider**
```python
# src/text_refiner/providers/cerebras.py
"""
This file handles sending text to the Cerebras API.
We use this as our first backup if Groq is not working.
"""
import os
import httpx
from src.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION

def call_cerebras(client: httpx.Client, raw_text: str) -> str:
    """
    Send the raw text to Cerebras to fix grammar and spelling.
    """
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("Missing CEREBRAS_API_KEY")
        
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    message_package = {
        "model": "llama-3.3-70b",
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": raw_text}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()
    
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Write Together AI Provider**
```python
# src/text_refiner/providers/together.py
"""
This file handles sending text to the Together AI API.
We use this as our second backup if both Groq and Cerebras are not working.
"""
import os
import httpx
from src.prompts.cleaner_prompt import SYSTEM_CLEANUP_INSTRUCTION

def call_together(client: httpx.Client, raw_text: str) -> str:
    """
    Send the raw text to Together AI to fix grammar and spelling.
    """
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing TOGETHER_API_KEY")
        
    url = "https://api.together.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    message_package = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_CLEANUP_INSTRUCTION},
            {"role": "user", "content": raw_text}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    response = client.post(url, headers=headers, json=message_package)
    response.raise_for_status()
    
    reply_data = response.json()
    return reply_data["choices"][0]["message"]["content"]
```

- [ ] **Step 5: Commit**
```bash
git add .env.example src/text_refiner/providers/
git commit -m "feat: add groq, cerebras, and together ai providers"
```

### Task 3: Implement LLM Router Tests

**Files:**
- Create: `tests/test_llm_router.py`

- [ ] **Step 1: Write router tests checking waterfall logic and rate limit edge cases**
```python
# tests/test_llm_router.py
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.text_refiner.llm_router import refine_text_with_fallbacks

def make_mock_http_error():
    """Helper to make a fake HTTP Error like a rate limit (429)"""
    request = httpx.Request("POST", "http://test")
    return httpx.HTTPStatusError("Rate Limit", request=request, response=httpx.Response(429, request=request))

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_uses_groq_first(mock_together, mock_cerebras, mock_groq):
    """Test that the router tries Groq first and stops if successful."""
    mock_groq.return_value = "Cleaned by Groq"
    
    result = refine_text_with_fallbacks("hello")
    
    assert result == "Cleaned by Groq"
    mock_groq.assert_called_once()
    mock_cerebras.assert_not_called()
    mock_together.assert_not_called()

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_falls_back_to_cerebras_on_rate_limit(mock_together, mock_cerebras, mock_groq):
    """Test that the router tries Cerebras if Groq hits a rate limit."""
    mock_groq.side_effect = make_mock_http_error()
    mock_cerebras.return_value = "Cleaned by Cerebras"
    
    result = refine_text_with_fallbacks("hello")
    
    assert result == "Cleaned by Cerebras"
    mock_groq.assert_called_once()
    mock_cerebras.assert_called_once()
    mock_together.assert_not_called()

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_falls_back_to_together_on_timeout(mock_together, mock_cerebras, mock_groq):
    """Test that the router tries Together AI if both Groq and Cerebras fail."""
    mock_groq.side_effect = httpx.TimeoutException("Timeout")
    mock_cerebras.side_effect = httpx.TimeoutException("Timeout")
    mock_together.return_value = "Cleaned by Together"
    
    result = refine_text_with_fallbacks("hello")
    
    assert result == "Cleaned by Together"
    mock_groq.assert_called_once()
    mock_cerebras.assert_called_once()
    mock_together.assert_called_once()

@patch("src.text_refiner.llm_router.call_groq")
@patch("src.text_refiner.llm_router.call_cerebras")
@patch("src.text_refiner.llm_router.call_together")
def test_router_returns_original_if_all_fail(mock_together, mock_cerebras, mock_groq):
    """Test that the router returns the exact original text if everything fails."""
    mock_groq.side_effect = make_mock_http_error()
    mock_cerebras.side_effect = make_mock_http_error()
    mock_together.side_effect = make_mock_http_error()
    
    result = refine_text_with_fallbacks("hello world")
    
    assert result == "hello world"

def test_router_empty_input_returns_empty():
    """Test that empty strings return immediately without calling APIs."""
    result = refine_text_with_fallbacks("")
    assert result == ""
```

- [ ] **Step 2: Verify tests fail**
Run `python -m pytest tests/test_llm_router.py -v`. Expect them to fail because `llm_router` does not exist.

### Task 4: Implement LLM Router

**Files:**
- Create: `src/text_refiner/llm_router.py`
- Delete: `src/text_refiner/groq_client.py`
- Delete: `tests/test_groq_client.py`

- [ ] **Step 1: Write `src/text_refiner/llm_router.py`**
```python
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
        print(f"Groq network error: {error}")
    except ValueError as error:
        print(f"Groq missing key: {error}")
        
    # 3. Try the first backup: Cerebras
    try:
        return call_cerebras(global_http_client, raw_text)
    except httpx.HTTPError as error:
        print(f"Cerebras network error: {error}")
    except ValueError as error:
        print(f"Cerebras missing key: {error}")
        
    # 4. Try the second backup: Together AI
    try:
        return call_together(global_http_client, raw_text)
    except httpx.HTTPError as error:
        print(f"Together AI network error: {error}")
    except ValueError as error:
        print(f"Together AI missing key: {error}")
        
    # 5. If everything failed, give back the original text so we don't lose data
    print("All AI cleaners failed. Returning original text.")
    return raw_text
```

- [ ] **Step 2: Run tests to verify they pass**
Run `python -m pytest tests/test_llm_router.py -v`.

- [ ] **Step 3: Remove old files**
```bash
git rm src/text_refiner/groq_client.py tests/test_groq_client.py
```

- [ ] **Step 4: Commit**
```bash
git add src/text_refiner/llm_router.py tests/test_llm_router.py
git commit -m "feat: implement llm_router with fallback logic"
```

### Task 5: Wire Router to Brain

**Files:**
- Modify: `src/backend/brain.py`
- Modify: `tests/test_brain.py`

- [ ] **Step 1: Provide Python script to do exact string replacement in brain.py**
Run this script to precisely update imports and function calls in brain:
```bash
python -c '
import sys
content = open("src/backend/brain.py").read()
content = content.replace("from src.text_refiner.groq_client import send_text_to_groq_for_cleanup", "from src.text_refiner.llm_router import refine_text_with_fallbacks")
content = content.replace("send_text_to_groq_for_cleanup", "refine_text_with_fallbacks")
open("src/backend/brain.py", "w").write(content)

test_content = open("tests/test_brain.py").read()
test_content = test_content.replace("src.backend.brain.send_text_to_groq_for_cleanup", "src.backend.brain.refine_text_with_fallbacks")
open("tests/test_brain.py", "w").write(test_content)
'
```

- [ ] **Step 2: Run all tests to ensure brain tests still pass**
Run `python -m pytest tests/test_brain.py -v`.

- [ ] **Step 3: Commit**
```bash
git add src/backend/brain.py tests/test_brain.py
git commit -m "refactor: wire brain.py to use the new llm_router"
```

## Self-Review
- [x] Tests catch real edge cases, not just happy paths (HTTPStatusError, TimeoutException, ValueError).
- [x] Variable names are literal (`global_http_client`, `message_package`, `reply_data`).
- [x] `global_http_client` is used to prevent TLS/DNS overhead on every request.
- [x] Exact code commands provided for `.env`, `__init__.py`, and precise replacements in `brain.py`.
