# src/utils/env_manager.py
"""
This file helps manage our secret API keys.
It can check if a key exists, ask the user for a new one, 
and save it to the .env file automatically.
"""
import os
from rich.console import Console
from rich.prompt import Prompt

# Initialize the Rich console for beautiful terminal output
console = Console()

def check_and_ask_for_api_key(provider_name: str, env_var_name: str) -> None:
    """
    Checks if an API key exists in the environment or .env file.
    If it is missing, it asks the user to paste it and saves it.
    
    Args:
        provider_name: The name of the AI company (e.g., "Groq").
        env_var_name: The name of the secret variable (e.g., "GROQ_API_KEY").
    """
    # 1. First, check if the key is already loaded in the environment
    if os.environ.get(env_var_name):
        return

    # 2. If not in environment, try to see if it's in the .env file but not loaded
    if os.path.exists(".env"):
        with open(".env", "r") as env_file:
            for line in env_file:
                # Look for lines like "KEY=VALUE"
                if line.startswith(f"{env_var_name}="):
                    # Found it! Extract the value and save it to the environment
                    value = line.strip().split("=", 1)[1]
                    os.environ[env_var_name] = value
                    return

    # 3. If we are here, the key is really missing.
    # We warn the user using beautiful colors.
    console.print(f"\n[bold yellow]⚠️  API key missing for {provider_name}.[/bold yellow]")
    console.print(f"You can get your key from the {provider_name} dashboard.\n")

    # 4. Ask the user to paste their key
    # Prompt.ask will wait for the user to type and press Enter
    user_key = Prompt.ask(f"Please paste your [bold cyan]{provider_name} API Key[/bold cyan]")

    if user_key:
        # 5. Save the key to the .env file so we don't ask again next time
        # We use "a" to append to the end of the file
        with open(".env", "a") as env_file:
            # Add a newline first to be safe
            env_file.write(f"\n{env_var_name}={user_key.strip()}\n")
        
        # 6. Also put it in the current session memory (environment)
        os.environ[env_var_name] = user_key.strip()
        
        console.print(f"[bold green]✅ Successfully saved {provider_name} API key to .env![/bold green]\n")
