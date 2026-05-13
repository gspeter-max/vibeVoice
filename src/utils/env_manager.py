# src/utils/env_manager.py
"""
This file helps manage our secret API keys.
It can check if a key exists, ask the user for a new one, 
and save it to the .env file automatically.
"""
import os
import sys
from rich.console import Console
from rich.prompt import Prompt

# Initialize the Rich console for beautiful terminal output
console = Console()

def is_interactive() -> bool:
    """
    Checks if the current process is running in an interactive terminal.
    Verifies both stdout (output) and stdin (input) are connected to a TTY.
    """
    return console.is_terminal and sys.stdin.isatty()

def save_to_env(key: str, value: str) -> None:
    """
    Saves a key-value pair to the .env file idempotently.
    If the key exists, it updates it. If not, it appends it.
    """
    env_path = ".env"
    lines = []
    
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    # Normalize value and build the line
    new_line = f"{key}={value.strip()}\n"
    found = False
    updated_lines = []

    for line in lines:
        if line.startswith(f"{key}="):
            updated_lines.append(new_line)
            found = True
        else:
            updated_lines.append(line)

    if not found:
        # Ensure file ends with newline before appending
        if updated_lines and not updated_lines[-1].endswith("\n"):
            updated_lines.append("\n")
        updated_lines.append(new_line)

    with open(env_path, "w") as f:
        f.writelines(updated_lines)
    
    # Update current process environment
    os.environ[key] = value.strip()

def check_and_ask_for_api_key(provider_name: str, env_var_name: str) -> None:
    """
    Checks if an API key exists. If missing and interactive, prompts user to enter it.
    """
    from dotenv import load_dotenv

    # load_dotenv is idempotent — safe to call repeatedly, won't overwrite real os.environ values.
    load_dotenv()

    if os.environ.get(env_var_name):
        return

    # Key is missing — check if we can prompt the user.
    if not is_interactive():
        console.print(f"[bold red]❌ API key missing for {provider_name} and no terminal detected to ask.[/bold red]")
        return

    console.print(f"\n[bold yellow]⚠️  API key missing for {provider_name}.[/bold yellow]")
    console.print(f"You can get your key from the {provider_name} dashboard.\n")

    try:
        user_key = Prompt.ask(f"Please paste your [bold cyan]{provider_name} API Key[/bold cyan]")
        if user_key:
            save_to_env(env_var_name, user_key)
            console.print(f"[bold green]✅ Successfully saved {provider_name} API key![/bold green]\n")
    except EOFError:
        return
