"""
wizard.py — The professional interactive setup for VibeVoice.
This script runs in the foreground to gather all user settings using a Textual Bento TUI,
updates the .env file, and prepares the environment for the background processes.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.getcwd())

# Load existing .env before starting the UI so it can pre-populate
load_dotenv()

from src.utils.wizard_tui import WizardApp
from src.utils.env_manager import is_interactive

def run_wizard():
    """
    Launches the Textual Bento Wizard.
    If not interactive, skips to avoid crashing in background ( becuase in tests we are running the app in background ).
    """
    if not is_interactive():
        return

    app = WizardApp()
    app.run()

if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        # Textual handles keyboard interrupt gracefully,
        # but we keep this for extra safety.
        sys.exit(1)
    except (RuntimeError, OSError) as e:
        print(f"Error starting wizard: {e}")
        sys.exit(1)
