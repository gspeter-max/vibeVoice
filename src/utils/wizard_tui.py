"""
wizard_tui.py — The Textual-based Bento Grid setup for VibeVoice.
Provides a modern, reactive interface for configuring providers, microphones, and models.
"""

import os
import pyaudio
from typing import Optional, List, Tuple
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Static, Button, Label, Input, Switch, Select
from textual.screen import ModalScreen
from textual.binding import Binding

# Import existing logic
from src.text_refiner.llm_router import PROVIDERS
from src.utils.env_manager import save_to_env
from src.audio.ear_runtime.controller import get_active_models

class ApiKeyModal(ModalScreen[Optional[str]]):
    """A modal dialog to enter an API key."""
    
    DEFAULT_CSS = """
    ApiKeyModal {
        align: center middle;
    }

    #dialog {
        grid-size: 1;
        grid-gutter: 1;
        grid-rows: 1fr 3;
        padding: 1 2;
        width: 60;
        height: 15;
        border: thick $accent;
        background: $surface;
    }

    #dialog Label {
        width: 100%;
        content-align: center middle;
    }

    #dialog Input {
        width: 100%;
    }
    """

    def __init__(self, provider_name: str, env_var: str, current_key: str = ""):
        super().__init__()
        self.provider_name = provider_name
        self.env_var = env_var
        self.current_key = current_key

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Enter API Key for [bold]{self.provider_name}[/bold]")
            yield Input(
                value=self.current_key, 
                placeholder=f"Paste {self.env_var} here...",
                password=True,
                id="key_input"
            )
            with Horizontal():
                yield Button("Save", variant="success", id="save")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.dismiss(self.query_one("#key_input").value)
        else:
            self.dismiss(None)

class WizardApp(App):
    """The main Bento Grid application for VibeVoice setup."""
    
    TITLE = "VibeVoice Setup Wizard"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    CSS = """
    Screen {
        background: $surface;
        align: center middle;
    }

    #bento-grid {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 1fr 1fr 1fr;
        grid-gutter: 1;
        width: 95%;
        height: 95%;
        min-width: 100;
        min-height: 28;
        padding: 1;
    }

    .bento-cell {
        border: round $primary;
        padding: 1;
        background: $panel;
    }

    /* Column 1: Providers */
    #provider-cell {
        row-span: 2;
    }

    .provider-btn {
        width: 100%;
        margin-bottom: 0;
    }

    /* Column 2: Hardware */
    #hardware-cell {
        row-span: 2;
    }

    /* Column 3: Software/Modes */
    #software-cell {
        row-span: 2;
    }

    #details-card {
        column-span: 2;
        background: $accent-darken-3;
    }

    #launch-card {
        column-span: 1;
    }

    #launch-btn {
        width: 100%;
        height: 100%;
    }

    .cell-title {
        text-style: bold;
        margin-bottom: 1;
        color: $accent;
    }

    Select {
        margin-bottom: 1;
    }

    .status-ready {
        color: $success;
        text-style: bold;
    }

    .status-missing {
        color: $error;
        text-style: bold;
    }

    Header {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    """

    def __init__(self):
        super().__init__()
        # 1. AI Provider State
        self.selected_provider_index = int(os.environ.get("VIBEVOICE_PROVIDER_INDEX", "0"))
        self.api_keys = {
            p["env_var"]: os.environ.get(p["env_var"], "") 
            for p in PROVIDERS
        }

        # 2. Hardware State (Microphones)
        self.pyaudio_instance = pyaudio.PyAudio()
        self.microphones = self._get_microphones()
        self.selected_mic_index = os.environ.get("VIBEVOICE_MIC_INDEX")
        
        valid_indices = [idx for name, idx in self.microphones]
        if self.selected_mic_index not in valid_indices:
            try:
                self.selected_mic_index = str(self.pyaudio_instance.get_default_input_device_info()["index"])
                if self.selected_mic_index not in valid_indices:
                    self.selected_mic_index = valid_indices[0] if valid_indices else "0"
            except:
                self.selected_mic_index = valid_indices[0] if valid_indices else "0"
                
        if not self.microphones:
            self.microphones.append(("No Microphone Found", "0"))

        # 3. Software/Mode State
        self.recording_mode = os.environ.get("RECORDING_MODE", "silence_streaming")
        self.stt_model = os.environ.get("STT_MODEL", "parakeet-tdt-0.6b-v3")
        self.telemetry_enabled = os.environ.get("STREAMING_TELEMETRY_ENABLED", "0") == "1"

    def _get_microphones(self) -> List[Tuple[str, str]]:
        """Fetch all input devices from PyAudio."""
        mics = []
        for i in range(self.pyaudio_instance.get_device_count()):
            info = self.pyaudio_instance.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                mics.append((info["name"], str(i)))
        return mics

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="bento-grid"):
            # CELL 1: AI Providers
            with Vertical(id="provider-cell", classes="bento-cell"):
                yield Label("1. AI REFINER", classes="cell-title")
                for i, p in enumerate(PROVIDERS):
                    variant = "primary" if i == self.selected_provider_index else "default"
                    yield Button(p["name"], id=f"prov_{i}", variant=variant, classes="provider-btn")
                
                yield Static("", id="key-status")
                yield Button("Update API Key", id="update-key-btn")

            # CELL 2: Hardware (Microphones)
            with Vertical(id="hardware-cell", classes="bento-cell"):
                yield Label("2. HARDWARE", classes="cell-title")
                yield Label("Input Microphone:")
                yield Select(self.microphones, value=self.selected_mic_index, id="mic-select")
                yield Static("\n[dim]The app will record from this device.[/dim]")

            # CELL 3: Software (Modes & Models)
            with Vertical(id="software-cell", classes="bento-cell"):
                yield Label("3. SOFTWARE", classes="cell-title")
                
                yield Label("Transcription Model:")
                stt_models = [(m, m) for m in get_active_models()]
                yield Select(stt_models, value=self.stt_model, id="model-select")

                yield Label("Recording Mode:")
                modes = [("Silence Streaming (Pro)", "silence_streaming"), ("No Streaming (Basic)", "no_streaming")]
                yield Select(modes, value=self.recording_mode, id="mode-select")

                with Horizontal():
                    yield Label("Telemetry: ")
                    yield Switch(value=self.telemetry_enabled, id="telemetry-switch")

            # CELL 4: Details (Bottom Left)
            with Vertical(id="details-card", classes="bento-cell"):
                yield Label("CONFIGURATION SUMMARY", classes="cell-title")
                yield Static("", id="details-text")

            # CELL 5: Launch (Bottom Right)
            with Vertical(id="launch-card", classes="bento-cell"):
                yield Button("LAUNCH VIBEVOICE", id="launch-btn", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.update_ui()

    def update_ui(self) -> None:
        """Refresh the UI based on current state."""
        provider = PROVIDERS[self.selected_provider_index]
        
        # 1. Update Provider Key Status
        has_key = bool(self.api_keys.get(provider["env_var"]))
        status_text = "[green]● Key Ready[/green]" if has_key else "[red]○ Key Missing[/red]"
        self.query_one("#key-status").update(status_text)

        for i in range(len(PROVIDERS)):
            btn = self.query_one(f"#prov_{i}", Button)
            btn.variant = "primary" if i == self.selected_provider_index else "default"

        # 2. Update Summary Details
        mic_name = "Unknown"
        for name, idx in self.microphones:
            if idx == self.selected_mic_index:
                mic_name = name
                break

        self.query_one("#details-text").update(
            f"AI Provider : [cyan]{provider['name']}[/cyan] ({provider['description']})\n"
            f"Microphone  : [yellow]{mic_name}[/yellow]\n"
            f"STT Model   : [green]{self.stt_model}[/green]\n"
            f"Mode        : [white]{self.recording_mode}[/white]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id.startswith("prov_"):
            self.selected_provider_index = int(button_id.split("_")[1])
            self.update_ui()
            
        elif button_id == "update-key-btn":
            provider = PROVIDERS[self.selected_provider_index]
            self.push_screen(
                ApiKeyModal(
                    provider["name"], 
                    provider["env_var"], 
                    self.api_keys.get(provider["env_var"], "")
                ),
                self.handle_key_save
            )
            
        elif button_id == "launch-btn":
            self.save_and_exit()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mic-select":
            self.selected_mic_index = str(event.value)
        elif event.select.id == "model-select":
            self.stt_model = str(event.value)
        elif event.select.id == "mode-select":
            self.recording_mode = str(event.value)
        self.update_ui()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "telemetry-switch":
            self.telemetry_enabled = event.value

    def handle_key_save(self, key: Optional[str]) -> None:
        if key is not None:
            provider = PROVIDERS[self.selected_provider_index]
            env_var = provider["env_var"]
            self.api_keys[env_var] = key
            save_to_env(env_var, key)
            self.notify(f"Key updated for {provider['name']}")
            self.update_ui()

    def on_unmount(self) -> None:
        """Cleanup resources when the app is closed."""
        self.pyaudio_instance.terminate()

    def save_and_exit(self) -> None:
        # Save all states to .env
        save_to_env("VIBEVOICE_PROVIDER_INDEX", str(self.selected_provider_index))
        save_to_env("VIBEVOICE_MIC_INDEX", self.selected_mic_index)
        save_to_env("RECORDING_MODE", self.recording_mode)
        save_to_env("STT_MODEL", self.stt_model)
        save_to_env("STREAMING_TELEMETRY_ENABLED", "1" if self.telemetry_enabled else "0")
        
        self.exit()

if __name__ == "__main__":
    app = WizardApp()
    app.run()
