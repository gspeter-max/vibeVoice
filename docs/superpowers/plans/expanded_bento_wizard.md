# Implementation Plan: Expanded Bento Wizard (Hardware & Modes)

Enhance the Textual Bento Wizard to include hardware microphone selection, recording mode toggles, and STT model selection.

## Objectives
- [x] Mic Selection: List and select hardware input devices.
- [x] Recording Mode: Toggle between `silence_streaming` and `no_streaming`.
- [x] STT Model: Select from available models (Parakeet, Moonshine, etc.).
- [x] Unified .env saving for all new parameters.

## Tasks

### 1. Hardware Microphone Enumeration
- [ ] Add `pyaudio` logic to `WizardApp` to list available input devices.
- [ ] Implement a `Select` widget or `ListView` for microphones in the Bento Grid.

### 2. Recording Mode & STT Model Selection
- [ ] Add a `RadioSet` or `Select` widget for Recording Mode.
- [ ] Add a `Select` widget for STT Model (fetching list from `src.audio.ear.get_active_models`).

### 3. TUI Layout Update (`src/utils/wizard_tui.py`)
- [ ] Redesign Bento Grid to 3 columns x 3 rows to fit:
    - Col 1: Providers (Groq/Cerebras)
    - Col 2: Hardware (Mics)
    - Col 3: Software (STT Model, Recording Mode, Telemetry)
    - Bottom: Launch Button

### 4. Persistence & Validation
- [ ] Update `save_and_exit` to persist:
    - `VIBEVOICE_MIC_INDEX`
    - `RECORDING_MODE`
    - `STT_MODEL` (to be used by brain.py)
- [ ] Verify `ear.py` and `brain.py` correctly pick up these new .env variables.

## Rules
- No live audio meters in the UI (per user request).
- Use Textual standard widgets for clean interaction.
- Ensure backwards compatibility (default to system defaults if .env is empty).
