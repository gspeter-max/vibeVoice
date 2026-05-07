# Textual Bento Wizard: Design & Implementation Context

## Context
We are upgrading the VibeVoice setup and provider selection experience to a "State of the Art" (2024/2025) Terminal User Interface (TUI). The current CLI prompts are functional but lack the premium feel required for a modern "Industry Standard" developer tool.

We have decided to build **Variant 2: The "Bento" Grid** using the **Textual** Python library.

## Visual Design (Bento Grid)
The UI will use a CSS-like grid layout with rounded borders and semantic colors to group information logically. It should feel like a high-end dashboard rather than a script.

```ascii
+-------------------------------------------------------+
|                 VIBEVOICE SETUP v2                    |
+---------------------------+---------------------------+
|                           |       SYSTEM STATUS       |
|      PRIMARY REFINER      |  [●] Engine: Parakeet     |
|         > GROQ <          |  [○] Mode:   Streaming    |
|                           |                           |
+-------------+-------------+-------------+-------------+
|  CEREBRAS   |  TOGETHER   |   API KEY   |   LATENCY   |
|  (Standby)  |  (Standby)  |  [ Valid ]  |   [ 0.1s ]  |
+-------------+-------------+-------------+-------------+
```

## Interaction Flow & Logic
1. **Interactive Selection:** Users can use their **Arrow Keys** or **Mouse** to click on the provider boxes (Groq, Cerebras, Together AI).
2. **Dynamic Key Checking:** When a provider is selected, the UI dynamically checks if the corresponding API key exists in the environment or `.env` file.
3. **Modal Input:** If the API key is missing, a Textual `Input` modal pops up asking the user to paste the key.
4. **Persistence:** The entered key is immediately saved to the `.env` file and loaded into `os.environ` (using `python-dotenv` and our `env_manager.py` logic) without freezing the UI.
5. **Reactivity:** The UI updates instantly to show the provider as "Ready" and sets it as the primary refiner in the LLM Router.

## Technical Requirements
- **Library:** `textual` (must be added to `pyproject.toml` dependencies).
- **Styling:** Use Textual CSS (TCSS) for the grid layout (`layout: grid;`, `grid-size`, `column-span`, `row-span`, `box.ROUNDED`).
- **State Management:** Use Textual's `reactive` attributes to sync the UI with the active provider state.
- **Integration:** The new Textual App will replace the `rich.prompt` logic currently at the start of `src/backend/brain.py`'s `start_server()` function.

## Next Steps for the AI Agent
1. Add `textual` to the project dependencies.
2. Create the Textual App class and its accompanying `.tcss` file.
3. Integrate the API key check/save logic into the Textual event loop.
4. Hook the Textual App into the `brain.py` startup sequence.
