# Textual Bento Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent**:
The goal is to upgrade the VibeVoice setup experience from a simple CLI prompt to a "State of the Art" Bento Grid TUI using the `Textual` library. 
The current setup logic is in `src/utils/wizard.py`, which is called by `start.sh` in the foreground. After the wizard exits, `start.sh` loads the `.env` file and starts the background `brain.py` process.

**Decision made:** We will replace the content of `src/utils/wizard.py` with a Textual App. We will NOT put the TUI inside `brain.py` because `brain.py` runs in the background, which would conflict with a terminal UI.

**Files to understand/read:**
- `src/utils/wizard.py`: Current Rich-based setup wizard.
- `src/utils/env_manager.py`: Handles `.env` reading and writing.
- `src/text_refiner/llm_router.py`: Contains the `PROVIDERS` list.
- `start.sh`: The orchestrator that runs the wizard and then the brain.
- `pyproject.toml`: Project dependencies.

**Architecture:**
- **Textual App**: A Python class inheriting from `textual.app.App`.
- **Bento Grid Layout**: Defined in a `.tcss` file using `layout: grid`.
- **Reactive State**: The App will use `reactive` attributes to track the `selected_provider_index`.
- **API Key Detection**: On provider selection, the app checks if the key exists using `env_manager.py`.
- **Modal Dialog**: A `ModalScreen` for entering missing API keys.
- **Idempotence**: Changes are saved to `.env` immediately upon selection or input.

**Important Rule to follow :**
- **CRITICAL:** add detailed docs in functions and explain the code and logic in comments.
- **CRITICAL:** make the code function name and variable name clear and easily to understand instead of short and confusing names.
    - Write code so that a 5-year-old developer should be able to read and understand.
    - No imagination or analogy — just clear, literal naming.
    - Write code function names, docs, and implementation so a developer gets the highest reading speed.
    - **Explain like a fresher**: Use simple, step-by-step documentation.
    - **Make the docs human-readable and literal**.

---

### Task 1: Environment Setup
- Read `GEMINI.md` and project instructions.
- Add `textual` dependency.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add textual to dependencies**
Add `textual` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 2: Sync dependencies**
Run: `uv pip install textual` or `pip install textual` depending on the environment.

---

### Task 2: Design the Bento Styles
- Create the CSS file for the TUI layout.

**Files:**
- Create: `src/utils/wizard.tcss`

- [ ] **Step 1: Write the TCSS layout**
Define a grid with 2 columns and 2 rows. Use rounded borders and semantic colors.

```css
/* src/utils/wizard.tcss */
Screen {
    layout: grid;
    grid-size: 2 2;
    grid-gutter: 1;
    padding: 1;
}

#header {
    column-span: 2;
    height: 3;
    content-align: center middle;
    background: $accent;
    color: white;
    text-style: bold;
}

.bento-box {
    border: round $primary;
    padding: 1;
    height: 100%;
}

.selected {
    border: double $success;
    background: $success 10%;
}

#provider-list {
    layout: vertical;
}

#status-card {
    background: $surface;
}

#action-area {
    content-align: right bottom;
}
```

---

### Task 3: Implement the Bento Wizard Logic
- Replace the existing `wizard.py` with the Textual implementation.

**Files:**
- Modify: `src/utils/wizard.py`

- [ ] **Step 1: Scaffold the Textual App**
Import `textual` components and setup the basic App structure.

- [ ] **Step 2: Implement Reactive State**
Add a `selected_index = reactive(0)` and a method to update it when a provider is clicked.

- [ ] **Step 3: Implement API Key Check**
On index change, check if the key exists in `.env`.

- [ ] **Step 4: Implement Modal Screen for API Key**
Create a `ModalScreen` that takes an input string and saves it via `env_manager.save_to_env`.

- [ ] **Step 5: Implement the "Launch" button**
The button should call `self.exit()` to allow `start.sh` to proceed.

---

### Task 4: Testing & Validation
- Ensure the state logic works without running the full TUI if possible, or verify manual flow.

**Files:**
- Create: `tests/test_wizard_logic.py`

- [ ] **Step 1: Write a test for provider selection logic**
Mock the `env_manager` and verify that selecting a provider correctly identifies if a key is missing.

- [ ] **Step 2: Run tests**
Run: `pytest tests/test_wizard_logic.py`

- [ ] **Step 3: Manual Verification**
Run: `python src/utils/wizard.py` and verify the UI looks like a Bento Grid and responds to clicks.

---

### Task 5: Integration Check
- Verify that `start.sh` correctly picks up the changes.

- [ ] **Step 1: Run start.sh**
Run: `./start.sh`
Expected: The new Bento UI appears, you select a provider, and then the Brain starts normally.
