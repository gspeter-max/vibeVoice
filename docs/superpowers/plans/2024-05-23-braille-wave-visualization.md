# Braille Wave Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent**:
- The goal is to replace the current block-based volume meter (`[███░░░]`) in `src/audio/ear.py` with a high-resolution scrolling **Braille Waveform**.
- Braille characters (`U+2800` to `U+28FF`) provide a 2x4 grid of dots, allowing for 4x higher vertical resolution and 2x higher horizontal resolution than standard blocks on a single terminal line.
- The visualization should feel "alive" and scrolling from right to left as the user speaks.
- The "Zen" terminal requirement must be preserved: the meter must remain on a single line and be properly cleared when transcription starts.

**Architecture:**
- A new utility class `BrailleWaveform` will be added to `src/audio/ear.py`.
- This class will maintain a fixed-size buffer of amplitude values.
- It will expose a `get_string()` method that converts the buffer into a string of Braille characters.
- The `ear.py` recording loop will be updated to feed RMS values into this class and print the resulting string.

**Files to read:**
- `src/audio/ear.py`: Main audio capture and visualization logic.

**Important Rule to follow:**
- **CRITICAL:** Add detailed docs in functions and explain the code and logic in comments.
- **CRITICAL:** Use clear, literal naming (e.g., `amplitude_history_buffer`, `map_amplitude_to_braille_dots`).
- **Explain like a fresher**: Write docs and code that a developer can read at "highest speed."

---

## Task 1: Research and Validation

- [ ] **Step 1: Verify Braille rendering in the current terminal**
  Run a simple command to ensure Braille characters render correctly.
  Run: `python3 -c "print('\u28FF\u287F\u283F\u281F\u280F\u2807\u2803\u2801')"`
  Expected: A series of Braille characters with decreasing dot counts.

- [ ] **Step 2: Read ear.py to locate the volume meter logic**
  Confirm the exact line for replacement.
  Run: `grep -n "Voice Level" src/audio/ear.py`

---

## Task 2: Implement BrailleWaveform Class

**Files:**
- Modify: `src/audio/ear.py` (Add the class near the top).

- [ ] **Step 1: Define the `BrailleWaveform` class**
  Implement the logic to map amplitudes to Braille dots.

```python
class BrailleWaveform:
    """
    Manages a scrolling Braille waveform for terminal visualization.
    Uses Unicode Braille characters (U+2800 - U+28FF) to provide high-resolution
    audio amplitude feedback on a single line.
    """
    def __init__(self, terminal_width: int = 40):
        # Each Braille character represents 2 time steps (2 columns of dots).
        # So for a width of 40 chars, we need 80 data points.
        self.max_data_points = terminal_width * 2
        self.amplitude_history = [0.0] * self.max_data_points
        
        # Braille dot mapping for vertical levels (0 to 4 dots high)
        # These are the bitmasks for the Braille dot patterns
        # Column 1 bits (left side of char): 1, 2, 3, 7
        # Column 2 bits (right side of char): 4, 5, 6, 8
        self.col1_dots = [0x00, 0x40, 0x40|0x04, 0x40|0x04|0x02, 0x40|0x04|0x02|0x01]
        self.col2_dots = [0x00, 0x80, 0x80|0x20, 0x80|0x20|0x10, 0x80|0x20|0x10|0x08]

    def add_amplitude(self, value: float):
        """Adds a new amplitude value (0.0 to 1.0) and scrolls the buffer."""
        # Clamp value to 0.0 - 1.0 range
        clamped_value = max(0.0, min(1.0, value))
        self.amplitude_history.append(clamped_value)
        if len(self.amplitude_history) > self.max_data_points:
            self.amplitude_history.pop(0)

    def get_braille_string(self) -> str:
        """Converts the amplitude history into a string of Braille characters."""
        braille_chars = []
        # Process data points in pairs (one Braille character per 2 points)
        for i in range(0, len(self.amplitude_history), 2):
            val1 = self.amplitude_history[i]
            val2 = self.amplitude_history[i+1] if i+1 < len(self.amplitude_history) else 0.0
            
            # Map 0.0-1.0 to 0-4 vertical dots
            level1 = int(val1 * 4)
            level2 = int(val2 * 4)
            
            # Combine the bits into a single Braille character code
            # Base Braille Unicode offset is 0x2800
            char_code = 0x2800 + self.col1_dots[level1] + self.col2_dots[level2]
            braille_chars.append(chr(char_code))
            
        return "".join(braille_chars)
```

- [ ] **Step 2: Add unit tests for `BrailleWaveform`**
  Create: `tests/audio/test_braille_wave.py`

```python
import pytest
from src.audio.ear import BrailleWaveform

def test_braille_wave_initialization():
    wave = BrailleWaveform(terminal_width=10)
    assert len(wave.amplitude_history) == 20
    assert all(v == 0.0 for v in wave.amplitude_history)

def test_braille_wave_scrolling():
    wave = BrailleWaveform(terminal_width=5)
    for i in range(10):
        wave.add_amplitude(i / 10.0)
    assert len(wave.amplitude_history) == 10
    assert wave.amplitude_history[-1] == 0.9

def test_braille_string_generation():
    wave = BrailleWaveform(terminal_width=5)
    # Fill with max values
    for _ in range(10):
        wave.add_amplitude(1.0)
    s = wave.get_braille_string()
    assert len(s) == 5
    # \u28FF is all 8 dots filled
    assert s == "\u28FF\u28FF\u28FF\u28FF\u28FF"
```

---

## Task 3: Integrate into ear.py

- [ ] **Step 1: Initialize the Braille wave in `Ear` class**
  Modify: `src/audio/ear.py` in `__init__`.
  
- [ ] **Step 2: Update the volume meter display loop**
  Modify: `src/audio/ear.py` around line 1155.
  Replace the block meter logic with the new Braille wave.

---

## Task 4: Final Validation and Polish

- [ ] **Step 1: Manual Test**
  Run `./start.sh` and speak.

- [ ] **Step 2: Cleanup and Commit**
  Remove any debug logs and commit.
