# Refactor Inference Layer Implementation Plan (Phase 1: Construction Only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent** : 
- We are extracting the AI model logic from `src/backend/backend_parakeet.py` and `src/streaming/nemotron.py` into a unified, deep `src/engines/` package.
- **CRITICAL ARCHITECTURE RULE:** You are operating in a parallel worktree. **DO NOT MODIFY `src/backend/brain.py` OR DELETE OLD FILES.** Your job is ONLY to construct the new engine modules and their comprehensive test suites. The final wiring into `brain.py` will happen in a later merge phase.
- Currently, `brain.py` uses `hasattr(backend, "add_audio_chunk_and_get_text")` to guess if a model is stateful. We want to replace this with a clean `TranscriptionEngine` interface that all models must follow.
- **CRITICAL:** Do not assume anything, strictly follow the plan, and ask questions if you don't understand anything.
- Files to read to understand the current logic:
  - `src/backend/backend_parakeet.py` - [ Contains the stateless Sherpa-ONNX model loading and transcription logic. ]
  - `src/streaming/nemotron.py` - [ Contains the stateful Nemotron streaming engine logic. ]

**Architecture:**
- `src/engines/base.py`: Defines the `TranscriptionEngine` abstract base class. It forces models to declare if they are stateful and provides standard methods for transcribing.
- `src/engines/parakeet.py`: A `ParakeetEngine` class that implements the interface. It is stateless.
- `src/engines/nemotron.py`: A `NemotronEngine` class that implements the interface. It is stateful.

**Important Rule to follow :**
- **CRITICAL: ** add detailed docs in functions and explain the code and logic in comments.  
- **CRITICAL:** make the code function name and variable name clear and easily to understand instead of short and confusing names.
  - so a 5 year old child easily understands.
  - do not put any imagination and analogy to understand for 5 year old child.
  - write code function name and docs and code like this: **developer gets highest speed to read the code**.
  - **Explain like a fresher** 
  - **Write docs in your step-by-step simple style.**
  - **Make the docs in function and file headers human-readable and literal.**

---
## Task Structure

### Task 1 : Read out instruction file
- [ ] Read `/Users/apple/.gemini/GEMINI.md` file.
- **CRITICAL: ** add detailed docs in functions and explain the code and logic in comments.  
- **CRITICAL:** make the code function name and variable name clear.
- Avoid surface level (happy path) tests, use detailed tests covering boundaries and failure states.

### Task 2: The Base Engine Interface

**Files:**
- Create: `src/engines/__init__.py`
- Create: `src/engines/base.py`
- Create: `tests/test_engines_base.py`

- [ ] **Step 1: Write the failing tests (Interface Compliance)**
```python
# tests/test_engines_base.py
import pytest
import numpy as np
from src.engines.base import TranscriptionEngine

def test_engine_interface_requires_is_stateful_method():
    class BadEngine(TranscriptionEngine):
        pass
    
    with pytest.raises(TypeError) as exc_info:
        # Should fail because it doesn't implement abstract methods
        engine = BadEngine()
    
    error_message = str(exc_info.value)
    assert "Can't instantiate abstract class" in error_message
    assert "is_stateful" in error_message
    assert "transcribe_chunk" in error_message

def test_engine_interface_default_methods():
    class GoodEngine(TranscriptionEngine):
        def is_stateful(self) -> bool:
            return False
            
        def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
            return "test"
            
    engine = GoodEngine()
    
    # clear_internal_memory should be safe to call even if not overridden
    engine.clear_internal_memory()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_engines_base.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write implementation**
```python
# src/engines/__init__.py
"""
The Engines package.
Contains the 'rulebook' for all AI models and their specific implementations.
"""
```

```python
# src/engines/base.py
from abc import ABC, abstractmethod
import numpy as np

class TranscriptionEngine(ABC):
    """
    The Rulebook for all AI transcription models.
    Any model we add to the app MUST follow these rules.
    This stops the Brain from having to guess how each model works.
    """

    @abstractmethod
    def is_stateful(self) -> bool:
        """
        Returns True if the model remembers previous audio chunks (like Nemotron).
        Returns False if the model treats every chunk as a brand new recording (like Parakeet).
        """
        pass

    @abstractmethod
    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """
        Takes raw audio numbers (floats between -1.0 and 1.0) and turns them into text.
        If the model is stateful, this returns the FULL transcript so far.
        If the model is stateless, this returns ONLY the text for this specific chunk.
        """
        pass

    def clear_internal_memory(self) -> None:
        """
        Tells a stateful model to forget everything and start fresh.
        Stateless models can just ignore this.
        """
        pass
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_engines_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/engines/ tests/test_engines_base.py
git commit -m "feat: define base TranscriptionEngine interface"
```

### Task 3: The Parakeet Engine (Stateless)

**Files:**
- Create: `src/engines/parakeet.py`
- Create: `tests/test_engines_parakeet.py`

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_engines_parakeet.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.engines.parakeet import ParakeetEngine

def test_parakeet_engine_is_stateless():
    engine = ParakeetEngine(model_name="dummy_model")
    assert engine.is_stateful() is False

def test_parakeet_engine_transcribes_audio():
    with patch("src.backend.backend_parakeet.load_speech_recognition_model_from_disk") as mock_load, \\
         patch("src.backend.backend_parakeet.convert_audio_to_text") as mock_convert:
        
        mock_convert.return_value = "hello world"
        engine = ParakeetEngine(model_name="parakeet-tdt-0.6b-v3")
        
        # Give it some fake audio
        fake_audio = np.zeros(1024, dtype=np.float32)
        result = engine.transcribe_chunk(fake_audio)
        
        assert result == "hello world"
        mock_convert.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_engines_parakeet.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**
```python
# src/engines/parakeet.py
import numpy as np
from src.engines.base import TranscriptionEngine

# We import the old backend logic to do the heavy lifting,
# but we wrap it in our clean new class.
try:
    import src.backend.backend_parakeet as legacy_backend
except ImportError:
    legacy_backend = None

class ParakeetEngine(TranscriptionEngine):
    """
    The implementation for Parakeet, Conformer, and Moonshine models.
    These models are "stateless", meaning they don't remember the past.
    They transcribe whatever chunk of audio you give them, right now.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._loaded_model = None
        
        # Load the model from disk into memory when the class is created
        if legacy_backend:
            self._loaded_model = legacy_backend.load_speech_recognition_model_from_disk(self.model_name)

    def is_stateful(self) -> bool:
        """Parakeet models do not remember past audio chunks."""
        return False

    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """
        Passes the audio to the Sherpa-ONNX backend and returns the text string.
        """
        if not legacy_backend or not self._loaded_model:
            return ""
        
        text = legacy_backend.convert_audio_to_text(self._loaded_model, audio_samples)
        return text.strip()

    def clear_internal_memory(self) -> None:
        """
        Stateless models have no memory to clear.
        We do nothing, and that is perfectly fine.
        """
        pass
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_engines_parakeet.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/engines/parakeet.py tests/test_engines_parakeet.py
git commit -m "feat: implement ParakeetEngine conforming to base interface"
```

### Task 4: The Nemotron Engine (Stateful)

**Files:**
- Create: `src/engines/nemotron.py`
- Create: `tests/test_engines_nemotron.py`

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_engines_nemotron.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.engines.nemotron import NemotronEngine

def test_nemotron_engine_is_stateful():
    with patch("src.streaming.nemotron.NemotronStreamingEngine") as mock_legacy:
        engine = NemotronEngine()
        assert engine.is_stateful() is True

def test_nemotron_engine_transcribes_and_clears_memory():
    with patch("src.streaming.nemotron.NemotronStreamingEngine") as mock_legacy_class:
        mock_instance = MagicMock()
        mock_instance.add_audio_chunk_and_get_text.return_value = "hello cumulative world"
        mock_legacy_class.return_value = mock_instance
        
        engine = NemotronEngine()
        
        fake_audio = np.zeros(1024, dtype=np.float32)
        result = engine.transcribe_chunk(fake_audio)
        
        assert result == "hello cumulative world"
        mock_instance.add_audio_chunk_and_get_text.assert_called_once_with(fake_audio)
        
        engine.clear_internal_memory()
        mock_instance.clear_internal_memory.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_engines_nemotron.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**
```python
# src/engines/nemotron.py
import numpy as np
from src.engines.base import TranscriptionEngine

try:
    from src.streaming.nemotron import NemotronStreamingEngine as LegacyNemotron
except ImportError:
    LegacyNemotron = None

class NemotronEngine(TranscriptionEngine):
    """
    The implementation for the Nemotron streaming model.
    This model is "stateful", meaning it keeps an internal buffer of the audio.
    When you give it a new chunk, it returns the transcript for the ENTIRE recording so far.
    """
    def __init__(self):
        self._engine = LegacyNemotron() if LegacyNemotron else None

    def is_stateful(self) -> bool:
        """Nemotron absolutely remembers past chunks."""
        return True

    def transcribe_chunk(self, audio_samples: np.ndarray) -> str:
        """
        Adds the audio to Nemotron's internal buffer and gets the full, cumulative text back.
        """
        if not self._engine:
            return ""
        
        text = self._engine.add_audio_chunk_and_get_text(audio_samples)
        return text.strip()

    def clear_internal_memory(self) -> None:
        """
        Forces Nemotron to wipe its internal buffer.
        This must be called when the user finishes a recording session,
        so the next recording doesn't start with the old words.
        """
        if self._engine:
            self._engine.clear_internal_memory()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_engines_nemotron.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/engines/nemotron.py tests/test_engines_nemotron.py
git commit -m "feat: implement NemotronEngine conforming to base interface"
```
