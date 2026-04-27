# Streaming Session Extraction Implementation Plan (Phase 1: Construction Only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent** : 
- We are building a new dedicated module `src/streaming/session.py` to handle the stateful tracking of audio streaming.
- **CRITICAL ARCHITECTURE RULE:** You are operating in a parallel worktree. **DO NOT MODIFY `src/audio/ear.py` OR `src/backend/brain.py` OR `src/streaming/streaming_shared_logic.py`.** Your job is ONLY to create the new module and its tests.
- Currently, `streaming_shared_logic.py` provides pure functions (`apply_last_chunk_overlap`, `analyze_duplicate_chunk_prefix`). The `Ear` and `Brain` have to manually manage the state (the tail bytes, the transcript history) and pass them into these functions.
- We want to build a `StreamingSession` class that holds this state internally, giving the Ear and Brain a much deeper, easier-to-use interface.
- **CRITICAL:** Do not assume anything, strictly follow the plan, and ask questions if you don't understand anything.

**Architecture:**
- Create `src/streaming/session.py`.
- Define a `StreamingSession` class.
- It will import and use the pure functions from `src.streaming.streaming_shared_logic`.
- It will hold `_last_chunk_tail_bytes`, `_transcript_parts`, `_chunk_started_at`.
- It will expose simple stateful methods: `process_outgoing_audio_chunk(audio_bytes, stop_session, silence_elapsed)`, `process_incoming_text_chunk(seq_index, raw_text)`.

**Important Rule to follow :**
- **CRITICAL: ** add detailed docs in functions and explain the code and logic in comments.  
- **CRITICAL:** make the code function name and variable name clear and easily to understand instead of short and confusing names.
- write code function name and docs and code like this: **developer gets highest speed to read the code**
- **Explain like a fresher** 

---
## Task Structure

### Task 1 : Read out instruction file
- [ ] read `/Users/apple/.gemini/GEMINI.md` file

### Task 2: The Session Module

**Files:**
- Create: `src/streaming/session.py`
- Create: `tests/test_streaming_session.py`

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_streaming_session.py
import pytest
import time
from src.streaming.session import StreamingSession

def test_session_tracks_tail_bytes_for_audio_overlap():
    session = StreamingSession(overlap_seconds=0.1, sample_rate=16000)
    
    # Fake audio: 1 second of zeros
    fake_audio = b"\\x00\\x00" * 16000 
    
    # First chunk shouldn't have any overlap added at the start
    processed_audio_1 = session.process_outgoing_audio_chunk(fake_audio, stop_session=False, silence_seconds=0.0)
    assert len(processed_audio_1) == len(fake_audio)
    
    # Second chunk should have the tail of the first chunk added to the start
    processed_audio_2 = session.process_outgoing_audio_chunk(fake_audio, stop_session=False, silence_seconds=0.0)
    
    # 0.1 seconds * 16000 samples * 2 bytes/sample = 3200 bytes overlap
    expected_overlap_bytes = int(0.1 * 16000 * 2)
    assert len(processed_audio_2) == len(fake_audio) + expected_overlap_bytes

def test_session_deduplicates_incoming_text():
    session = StreamingSession()
    
    # Simulate Brain receiving chunks
    clean_1, stats_1 = session.process_incoming_text_chunk(0, "Hello world")
    assert clean_1 == "Hello world"
    
    # Chunk 2 overlaps with "world"
    clean_2, stats_2 = session.process_incoming_text_chunk(1, "world this is great")
    
    # The session should remember "Hello world" and trim "world" from chunk 2
    assert clean_2 == "this is great"
    assert session.get_full_transcript() == "Hello world this is great"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_streaming_session.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation for session.py**
```python
# src/streaming/session.py
import time
from typing import Tuple, Dict, Any

# We import the math functions, but we hide the complexity inside this class.
from src.streaming.streaming_shared_logic import (
    apply_last_chunk_overlap,
    analyze_duplicate_chunk_prefix,
    AnalysisResult
)

class StreamingSession:
    """
    Remembers the 'story' of the current recording.
    It keeps track of the audio overlap for the Ear, and the text history for the Brain.
    """
    def __init__(self, overlap_seconds: float = 1.0, sample_rate: int = 16000):
        self._overlap_seconds = overlap_seconds
        self._sample_rate = sample_rate
        self._overlap_byte_count = int(self._sample_rate * 2 * self._overlap_seconds)
        
        # State for the Ear (Audio)
        self._last_chunk_tail_bytes = b""
        self._chunk_started_at = time.time()
        
        # State for the Brain (Text)
        self._transcript_parts: Dict[int, str] = {}

    def process_outgoing_audio_chunk(self, audio_bytes: bytes, stop_session: bool, silence_seconds: float) -> bytes:
        """
        Called by the Ear before sending audio to the Brain.
        It takes the end of the previous chunk and glues it to the start of this chunk.
        This prevents words from being cut in half.
        """
        silence_byte_count = int(silence_seconds * self._sample_rate * 2)
        
        result = apply_last_chunk_overlap(
            current_chunk_audio_bytes=audio_bytes,
            last_chunk_tail_bytes=self._last_chunk_tail_bytes,
            overlap_audio_byte_count=self._overlap_byte_count,
            silence_audio_byte_count=silence_byte_count,
            sample_rate=self._sample_rate,
            stop_session=stop_session
        )
        
        self._last_chunk_tail_bytes = result.next_chunk_tail_bytes
        self._chunk_started_at = time.time()
        
        return result.overlapped_audio_bytes

    def process_incoming_text_chunk(self, sequence_index: int, raw_text: str) -> Tuple[str, AnalysisResult]:
        """
        Called by the Brain after transcribing a chunk.
        It looks at the previous chunk's text and removes any duplicated words from the start of the new text.
        """
        last_text = self._transcript_parts.get(sequence_index - 1, "")
        
        analysis = analyze_duplicate_chunk_prefix(last_text, raw_text)
        
        # Save the clean text so the next chunk can check against it
        self._transcript_parts[sequence_index] = analysis.cleaned_text
        
        return analysis.cleaned_text, analysis

    def get_full_transcript(self) -> str:
        """
        Combines all the clean text chunks into one final string.
        """
        parts = [part for seq, part in sorted(self._transcript_parts.items()) if part]
        return " ".join(parts).strip()
        
    def reset_audio_state(self):
        """Clears the audio overlap buffer for a brand new recording."""
        self._last_chunk_tail_bytes = b""
        self._chunk_started_at = time.time()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_streaming_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/streaming/session.py tests/test_streaming_session.py
git commit -m "feat: add stateful StreamingSession module"
```
