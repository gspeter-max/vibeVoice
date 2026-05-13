# Remove Dead Dedup Wrapper Functions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **code-change** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---

## Goal Information for Fresh Agent

During a code audit session, two dead functions were identified in `streaming_shared_logic.py`:

1. **`remove_duplicate_chunk_prefix`** — a thin wrapper around `analyze_duplicate_chunk_prefix` that throws away the detailed result and returns only the cleaned string. Every caller only needs `.cleaned_text` from the full result, so the wrapper adds indirection with zero benefit.

2. **`combined_overlap_score`** — a standalone score calculator that was supposed to be reusable, but `analyze_duplicate_chunk_prefix` already computes the same score inline at its own loop body. Zero external callers exist.

Also found: `test_brain.py` has a duplicate test (`test_dedupe_with_last_chunk_removes_repeated_prefix`) that tests the same dedup logic already covered by `test_streaming_shared_logic.py`. It belongs in the wrong file.

**Decision made:** Remove both dead functions and their wrapper, migrate all callers to use `analyze_duplicate_chunk_prefix(...).cleaned_text` directly, remove the duplicate test, and clean up all orphaned imports.

**Files to read before starting:**
- `src/streaming/streaming_shared_logic.py` — contains both dead functions and `analyze_duplicate_chunk_prefix`
- `src/backend/brain.py` — has an orphaned import of `remove_duplicate_chunk_prefix` (never called at runtime)
- `evaluation/parakeet_v2_streaming_evaluation.py` — has 2 live call sites + 1 local wrapper function calling `remove_duplicate_chunk_prefix`
- `tests/test_streaming_shared_logic.py` — has 3 tests calling `remove_duplicate_chunk_prefix` that need to be migrated to `analyze_`
- `tests/test_brain.py` — has 1 duplicate test + 1 orphaned import to remove

---

## Architecture

```
BEFORE:
  evaluate/           -> remove_duplicate_chunk_prefix(last, curr)
  test_brain.py       -> remove_duplicate_chunk_prefix(last, curr)  [duplicate test]
  test_shared_logic   -> remove_duplicate_chunk_prefix(last, curr)
  brain.py            -> imports remove_duplicate_chunk_prefix  [never called]

  remove_duplicate_chunk_prefix()
    calls analyze_duplicate_chunk_prefix()
  combined_overlap_score()  <- zero callers anywhere

AFTER:
  evaluate/           -> analyze_duplicate_chunk_prefix(last, curr).cleaned_text
  test_shared_logic   -> analyze_duplicate_chunk_prefix(last, curr).cleaned_text
  brain.py            -> import removed
  test_brain.py       -> duplicate test deleted, import removed

  analyze_duplicate_chunk_prefix()  <- the single source of truth
```

---

## Important Rules to Follow

- **CRITICAL:** Add detailed docs in functions and explain the code logic in comments.
- **CRITICAL:** Make function names and variable names clear — a 5-year-old developer should be able to read and understand. No short, confusing names.
- Write docs in step-by-step simple style. Make them human-readable and literal.
- Avoid surface-level (happy path) tests. Tests must cover real edge cases.
- Do not assume anything. If something is unclear, ask the user.

---

## Task 1: Read Instruction Files

- [ ] Read `/Users/apple/.gemini/GEMINI.md`
- [ ] Read this plan file from top to bottom before writing any code

---

## Task 2: Remove `combined_overlap_score` from `streaming_shared_logic.py`

**Files:**
- Modify: `src/streaming/streaming_shared_logic.py` (L298-322)

This function computes a weighted average of `character_similarity` and `token_overlap_score`. It was intended as a reusable utility, but `analyze_duplicate_chunk_prefix` already computes the exact same thing inline at its own loop body. Zero callers exist outside this file.

- [ ] **Step 1: Verify zero callers**

Run:
```bash
grep -rn "combined_overlap_score" /Users/apple/project/vibeVoice --include="*.py"
```
Expected: Only the function definition line itself. No call sites.

- [ ] **Step 2: Delete the function**

Delete lines L298-322 in `src/streaming/streaming_shared_logic.py`:
```python
# DELETE THIS ENTIRE FUNCTION — zero callers, analyze_ does this inline
def combined_overlap_score(
    words_a: list[str],
    words_b: list[str],
    char_weight: float = 0.6,
    token_weight: float = 0.4,
) -> float:
    """..."""
    char_score = character_similarity(words_a, words_b)
    token_score = token_overlap_score(words_a, words_b)
    combined = (char_score * char_weight) + (token_score * token_weight)
    log.debug(...)
    return combined
```

- [ ] **Step 3: Run tests to make sure nothing broke**

```bash
cd /Users/apple/project/vibeVoice
pytest tests/test_streaming_shared_logic.py -v
```
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/streaming/streaming_shared_logic.py
git commit -m "refactor: remove dead combined_overlap_score — analyze_ computes it inline"
```

---

## Task 3: Migrate `evaluation/parakeet_v2_streaming_evaluation.py`

**Files:**
- Modify: `evaluation/parakeet_v2_streaming_evaluation.py`

**Change 1 — Update the import block (L9-20):**

Remove `remove_duplicate_chunk_prefix` from import. Add `analyze_duplicate_chunk_prefix`.

```python
# BEFORE
from src.streaming.streaming_shared_logic import (
    DEFAULT_ENERGY_RATIO,
    DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_SILENCE_TIMEOUT_SECONDS,
    DEFAULT_VAD_ENERGY_THRESHOLD,
    DEFAULT_VAD_SCORE_THRESHOLD,
    apply_last_chunk_overlap,
    normalize_text_for_word_error_rate,
    remove_duplicate_chunk_prefix,
    should_split_chunk_after_silence,
)

# AFTER
from src.streaming.streaming_shared_logic import (
    DEFAULT_ENERGY_RATIO,
    DEFAULT_MINIMUM_CHUNK_AGE_BEFORE_SILENCE_SPLIT_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_SILENCE_TIMEOUT_SECONDS,
    DEFAULT_VAD_ENERGY_THRESHOLD,
    DEFAULT_VAD_SCORE_THRESHOLD,
    analyze_duplicate_chunk_prefix,
    apply_last_chunk_overlap,
    normalize_text_for_word_error_rate,
    should_split_chunk_after_silence,
)
```

**Change 2 — Delete local wrapper function `remove_repeated_words_from_current_chunk_text` (L226-236):**

This local function only calls `remove_duplicate_chunk_prefix`. Since we are removing that function, this wrapper also becomes dead. Delete it entirely.

```python
# DELETE THIS ENTIRE FUNCTION
def remove_repeated_words_from_current_chunk_text(
    last_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 8,
) -> str:
    return remove_duplicate_chunk_prefix(
        last_chunk_text,
        current_chunk_text,
        max_overlap_words=max_overlap_words,
    )
```

**Change 3 — Update call site at L452:**

```python
# BEFORE
cleaned_chunk_text_after_dedup = remove_duplicate_chunk_prefix(
    last_chunk_text,
    raw_chunk_text_with_overlap,
    max_overlap_words=max_overlap_words,
)

# AFTER
cleaned_chunk_text_after_dedup = analyze_duplicate_chunk_prefix(
    last_chunk_text,
    raw_chunk_text_with_overlap,
    max_overlap_words=max_overlap_words,
).cleaned_text
```

**Change 4 — Update call site at L501:**

```python
# BEFORE
cleaned_final_chunk_text_after_dedup = remove_duplicate_chunk_prefix(
    last_chunk_text,
    raw_final_chunk_text_with_overlap,
    max_overlap_words=max_overlap_words,
)

# AFTER
cleaned_final_chunk_text_after_dedup = analyze_duplicate_chunk_prefix(
    last_chunk_text,
    raw_final_chunk_text_with_overlap,
    max_overlap_words=max_overlap_words,
).cleaned_text
```

- [ ] **Step 1: Apply all 4 changes above**

- [ ] **Step 2: Verify no remaining references**

```bash
grep -n "remove_duplicate_chunk_prefix\|remove_repeated_words" evaluation/parakeet_v2_streaming_evaluation.py
```
Expected: Zero results.

- [ ] **Step 3: Commit**

```bash
git add evaluation/parakeet_v2_streaming_evaluation.py
git commit -m "refactor: migrate evaluation script from remove_ wrapper to analyze_.cleaned_text"
```

---

## Task 4: Remove orphaned import from `brain.py`

**Files:**
- Modify: `src/backend/brain.py` (L22-25)

`brain.py` imports `remove_duplicate_chunk_prefix` but never calls it. The runtime uses `analyze_duplicate_chunk_prefix` directly. Just remove the import line.

```python
# BEFORE
from src.streaming.streaming_shared_logic import (
    analyze_duplicate_chunk_prefix,
    remove_duplicate_chunk_prefix,
)

# AFTER
from src.streaming.streaming_shared_logic import (
    analyze_duplicate_chunk_prefix,
)
```

- [ ] **Step 1: Apply the import change**

- [ ] **Step 2: Verify**

```bash
grep -n "remove_duplicate_chunk_prefix" src/backend/brain.py
```
Expected: Zero results.

- [ ] **Step 3: Run brain tests**

```bash
pytest tests/test_brain.py tests/test_brain_nemotron.py -v
```
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/backend/brain.py
git commit -m "refactor: remove orphaned import of remove_duplicate_chunk_prefix from brain.py"
```

---

## Task 5: Migrate tests in `test_streaming_shared_logic.py`

**Files:**
- Modify: `tests/test_streaming_shared_logic.py`

**Change 1 — Update import (L1-7):**

Remove `remove_duplicate_chunk_prefix`. `analyze_duplicate_chunk_prefix` is already imported.

```python
# BEFORE
from src.streaming.streaming_shared_logic import (
    apply_last_chunk_overlap,
    analyze_duplicate_chunk_prefix,
    normalize_text_for_word_error_rate,
    remove_duplicate_chunk_prefix,
    should_split_chunk_after_silence,
)

# AFTER
from src.streaming.streaming_shared_logic import (
    apply_last_chunk_overlap,
    analyze_duplicate_chunk_prefix,
    normalize_text_for_word_error_rate,
    should_split_chunk_after_silence,
)
```

**Change 2 — Migrate test at L53-58:**

```python
# BEFORE
def test_remove_duplicate_chunk_prefix_removes_exact_overlap_only():
    assert remove_duplicate_chunk_prefix(
        "things are happening fine",
        "things are happening fine and doing work",
        max_overlap_words=8,
    ) == "and doing work"

# AFTER
def test_analyze_duplicate_chunk_prefix_removes_exact_overlapping_words_from_start():
    """
    When chunk 2 starts with the same words that chunk 1 ended with,
    analyze_ should remove those repeated words and return only the new content.

    Example:
      chunk 1 ends with:   "things are happening fine"
      chunk 2 starts with: "things are happening fine and doing work"
      Result should be:    "and doing work"
    """
    result = analyze_duplicate_chunk_prefix(
        "things are happening fine",
        "things are happening fine and doing work",
        max_overlap_words=8,
    )
    assert result.cleaned_text == "and doing work"
    assert result.trim_applied is True
```

**Change 3 — Migrate test at L61-66:**

```python
# BEFORE
def test_remove_duplicate_chunk_prefix_ignores_case_and_edge_punctuation_for_matching():
    assert remove_duplicate_chunk_prefix(
        "that I made.",
        "That I made a few months ago while writing an article for Italian Wired.",
        max_overlap_words=8,
    ) == "a few months ago while writing an article for Italian Wired."

# AFTER
def test_analyze_duplicate_chunk_prefix_ignores_letter_case_and_punctuation_when_matching():
    """
    The matching should be case-insensitive and should ignore punctuation at word edges.
    "that I made." and "That I made" should be treated as the same words.

    Example:
      chunk 1 ends with:   "that I made."
      chunk 2 starts with: "That I made a few months ago..."
      Result should be:    "a few months ago while writing an article for Italian Wired."
    """
    result = analyze_duplicate_chunk_prefix(
        "that I made.",
        "That I made a few months ago while writing an article for Italian Wired.",
        max_overlap_words=8,
    )
    assert result.cleaned_text == "a few months ago while writing an article for Italian Wired."
    assert result.trim_applied is True
```

**Change 4 — Migrate test at L69-74:**

```python
# BEFORE
def test_remove_duplicate_chunk_prefix_keeps_text_when_overlap_trim_would_be_too_small():
    assert remove_duplicate_chunk_prefix(
        "once in my",
        "once in my life.",
        max_overlap_words=8,
    ) == "once in my life."

# AFTER
def test_analyze_duplicate_chunk_prefix_keeps_original_text_when_trim_would_leave_almost_nothing():
    """
    Safety check: if removing the overlapping words would leave the new chunk with
    1 or fewer words, we skip the trim to avoid losing real content.

    Example:
      chunk 1 ends with:   "once in my"
      chunk 2 starts with: "once in my life."
      Removing "once in my" would leave only "life." — just 1 word.
      So the full text "once in my life." should be kept unchanged.
    """
    result = analyze_duplicate_chunk_prefix(
        "once in my",
        "once in my life.",
        max_overlap_words=8,
    )
    assert result.cleaned_text == "once in my life."
    assert result.skipped_because_result_too_small is True
    assert result.trim_applied is False
```

- [ ] **Step 1: Apply all 4 changes**

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_streaming_shared_logic.py -v
```
Expected: All tests pass.

- [ ] **Step 3: Verify no remaining references**

```bash
grep -n "remove_duplicate_chunk_prefix" tests/test_streaming_shared_logic.py
```
Expected: Zero results.

- [ ] **Step 4: Commit**

```bash
git add tests/test_streaming_shared_logic.py
git commit -m "refactor: migrate test_streaming_shared_logic from remove_ wrapper to analyze_ directly"
```

---

## Task 6: Remove duplicate test and import from `test_brain.py`

**Files:**
- Modify: `tests/test_brain.py`

**Change 1 — Remove import at L25:**

```python
# DELETE THIS LINE
from src.streaming.streaming_shared_logic import remove_duplicate_chunk_prefix
```

**Change 2 — Delete duplicate test function at L158-168:**

This test tests `remove_duplicate_chunk_prefix` as a standalone function. That same logic is already covered by `test_streaming_shared_logic.py`. Brain tests should only test brain behavior, not shared logic utilities.

```python
# DELETE THIS ENTIRE FUNCTION
def test_dedupe_with_last_chunk_removes_repeated_prefix():
    """
    Direct unit test for the deduplication helper function.
    Verifies that overlapping words from the previous chunk are trimmed off.
    """
    cleaned = remove_duplicate_chunk_prefix(
        "I want to see that things are happening fine",
        "things are happening fine and doing H3 grid",
    )
    assert cleaned == "doing H3 grid"
```

- [ ] **Step 1: Apply both changes**

- [ ] **Step 2: Run brain tests**

```bash
pytest tests/test_brain.py -v
```
Expected: All remaining tests pass. The deleted test is gone.

- [ ] **Step 3: Verify no remaining references**

```bash
grep -n "remove_duplicate_chunk_prefix" tests/test_brain.py
```
Expected: Zero results.

- [ ] **Step 4: Commit**

```bash
git add tests/test_brain.py
git commit -m "refactor: remove duplicate dedup test and orphaned import from test_brain.py"
```

---

## Task 7: Delete `remove_duplicate_chunk_prefix` from `streaming_shared_logic.py`

**IMPORTANT:** Only do this task AFTER Tasks 3–6 are fully complete and committed.
All callers must be migrated before deleting the function.

**Files:**
- Modify: `src/streaming/streaming_shared_logic.py` (L392-427)

- [ ] **Step 1: Verify zero callers remain**

```bash
grep -rn "remove_duplicate_chunk_prefix" /Users/apple/project/vibeVoice --include="*.py"
```
Expected: **Zero results.** If any results appear, stop and fix them first before continuing.

- [ ] **Step 2: Delete the entire function**

```python
# DELETE THIS ENTIRE FUNCTION
def remove_duplicate_chunk_prefix(
    last_chunk_text: str,
    current_chunk_text: str,
    *,
    max_overlap_words: int = 15,
) -> str:
    """..."""
    analysis = analyze_duplicate_chunk_prefix(
        last_chunk_text,
        current_chunk_text,
        max_overlap_words=max_overlap_words,
    )
    if analysis.trim_applied:
        log.debug(...)
    return analysis.cleaned_text
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```
Expected: All tests pass. Zero failures.

- [ ] **Step 4: Final dead code confirmation**

```bash
grep -rn "remove_duplicate_chunk_prefix\|combined_overlap_score" /Users/apple/project/vibeVoice --include="*.py"
```
Expected: Zero results.

- [ ] **Step 5: Commit**

```bash
git add src/streaming/streaming_shared_logic.py
git commit -m "refactor: delete remove_duplicate_chunk_prefix — all callers migrated to analyze_.cleaned_text"
```

---

## Verification Plan

### Final full test run

```bash
cd /Users/apple/project/vibeVoice
pytest tests/ -v
```
Expected: All tests pass. Zero failures.

### Final dead code check

```bash
grep -rn "remove_duplicate_chunk_prefix\|combined_overlap_score" /Users/apple/project/vibeVoice --include="*.py"
```
Expected: Zero results.

### Import sanity check

```bash
python -c "from src.streaming.streaming_shared_logic import analyze_duplicate_chunk_prefix; print('OK')"
python -c "import src.backend.brain; print('OK')"
```
Expected: Both print `OK` with no import errors.
