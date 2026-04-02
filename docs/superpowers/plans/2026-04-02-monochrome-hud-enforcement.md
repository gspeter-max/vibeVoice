# Monochrome HUD Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the live HUD always renders a premium white waveform and cannot accidentally continue showing stale colorful bars from an older process or ambiguous runtime state.

**Architecture:** Keep the waveform renderer monochrome at the theme layer, add explicit runtime signatures in the HUD so the active process can be identified from logs, and harden launcher cleanup so stale GUI processes are terminated before a fresh HUD instance is started. Verification should combine unit assertions for monochrome color output with focused runtime/launcher checks.

**Tech Stack:** Python 3.11, PySide6, pytest, shell startup script

---

## File Structure

- Modify: `src/theme_manager.py`
  Purpose: Own the authoritative monochrome bar-color policy.
- Modify: `src/hud.py`
  Purpose: Expose a runtime monochrome signature, centralize bar color mode metadata, and log the active renderer configuration at startup.
- Modify: `start.sh`
  Purpose: Kill stale HUD instances more aggressively and surface the monochrome mode in launcher output.
- Modify: `tests/test_theme_manager.py`
  Purpose: Lock the monochrome bar policy and runtime configuration with regression tests.

---

### Task 1: Lock Runtime Monochrome Configuration in Tests

**Files:**
- Modify: `tests/test_theme_manager.py`
- Test: `tests/test_theme_manager.py`

- [ ] **Step 1: Add failing tests for runtime monochrome configuration**

```python
def test_hud_declares_monochrome_bar_mode():
    assert hud.BAR_COLOR_MODE == "monochrome"


def test_hud_runtime_signature_mentions_monochrome():
    signature = hud.runtime_signature()
    assert "monochrome" in signature
    assert "bars=25" in signature
```

- [ ] **Step 2: Run the test file to verify the new assertions fail**

Run: `pytest tests/test_theme_manager.py -q`
Expected: FAIL because `hud.BAR_COLOR_MODE` and `hud.runtime_signature()` do not exist yet.

- [ ] **Step 3: Implement the minimal runtime metadata in the HUD**

```python
BAR_COLOR_MODE = "monochrome"


def runtime_signature() -> str:
    return (
        f"mode={BAR_COLOR_MODE} bars={NUM_BARS} "
        f"bar_w={BAR_W:.1f} gap={BAR_GAP:.1f} active_w={PILL_W_ACTIVE}"
    )
```

- [ ] **Step 4: Run the test file to verify it passes**

Run: `pytest tests/test_theme_manager.py -q`
Expected: PASS

---

### Task 2: Prove the Live HUD Renderer at Startup

**Files:**
- Modify: `src/hud.py`
- Test: `tests/test_theme_manager.py`

- [ ] **Step 1: Add a failing test for explicit startup-visible monochrome proof**

```python
def test_theme_manager_returns_exact_white_rgb():
    tm = ThemeManager(THEME_ORIGINAL)
    color = tm.get_bar_color(
        bar_index=12,
        total_bars=25,
        voice_intensity=0.7,
        bar_height_factor=0.9,
        frequency_bands={"bass": 0.8, "mid": 0.1, "treble": 0.1},
    )
    assert (color.red(), color.green(), color.blue()) == (255, 255, 255)
```

- [ ] **Step 2: Run the focused test to confirm the expected baseline**

Run: `pytest tests/test_theme_manager.py::test_theme_manager_returns_exact_white_rgb -q`
Expected: PASS or FAIL only if monochrome output regressed.

- [ ] **Step 3: Log the monochrome startup signature from the HUD**

```python
print(f"[HUD] Theme mode: {runtime_signature()}", flush=True)
preview = self._theme_manager.get_bar_color(
    bar_index=NUM_BARS // 2,
    total_bars=NUM_BARS,
    voice_intensity=1.0,
    bar_height_factor=1.0,
    frequency_bands=self._frequency_bands,
)
print(
    f"[HUD] Theme preview RGB=({preview.red()},{preview.green()},{preview.blue()}) "
    f"alpha={preview.alpha()}",
    flush=True,
)
```

- [ ] **Step 4: Re-run the full theme test file**

Run: `pytest tests/test_theme_manager.py -q`
Expected: PASS

---

### Task 3: Remove Stale HUD Process Risk from the Launcher

**Files:**
- Modify: `start.sh`

- [ ] **Step 1: Add launcher cleanup helpers**

```bash
kill_pid_file_process() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        kill "$(cat "$pid_file")" 2>/dev/null || true
        kill -9 "$(cat "$pid_file")" 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

kill_hud_processes() {
    lsof -ti :57234 | xargs kill -9 2>/dev/null || true
    pkill -f "src/hud.py" 2>/dev/null || true
}
```

- [ ] **Step 2: Replace the duplicated inline cleanup logic with the helpers**

```bash
cleanup() {
    echo ""
    echo "  Cleaning up..."
    kill_pid_file_process /tmp/parakeet-brain.pid
    kill_pid_file_process /tmp/parakeet-hud.pid
    kill_hud_processes
    rm -f /tmp/parakeet.sock
    echo "  Done. Goodbye."
}
```

- [ ] **Step 3: Use the same HUD cleanup before starting a fresh instance**

```bash
echo "  Cleaning up old processes..."
kill_pid_file_process /tmp/parakeet-brain.pid
kill_pid_file_process /tmp/parakeet-hud.pid
kill_hud_processes
rm -f /tmp/parakeet.sock /tmp/parakeet-brain.pid /tmp/parakeet-hud.pid
```

- [ ] **Step 4: Keep the launcher theme banner aligned with monochrome behavior**

Run: `rg -n "Theme    :" start.sh`
Expected: one line showing `Dark with premium white waveform bars`

---

### Task 4: Focused Verification

**Files:**
- Test: `tests/test_theme_manager.py`
- Test: `tests/test_ear_hold_state.py`
- Test: `tests/test_brain.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Run theme and HUD regression tests**

Run: `pytest tests/test_theme_manager.py -q`
Expected: PASS

- [ ] **Step 2: Run the ear hold-state regression tests**

Run: `pytest tests/test_ear_hold_state.py -q`
Expected: PASS

- [ ] **Step 3: Run the lightweight brain/integration tests**

Run: `pytest tests/test_brain.py tests/test_integration.py -q`
Expected: PASS

- [ ] **Step 4: Inspect the HUD log after a fresh launcher run**

Run: `tail -n 40 logs/hud.log`
Expected: startup lines include `Theme mode: mode=monochrome` and `Theme preview RGB=(255,255,255)`

