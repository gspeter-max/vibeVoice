# Hold-to-Record Integration Test Plan

**Date:** 2026-03-28
**Feature:** Hold-to-Record (push-to-talk mouse activation)
**Implementation:** Tasks 1-8 complete

---

## Test Environment Setup

1. **Start Brain Process:**
   ```bash
   python -m src.brain
   ```

2. **Start Ear Process (in new terminal):**
   ```bash
   python -m src.ear
   ```

3. **Verify HUD Running:**
   - Should see pill-shaped window at bottom of screen
   - Pill should be in idle state (small, outlined)

---

## Test Cases

### Test 1: Basic Hold-to-Record

**Steps:**
1. Press and hold right mouse button
2. Wait 1+ seconds
3. Continue holding for 2-3 seconds (simulating speaking)
4. Release mouse button

**Expected Results:**
- ✅ After ~1 second of holding: "RECORDING+STREAMING via MOUSE HOLD" message appears
- ✅ HUD expands to show animated bars (listening state)
- ✅ Volume meter displays in terminal
- ✅ On release: "Mouse released - STOPPING recording" message appears
- ✅ HUD shows processing state, then returns to idle

**Status:** _____ PASS / FAIL

**Notes:**
-


---

### Test 2: Early Release (< 1 second)

**Steps:**
1. Press and hold right mouse button
2. Release after 0.5 seconds (< 1 second threshold)

**Expected Results:**
- ✅ No recording starts
- ✅ No error messages
- ✅ Silent reset (nothing happens visibly)
- ✅ HUD remains in idle state

**Status:** _____ PASS / FAIL

**Notes:**
-


---

### Test 3: Right CMD Keyboard Shortcut Still Works

**Steps:**
1. Press Right CMD to start recording
2. Hold briefly, then release to stop
3. Optional: tap Right CMD quickly to confirm it latches until the next press

**Expected Results:**
- ✅ Recording starts on press
- ✅ "RECORDING+STREAMING" message appears
- ✅ HUD shows listening state
- ✅ Release after a short hold stops recording
- ✅ Very quick tap latches until the next press

**Status:** _____ PASS / FAIL

**Notes:**
-


---

### Test 4: Mixed Methods (No Conflicts)

**Steps:**
1. Press and hold Right CMD long enough to start recording
2. While still recording, press and hold right mouse button for 2+ seconds
3. Release mouse button
4. Release Right CMD to stop recording

**Expected Results:**
- ✅ Recording starts via Right CMD (normal)
- ✅ Holding mouse button is ignored (no duplicate recording)
- ✅ No error messages or crashes
- ✅ Recording continues normally
- ✅ Releasing Right CMD stops recording

**Status:** _____ PASS / FAIL

**Notes:**
-


---

### Test 5: Brain Not Ready Error Handling

**Steps:**
1. Stop brain process (Ctrl+C in brain terminal)
2. Press and hold right mouse button for 1+ seconds
3. Release mouse button

**Expected Results:**
- ✅ Error message: "❌ Failed to open brain stream"
- ✅ Hold flag resets (`_is_holding = False`)
- ✅ No crash or hang
- ✅ Can retry after restarting brain

**Status:** _____ PASS / FAIL

**Notes:**
-


---

### Test 6: Hold Duration Timing Accuracy

**Steps:**
1. Press and hold right mouse button
2. Count seconds aloud (use stopwatch if available)
3. Note when recording starts
4. Release after 3-4 seconds

**Expected Results:**
- ✅ Recording starts approximately 1.0 seconds after press (±0.1s tolerance)
- ✅ Timing feels natural, not too fast or too slow
- ✅ No noticeable delay beyond the 1-second threshold

**Status:** _____ PASS / FAIL

**Actual timing observed:** _____ seconds

**Notes:**
-


---

### Test 7: Rapid Press-Release Sequences

**Steps:**
1. Press and hold right mouse button for 0.3 seconds, release
2. Immediately press and hold for 0.3 seconds, release
3. Press and hold for 1.5 seconds, release (should trigger recording)
4. While recording, press and release mouse button rapidly

**Expected Results:**
- ✅ Early releases (0.3s) are silent, no errors
- ✅ Third hold (1.5s) triggers recording normally
- ✅ Rapid clicks during recording are ignored (no interference)
- ✅ No state corruption or crashes

**Status:** _____ PASS / FAIL

**Notes:**
-


---

## Summary

### Tests Passed: _____ / 7

### Issues Found:

1.
2.
3.

### Overall Assessment:

- [ ] Feature works as designed
- [ ] All requirements met
- [ ] Ready for production use

### Recommendations:

1.
2.
3.

---

## Tester Notes

Add any additional observations, edge cases discovered, or suggestions for improvement:

---
