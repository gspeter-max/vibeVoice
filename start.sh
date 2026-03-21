#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Parakeet Flow v2 — start.sh                                ║
# ║                                                              ║
# ║  Backend toggle (set before running):                        ║
# ║    BACKEND=faster_whisper ./start.sh   ← default             ║
# ║    BACKEND=openvino       ./start.sh   ← Intel iGPU          ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

export BACKEND="${BACKEND:-faster_whisper}"
# Fix: ctranslate2 and residual torch/NeMo both bundle libiomp5.dylib — allow coexistence
export KMP_DUPLICATE_LIB_OK=TRUE
VENV_PYTHON="./.venv/bin/python"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        🎙️  PARAKEET FLOW  v2                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Backend  : $BACKEND"
echo "  Python   : $($VENV_PYTHON --version 2>&1)"
echo ""

# ── Sanity check ────────────────────────────────────────────────
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ .venv not found. Run:  uv venv && uv pip install -e ."
    exit 1
fi

# ── Kill any stale brain ─────────────────────────────────────────
if [ -f /tmp/parakeet-brain.pid ]; then
    OLD_PID=$(cat /tmp/parakeet-brain.pid)
    kill "$OLD_PID" 2>/dev/null && echo "  Stopped old Brain (PID $OLD_PID)" || true
    rm -f /tmp/parakeet-brain.pid
fi
rm -f /tmp/parakeet.sock

# ── Start Brain in background ────────────────────────────────────
echo "  Starting Brain..."
BACKEND="$BACKEND" "$VENV_PYTHON" brain.py > brain.log 2>&1 &
BRAIN_PID=$!
echo $BRAIN_PID > /tmp/parakeet-brain.pid
echo "  Brain PID: $BRAIN_PID  |  log: brain.log"

# ── Wait for socket (up to 120s for first-run model download) ───
echo ""
echo "  Waiting for Brain to be ready (first run downloads model)..."
WAIT=0
MAX_WAIT=120
while [ ! -S /tmp/parakeet.sock ]; do
    sleep 1
    WAIT=$((WAIT + 1))

    # Show progress dots
    printf "."

    # Check if brain died
    if ! kill -0 "$BRAIN_PID" 2>/dev/null; then
        echo ""
        echo "❌ Brain crashed on startup. Last log:"
        echo "─────────────────────────────────────"
        tail -30 brain.log
        echo "─────────────────────────────────────"
        exit 1
    fi

    if [ $WAIT -ge $MAX_WAIT ]; then
        echo ""
        echo "❌ Timed out waiting for Brain. Check brain.log"
        exit 1
    fi
done

echo ""
echo ""
echo "  ✅ Brain is Online!"
echo "══════════════════════════════════════════════════"
echo ""

# ── Start Ear (foreground — Ctrl+C to stop) ─────────────────────
BACKEND="$BACKEND" "$VENV_PYTHON" ear.py

# ── Cleanup when Ear exits ───────────────────────────────────────
echo ""
echo "  Stopping Brain (PID $BRAIN_PID)..."
kill "$BRAIN_PID" 2>/dev/null || true
rm -f /tmp/parakeet-brain.pid /tmp/parakeet.sock
echo "  Done. Goodbye."