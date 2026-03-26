#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Parakeet Flow v2 — start.sh                                 ║
# ║                                                              ║
# ║  Backend toggle (set before running):                        ║
# ║    BACKEND=faster_whisper ./start.sh   ← default             ║
# ║    BACKEND=openvino       ./start.sh   ← Intel iGPU          ║
# ║  Thread count for Parakeet (for testing/benchmarking):       ║
# ║    PARAKEET_THREADS=2 ./start.sh   ← Use 2 threads           ║
# ║    PARAKEET_THREADS=4 ./start.sh   ← Use 4 threads           ║
# ║    PARAKEET_THREADS=6 ./start.sh   ← Use 6 threads           ║
# ║    PARAKEET_THREADS=12 ./start.sh  ← Use 12 threads          ║
# ║                                                              ║
# ║  Voice Isolation toggle (for macOS):                         ║
# ║    VOICE_ISOLATION=1 ./start.sh   ← enable (default off)     ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

cleanup() {
    echo ""
    echo "  Cleaning up..."
    [ -f /tmp/parakeet-brain.pid ] && kill "$(cat /tmp/parakeet-brain.pid)" 2>/dev/null; kill -9 "$(cat /tmp/parakeet-brain.pid)" 2>/dev/null; rm -f /tmp/parakeet-brain.pid || true
    [ -f /tmp/parakeet-hud.pid ] && kill "$(cat /tmp/parakeet-hud.pid)" 2>/dev/null; kill -9 "$(cat /tmp/parakeet-hud.pid)" 2>/dev/null; rm -f /tmp/parakeet-hud.pid || true
    rm -f /tmp/parakeet.sock
    echo "  Done. Goodbye."
}
trap cleanup EXIT INT TERM

export BACKEND="${BACKEND:-faster_whisper}"
export VOICE_ISOLATION="${VOICE_ISOLATION:-0}"
# Fix: ctranslate2 and residual torch/NeMo both bundle libiomp5.dylib — allow coexistence
export KMP_DUPLICATE_LIB_OK=TRUE
# Fix: Qt/PySide6 windows won't render on Intel Mac (Sonoma+) without this
export QT_MAC_WANTS_LAYER=1
export PARAKEET_THREADS="${PARAKEET_THREADS:-}"
VENV_PYTHON="./.venv/bin/python"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        🎙️  PARAKEET FLOW  v2                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Backend  : $BACKEND"
echo "  Threads  : ${PARAKEET_THREADS:-auto (all cores)}"
echo "  Python   : $($VENV_PYTHON --version 2>&1)"
echo ""

# ── Sanity check ────────────────────────────────────────────────
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ .venv not found. Run:  uv venv && uv pip install -e ."
    exit 1
fi

# ── Kill any stale processes ─────────────────────────────────────
echo "  Cleaning up old processes..."
kill $(cat /tmp/parakeet-brain.pid 2>/dev/null) 2>/dev/null || true
kill $(cat /tmp/parakeet-hud.pid 2>/dev/null) 2>/dev/null || true
rm -f /tmp/parakeet.sock /tmp/parakeet-brain.pid /tmp/parakeet-hud.pid

# ── Ensure logs directory exists ─────────────────────────────────
mkdir -p logs

# ── Start Brain in background ────────────────────────────────────
echo "  Starting Brain..."
BACKEND="$BACKEND" "$VENV_PYTHON" src/brain.py > logs/brain.log 2>&1 &
BRAIN_PID=$!
echo $BRAIN_PID > /tmp/parakeet-brain.pid
echo "  Brain PID: $BRAIN_PID  |  log: logs/brain.log"

# ── Wait for socket (up to 120s for first-run model download) ───
echo ""
echo "  Waiting for Brain to be ready (first run downloads model)..."
WAIT=0
MAX_WAIT=120
while [ ! -S /tmp/parakeet.sock ]; do
    sleep 1
    WAIT=$((WAIT + 1))

    printf "."

    if ! kill -0 "$BRAIN_PID" 2>/dev/null; then
        echo ""
        echo "❌ Brain crashed on startup."
        if [ -f "logs/brain.log" ]; then
            echo "Last log:"
            echo "─────────────────────────────────────"
            tail -30 logs/brain.log
            echo "─────────────────────────────────────"
        else
            echo "No log file found. Possible issues:"
            echo "  - logs/ directory doesn't exist"
            echo "  - Brain crashed before logging could start"
            echo "  - Python import error (missing dependencies)"
            echo ""
            echo "Try running: src/brain.py manually to see the error"
        fi
        exit 1
    fi

    if [ $WAIT -ge $MAX_WAIT ]; then
        echo ""
        echo "❌ Timed out waiting for Brain. Check logs/brain.log"
        exit 1
    fi
done

echo ""
echo ""
echo "  ✅ Brain is Online!"
echo "══════════════════════════════════════════════════"
echo ""

# ── Start HUD in background ──────────────────────────────────────
# CRITICAL: must be launched directly from the shell here, NOT as a
# subprocess of ear.py.  When Qt/PySide6 is spawned too deep in a
# subprocess chain, macOS WindowServer rejects the GUI memory allocation
# with SIGABRT (vm_map_enter failure).  Shell-level launch fixes this.
echo "  Starting HUD..."
# Kill anything stale still holding the IPC port
lsof -ti :57234 | xargs kill -9 2>/dev/null || true
"$VENV_PYTHON" src/hud.py > logs/hud.log 2>&1 &
HUD_PID=$!
echo $HUD_PID > /tmp/parakeet-hud.pid
echo "  HUD   PID: $HUD_PID  |  log: logs/hud.log"
sleep 0.8   # give Qt/Cocoa time to connect to WindowServer

# ── Start Ear (foreground — Ctrl+C to stop) ─────────────────────
BACKEND="$BACKEND" "$VENV_PYTHON" src/ear.py

