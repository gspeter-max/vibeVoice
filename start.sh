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
# ║  Brain logs always print in this terminal.                   ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

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

cleanup() {
    echo ""
    echo "  Cleaning up..."
    kill_pid_file_process /tmp/parakeet-brain.pid
    kill_pid_file_process /tmp/parakeet-hud.pid
    kill_hud_processes
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
echo "  Theme    : Dark with premium white waveform bars"
echo "  Logs     : live terminal output"
echo ""

# ── Sanity check ────────────────────────────────────────────────
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ .venv not found. Run:  uv venv && uv pip install -e ."
    exit 1
fi

# ── Kill any stale processes ─────────────────────────────────────
echo "  Cleaning up old processes..."
kill_pid_file_process /tmp/parakeet-brain.pid
kill_pid_file_process /tmp/parakeet-hud.pid
kill_hud_processes
rm -f /tmp/parakeet.sock /tmp/parakeet-brain.pid /tmp/parakeet-hud.pid

# ── Ensure logs directory exists ─────────────────────────────────
mkdir -p logs

# ── Start Brain in background ────────────────────────────────────
echo "  Starting Brain..."
BACKEND="$BACKEND" "$VENV_PYTHON" src/brain.py &
BRAIN_PID=$!
echo $BRAIN_PID > /tmp/parakeet-brain.pid
echo "  Brain PID: $BRAIN_PID  |  live terminal output"

# ── Wait for socket (up to 120s for first-run model download) ───
echo ""
echo "  Waiting for Brain to be ready..."
if [ ! -d ~/.cache/parakeet-flow/models/deepdml ]; then
    echo "  ⚠️  First run: Downloading model (~1.5 GB)..."
    echo "  This may take 5-10 minutes depending on your internet speed."
    echo "  Subsequent starts will be much faster!"
    echo ""
fi
WAIT=0
MAX_WAIT=300  # Increased to 5 minutes for first download
while [ ! -S /tmp/parakeet.sock ]; do
    sleep 1
    WAIT=$((WAIT + 1))

    # Show progress indicator with time
    if [ $((WAIT % 10)) -eq 0 ]; then
        printf ". [%02ds/%02ds]\n" "$WAIT" "$MAX_WAIT"
    else
        printf "."
    fi

    if ! kill -0 "$BRAIN_PID" 2>/dev/null; then
        echo ""
        echo "❌ Brain crashed on startup."
        echo "Scroll up in this terminal for Brain's output."
        exit 1
    fi

    if [ $WAIT -ge $MAX_WAIT ]; then
        echo ""
        echo "❌ Timed out waiting for Brain. Watch the terminal output above."
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
kill_hud_processes
"$VENV_PYTHON" src/hud.py > logs/hud.log 2>&1 &
HUD_PID=$!
echo $HUD_PID > /tmp/parakeet-hud.pid
echo "  HUD   PID: $HUD_PID  |  log: logs/hud.log"
sleep 0.8   # give Qt/Cocoa time to connect to WindowServer

# ── Start Ear (foreground — Ctrl+C to stop) ─────────────────────
BACKEND="$BACKEND" "$VENV_PYTHON" src/ear.py
