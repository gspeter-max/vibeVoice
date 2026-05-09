#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Parakeet Flow v2 — start.sh                                 ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

log_info() { echo "  $1"; }
log_warn() { echo "⚠️  $1"; }
log_error() { echo "❌ $1"; }

kill_pid_file_process() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

kill_hud_processes() {
    lsof -ti :57234 | xargs kill -9 2>/dev/null || true
    pkill -f "src/ui/hud.py" 2>/dev/null || true
}

cleanup() {
    echo -e "\n  Cleaning up..."
    kill_pid_file_process /tmp/parakeet-brain.pid
    kill_pid_file_process /tmp/parakeet-hud.pid
    kill_hud_processes
    rm -f /tmp/parakeet.sock
    log_info "Done. Goodbye."
}
trap cleanup EXIT INT TERM

# Configuration
VENV_PYTHON="./.venv/bin/python"

# 0. Ensure .venv exists (Auto-setup for new clones)
if [ ! -f "$VENV_PYTHON" ]; then
    log_info "No virtual environment found. Starting auto-setup..."
    
    # Linux-specific check for PortAudio (pyaudio dependency)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! ldconfig -p | grep -q libportaudio >/dev/null 2>&1; then
            log_error "Missing system audio headers (PortAudio)."
            log_info "Please run: sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-dev gcc"
            exit 1
        fi
    fi

    if command -v uv >/dev/null 2>&1; then
        log_info "Detected 'uv'. Running 'uv sync'..."
        uv sync
    else
        log_warn "'uv' not found. Falling back to standard venv/pip (this might be slower)..."
        python3 -m venv .venv
        ./.venv/bin/pip install -e .
    fi
    
    if [ ! -f "$VENV_PYTHON" ]; then
        log_error "Auto-setup failed. Please install dependencies manually."
        exit 1
    fi
    log_info "✅ Environment setup complete."
fi

# 1. Run the Setup Wizard (Foreground)
# This handles Provider selection, API keys, Mode, and Telemetry using Rich.
"$VENV_PYTHON" src/utils/wizard.py

# 2. Load the UPDATED environment
[ -f .env ] && { set -a; . ./.env; set +a; }

export BACKEND="${BACKEND:-parakeet}"
export QT_MAC_WANTS_LAYER=1 # Intel Mac Sonoma+ fix
export KMP_DUPLICATE_LIB_OK=TRUE # Fix: ctranslate2 and others bundle libiomp5.dylib
export PARAKEET_THREADS="${PARAKEET_THREADS:-}"
export STREAMING_TELEMETRY_ENABLED="${STREAMING_TELEMETRY_ENABLED:-0}"
export RECORDING_MODE="${RECORDING_MODE:-silence_streaming}"
export STREAMING_TELEMETRY_DIR="${STREAMING_TELEMETRY_DIR:-logs/streaming_sessions}"


# Startup Banner
echo "
╔══════════════════════════════════════════════════╗
║        🎙️  PARAKEET FLOW  v2                      ║
╚══════════════════════════════════════════════════╝
  Backend  : $BACKEND
  Mode     : $RECORDING_MODE
  Telemetry: $([ "$STREAMING_TELEMETRY_ENABLED" = "1" ] && echo "enabled -> $STREAMING_TELEMETRY_DIR" || echo "disabled")
  Threads  : ${PARAKEET_THREADS:-auto (all cores)}
  Python   : $($VENV_PYTHON --version 2>&1)
  Theme    : Dark with premium white waveform bars
  Logs     : live terminal output
"

[[ "${START_SH_DRY_RUN:-0}" == "1" ]] && { log_info "Dry run: exiting before Brain startup"; exit 0; }

# Sanity Check
[ -f "$VENV_PYTHON" ] || { log_error ".venv not found. Run: uv venv && uv pip install -e ."; exit 1; }

# Cleanup stale processes
log_info "Cleaning up old processes..."

kill_pid_file_process /tmp/parakeet-brain.pid
kill_pid_file_process /tmp/parakeet-hud.pid
kill_hud_processes
rm -f /tmp/parakeet.sock
mkdir -p logs

# Start Brain
log_info "Starting Brain..."
"$VENV_PYTHON" src/backend/brain.py &
BRAIN_PID=$!
echo $BRAIN_PID > /tmp/parakeet-brain.pid
log_info "Brain PID: $BRAIN_PID | live terminal output"

# Wait for Brain
echo -n "  Waiting for Brain to be ready"
[ -d ~/.cache/parakeet-flow/models/deepdml ] || log_warn "First run: Downloading model (~1.5 GB)..."

WAIT=0
MAX_WAIT=300
while [ ! -S /tmp/parakeet.sock ]; do
    sleep 1
    ((WAIT++))
    ((WAIT % 10 == 0)) && printf ". [%02ds/%02ds]\n  " "$WAIT" "$MAX_WAIT" || printf "."
    kill -0 "$BRAIN_PID" 2>/dev/null || { echo; log_error "Brain crashed on startup."; exit 1; }
    [[ $WAIT -ge $MAX_WAIT ]] && { echo; log_error "Timed out waiting for Brain."; exit 1; }
done
echo -e "\n\n  ✅ Brain is Online!\n══════════════════════════════════════════════════\n"

# Start HUD
log_info "Starting HUD..."
kill_hud_processes
"$VENV_PYTHON" src/ui/hud.py > logs/hud.log 2>&1 &
HUD_PID=$!
echo $HUD_PID > /tmp/parakeet-hud.pid
log_info "HUD PID: $HUD_PID | log: logs/hud.log"
sleep 0.8 # Allow Qt/Cocoa connection

# Start Ear
"$VENV_PYTHON" src/audio/ear_runtime/runtime.py
