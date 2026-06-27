#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Parakeet Flow v2 — start.sh                                 ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

# Colors
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m'

log_info() { echo -e "  ${BLUE}ℹ${NC} $1"; }
log_warn() { echo -e "  ${YELLOW}⚠️${NC} $1"; }
log_error() { echo -e "  ${RED}✗${NC} $1"; }
log_success() { echo -e "  ${GREEN}✓${NC} $1"; }

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
    log_success "Done. Goodbye."
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
[ -f .env ] && {
    set -a
    . ./.env
    set +a
}

export BACKEND="${BACKEND:-parakeet}"
export QT_MAC_WANTS_LAYER=1      # Intel Mac Sonoma+ fix
export KMP_DUPLICATE_LIB_OK=TRUE # Fix: ctranslate2 and others bundle libiomp5.dylib
export PARAKEET_THREADS="${PARAKEET_THREADS:-}"
export STREAMING_TELEMETRY_ENABLED="${STREAMING_TELEMETRY_ENABLED:-0}"
export RECORDING_MODE="${RECORDING_MODE:-silence_streaming}"
export STREAMING_TELEMETRY_DIR="${STREAMING_TELEMETRY_DIR:-logs/streaming_sessions}"

# Startup Banner
echo -e "
  ${CYAN}🎙️  PARAKEET FLOW v2${NC}
  ${GRAY}──────────────────────────────────────────────────${NC}
  ${BLUE}Backend${NC}   : $BACKEND
  ${BLUE}Mode${NC}      : $RECORDING_MODE
  ${BLUE}Telemetry${NC} : $([ "$STREAMING_TELEMETRY_ENABLED" = "1" ] && echo -e "${GREEN}Enabled${NC} (${GRAY}$STREAMING_TELEMETRY_DIR${NC})" || echo -e "${GRAY}Disabled${NC}")
  ${BLUE}Threads${NC}   : ${PARAKEET_THREADS:-auto}
  ${BLUE}Python${NC}    : $($VENV_PYTHON --version 2>&1 | awk '{print $2}')
  ${GRAY}──────────────────────────────────────────────────${NC}
"

[[ "${START_SH_DRY_RUN:-0}" == "1" ]] && {
    log_info "Dry run: exiting before Brain startup"
    exit 0
}

# Sanity Check
[ -f "$VENV_PYTHON" ] || {
    log_error ".venv not found. Run: uv venv && uv pip install -e ."
    exit 1
}

# Cleanup stale processes
log_info "Cleaning up old processes..."

kill_pid_file_process /tmp/parakeet-brain.pid
kill_pid_file_process /tmp/parakeet-hud.pid
kill_hud_processes
rm -f /tmp/parakeet.sock
mkdir -p logs

# Start Brain
log_info "Starting Brain..."
osascript -e "tell application \"Terminal\" to do script \"cd '$(pwd)' && $VENV_PYTHON src/backend/brain.py\""
# "$VENV_PYTHON" src/backend/brain.py &
BRAIN_PID=0
log_info "Brain started in a new Terminal window"

# Wait for Brain
[ -d ~/.cache/parakeet-flow/models/deepdml ] || log_warn "First run: Downloading model (~1.5 GB)..."

WAIT=0
MAX_WAIT=300
SPINNER=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
GREEN='\033[0;32m'
NC='\033[0m'

while [ ! -S /tmp/parakeet.sock ]; do
    sleep 0.2
    ((WAIT++))
    elapsed=$((WAIT/5))

    SPIN="${SPINNER[WAIT % ${#SPINNER[@]}]}"
    printf "\r\033[K  %s Waiting for Brain to be ready... [%02ds/%02ds]" "$SPIN" "$elapsed" "$MAX_WAIT"

    if [ "$BRAIN_PID" -ne 0 ]; then
        kill -0 "$BRAIN_PID" 2>/dev/null || {
            printf "\n"
            log_error "Brain crashed on startup."
            exit 1
        }
    fi
    if [[ $elapsed -ge $MAX_WAIT ]]; then
        printf "\n"
        log_error "Timed out waiting for Brain."
        exit 1
    fi
done
printf "\r\033[K${GREEN}  ✅ Brain is Online!${NC}\n"
echo "══════════════════════════════════════════════════"

# Start HUD
log_info "Starting HUD..."
kill_hud_processes
"$VENV_PYTHON" src/ui/hud.py >logs/hud.log 2>&1 &
HUD_PID=$!
echo $HUD_PID >/tmp/parakeet-hud.pid
log_info "HUD started (PID: ${GRAY}$HUD_PID${NC} | log: ${GRAY}logs/hud.log${NC})"
sleep 0.8 # Allow Qt/Cocoa connection

# Start Ear
"$VENV_PYTHON" -m pdb src/audio/ear_runtime/runtime.py
