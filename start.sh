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
ORANGE='\033[38;5;208m'
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

export BACKEND="${BACKEND:-parakeet}"
export QT_MAC_WANTS_LAYER=1      # Intel Mac Sonoma+ fix
export KMP_DUPLICATE_LIB_OK=TRUE # Fix: ctranslate2 and others bundle libiomp5.dylib
export PARAKEET_THREADS="${PARAKEET_THREADS:-}"
export STREAMING_TELEMETRY_ENABLED="${STREAMING_TELEMETRY_ENABLED:-0}"
export RECORDING_MODE="${RECORDING_MODE:-silence_streaming}"
export STREAMING_TELEMETRY_DIR="${STREAMING_TELEMETRY_DIR:-logs/streaming_sessions}"
export EAR_TO_BRAIN_SCOKET_PATH="${EAR_TO_BRAIN_SCOKET_PATH:-/tmp/ear_to_brain_socket_path.sock}"

cleanup() {
    echo -e "\n  Cleaning up..."
    kill_pid_file_process /tmp/parakeet-brain.pid
    kill_pid_file_process /tmp/parakeet-hud.pid
    kill_hud_processes
    rm -f $EAR_TO_BRAIN_SCOKET_PATH
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
"$VENV_PYTHON" src/wizard/wizard.py

# 2. Load the UPDATED environment
[ -f .env ] && {
    set -a
    . ./.env
    set +a
}
# Startup Banner
echo -e "
  ${ORANGE}┌──────────────────────────────────────────────┐${NC}
  ${ORANGE}│  * Welcome to VibeVoice research preview!    │${NC}
  ${ORANGE}└──────────────────────────────────────────────┘${NC}
"
echo -e "${ORANGE}"
cat <<'EOF'
  ██╗   ██╗██╗██████╗ ███████╗██╗   ██╗██████╗ ██╗ ██████╗███████╗
  ██║   ██║██║██╔══██╗██╔════╝██║   ██║██╔══██╗██║██╔════╝██╔════╝
  ██║   ██║██║██████╔╝█████╗  ██║   ██║██║  ██║██║██║     █████╗  
  ╚██╗ ██╔╝██║██╔══██╗██╔══╝  ╚██╗ ██╔╝██║  ██║██║██║     ██╔══╝  
   ╚████╔╝ ██║██████╔╝███████╗ ╚████╔╝ ██████╔╝██║╚██████╗███████╗
    ╚═══╝  ╚═╝╚═════╝ ╚══════╝  ╚═══╝  ╚═════╝ ╚═╝ ╚═════╝╚══════╝
EOF
echo -e "${NC}"

echo -e "
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

# 1. Cleanup stale processes
printf "  [1/3] ⚙️  Cleaning up stale processes..."
kill_pid_file_process /tmp/parakeet-brain.pid
kill_pid_file_process /tmp/parakeet-hud.pid
kill_hud_processes
# cleanup brian.log
rm -rf logs/brain.log
rm -f $EAR_TO_BRAIN_SCOKET_PATH
mkdir -p logs
printf "\r\033[K  [1/3] ⚙️  Cleaning up stale processes...  ${GREEN}✓ Done${NC}\n"
if [[ "$BACKEND" == "nemotron" ]]; then
    MODEL_FOLDER="nemotron-0.6b-onnx"
else
    MODEL_NAME="${STT_MODEL:-parakeet-tdt-0.6b-v3}"
    if [[ "$MODEL_NAME" == *"moonshine"* ]]; then
        MODEL_FOLDER="sherpa-onnx-${MODEL_NAME}-en-int8"
    else
        MODEL_FOLDER="sherpa-onnx-nemo-${MODEL_NAME}-int8"
    fi
fi
# 2. Start Brain — launch BEFORE the spinner so the socket has time to appear

printf "  [3/3] 🖥️  Initializing HUD window..."
kill_hud_processes
"$VENV_PYTHON" src/ui/hud.py >logs/hud.log 2>&1 &
HUD_PID=$!
echo $HUD_PID >/tmp/parakeet-hud.pid
sleep 0.8
printf "\r\033[K  [3/3] 🖥️  Initializing HUD window...      ${GREEN}✓ Ready${NC} ${GRAY}(PID: $HUD_PID)${NC}\n"
echo "══════════════════════════════════════════════════"

[ -d "$HOME/.cache/parakeet-flow/models/$MODEL_FOLDER" ] || log_warn "First run: Downloading model (~1.5 GB)..."
printf "  [2/3] 🧠  Launching Brain server..."

"$VENV_PYTHON" src/backend/brain.py &
echo $! >/tmp/parakeet-brain.pid

sleep 0.2
BRAIN_PID=$(cat /tmp/parakeet-brain.pid)
WAIT=0
MAX_WAIT=300
SPINNER=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
while [ ! -S "$EAR_TO_BRAIN_SCOKET_PATH" ]; do
    sleep 0.1
    ((WAIT++))
    elapsed=$((WAIT / 10))

    SPIN="${SPINNER[WAIT % ${#SPINNER[@]}]}"
    printf "\r\033[K  [2/3] 🧠  Launching Brain server...       %s Waiting [%02ds/%02ds]" "$SPIN" "$elapsed" "$MAX_WAIT"

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
printf "\r\033[K  [2/3] 🧠  Launching Brain server...       ${GREEN}✓ Online${NC}\n"

# Start Ear
"$VENV_PYTHON" src/audio/ear_runtime/runtime.py
