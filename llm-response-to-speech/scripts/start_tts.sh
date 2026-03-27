#!/bin/bash
# Start Kokoro TTS service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🎙️  Starting Kokoro TTS service..."
echo ""

# Run Kokoro host in background
PYTHONPATH="$PROJECT_DIR" ../.venv/bin/python -c "
from models.kokoro_host import KokoroTTSHost
host = KokoroTTSHost()
try:
    host.start()
except KeyboardInterrupt:
    host.stop()
" &

# Give it a moment to start
sleep 1

# Check if it started
if [ -f "/tmp/tts-kokoro.pid" ]; then
    PID=$(cat /tmp/tts-kokoro.pid)
    echo "✅ Kokoro TTS service started (PID: $PID)"
    echo ""
    echo "Socket: /tmp/tts-kokoro.sock"
    echo "Logs: logs/kokoro.log"
    echo ""
    echo "To stop: ./scripts/stop_tts.sh"
    echo "To test: python tests/test_kokoro_client.py"
else
    echo "❌ Failed to start service"
    exit 1
fi
