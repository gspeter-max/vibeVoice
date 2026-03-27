#!/bin/bash
# Stop Kokoro TTS service

set -e

PID_FILE="/tmp/tts-kokoro.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "🛑 Stopping Kokoro TTS service (PID: $PID)..."
    kill "$PID" 2>/dev/null || true

    # Wait for process to stop
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done

    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 "$PID" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "✅ Stopped"
else
    echo "❌ Kokoro TTS service not running (no PID file)"
fi
