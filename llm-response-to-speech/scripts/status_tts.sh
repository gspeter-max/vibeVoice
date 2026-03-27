#!/bin/bash
# Check Kokoro TTS service status

PID_FILE="/tmp/tts-kokoro.pid"
SOCKET_FILE="/tmp/tts-kokoro.sock"

echo "🎙️  Kokoro TTS Service Status"
echo "============================="
echo ""

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Running (PID: $PID)"
        echo ""
        echo "Socket: $SOCKET_FILE"
        if [ -S "$SOCKET_FILE" ]; then
            echo "✅ Socket exists"
        else
            echo "❌ Socket missing"
        fi
        echo ""
        echo "Recent logs:"
        tail -n 5 logs/kokoro.log 2>/dev/null || echo "No logs yet"
    else
        echo "❌ Not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "❌ Not running"
fi
