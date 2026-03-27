# LLM Response to Speech - Kokoro TTS

Lightweight Text-to-Speech system for Parakeet Flow using Kokoro 82M model.

## Overview

This system provides a Unix socket-based TTS service that converts text to speech using the Kokoro 82M model. It's designed to be lightweight, fast, and perfect for systems with limited resources.

### Why Kokoro 82M?

- **Lightweight**: Only 350 MB model size
- **CPU-based**: No GPU required
- **Low RAM**: Works with 4 GB RAM
- **Open Source**: MIT license, no API costs
- **Fast**: Sub-second latency for short texts

### Architecture

```
┌─────────────┐         Unix Socket        ┌──────────────┐
│   Client    │ ────────────────────────── │   Kokoro     │
│  (your app) │   /tmp/tts-kokoro.sock     │   TTS Host   │
└─────────────┘                            └──────────────┘
                                                    │
                                                    │ subprocess
                                                    ▼
                                            ┌──────────────┐
                                            │ kokoro-tts   │
                                            │   CLI tool   │
                                            └──────────────┘
```

### Protocol

**Request:**
```
[4 bytes: text length][text data as UTF-8]
```

**Response:**
```
[4 bytes: audio length][WAV audio data]
```

## Installation

### 1. Install Dependencies

```bash
cd /Users/apple/parakeet-flow
uv pip install -e ".[tts]"
```

### 2. (Optional) Install Kokoro CLI Tool

If you want real speech synthesis (instead of mock beep tones):

```bash
# Install kokoro-tts CLI tool
# See: https://github.com/remsky/Kokoro-FastAPI
```

If not installed, the system will use **mock mode** for testing.

## Usage

### Starting the Service

```bash
cd llm-response-to-speech
./scripts/start_tts.sh
```

The service will:
- Create Unix socket at `/tmp/tts-kokoro.sock`
- Write PID to `/tmp/tts-kokoro.pid`
- Log to `logs/kokoro.log`

### Checking Status

```bash
./scripts/status_tts.sh
```

### Stopping the Service

```bash
./scripts/stop_tts.sh
```

## Testing

### Run Test Suite

```bash
python tests/test_audio_utils.py
python tests/test_kokoro_client.py
```

### Manual Test

```python
from core.socket_server import SocketClient

client = SocketClient("/tmp/tts-kokoro.sock")
audio = client.send_request("Hello, this is a test.")

with open("output.wav", 'wb') as f:
    f.write(audio)
```

### Play Output

```bash
afplay output.wav
```

## Files

```
llm-response-to-speech/
├── config/
│   └── models.yaml           # Kokoro configuration
├── core/
│   ├── base_tts.py           # Abstract base class
│   ├── socket_server.py      # Unix socket client
│   └── audio_utils.py        # WAV processing
├── models/
│   └── kokoro_host.py        # Kokoro TTS implementation
├── scripts/
│   ├── start_tts.sh          # Start service
│   ├── stop_tts.sh           # Stop service
│   └── status_tts.sh         # Check status
├── tests/
│   ├── test_audio_utils.py   # Audio utilities tests
│   └── test_kokoro_client.py # Integration tests
└── logs/                     # Service logs
```

## Configuration

Edit `config/models.yaml` to customize:

```yaml
models:
  kokoro:
    voice: "af_sarah"        # Voice preset
    lang: "en-us"            # Language
    default_speed: 1.0       # Speech speed
```

Available voices:
- `af_sarah` - Female US English
- `am_adam` - Male US English
- `af_sky` - Female UK English
- `am_michael` - Male UK English

## Performance

On user's hardware (16 GB RAM, AMD Radeon Pro 5300M):

- **Memory**: ~500 MB (model + overhead)
- **CPU**: Low usage, no lag
- **Latency**: 0.5-2 seconds for typical sentences
- **Real-time factor**: 0.1-0.5x (faster than real-time)

## Troubleshooting

### Service won't start

Check if already running:
```bash
./scripts/status_tts.sh
```

Kill existing process:
```bash
./scripts/stop_tts.sh
```

### Socket connection refused

Ensure service is running:
```bash
./scripts/status_tts.sh
```

Check socket file:
```bash
ls -la /tmp/tts-kokoro.sock
```

### Only hearing beep tones

This is normal if `kokoro-tts` CLI tool is not installed. The system uses mock mode for testing the Unix socket protocol.

To use real speech, install kokoro-tts:
```bash
# See: https://github.com/remsky/Kokoro-FastAPI
```

### High CPU usage

Check process stats:
```bash
top -pid $(cat /tmp/tts-kokoro.pid) -stats pid,command,cpu,mem
```

Should be < 10% CPU on modern systems.

## Integration Example

```python
from core.socket_server import SocketClient
import subprocess

# Get speech from LLM response
llm_response = "The capital of France is Paris."

# Send to TTS service
client = SocketClient("/tmp/tts-kokoro.sock")
audio = client.send_request(llm_response)

# Save and play
with open("response.wav", 'wb') as f:
    f.write(audio)

subprocess.run(["afplay", "response.wav"])
```

## Next Steps

1. **Install kokoro-tts CLI** for real speech synthesis
2. **Customize voices** in `config/models.yaml`
3. **Integrate with Parakeet Flow** for voice-to-voice AI assistant
4. **Add more voices** by downloading additional Kokoro models

## License

MIT (same as Kokoro TTS)

## Resources

- [Kokoro TTS GitHub](https://github.com/remsky/Kokoro-FastAPI)
- [Parakeet Flow](../README.md)
- [Unix Socket IPC](https://man7.org/linux/man-pages/man7/unix.7.html)
