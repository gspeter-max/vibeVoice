# 🎙️ Parakeet Flow v2
<img width="2752" height="1536" alt="Gemini_Generated_Image_o99mqio99mqio99m" src="https://github.com/user-attachments/assets/f7911baf-2c25-4d58-a07c-84050f2e85c2" />

> Local, Private, Fast Voice-to-Text for your Desktop.

Parakeet Flow v2 is a push-to-talk voice-to-text tool that transcribes your speech locally and **auto-types it instantly into whatever application you're currently using** (Browser, IDE, Slack, etc.).

It is highly optimized for low-latency inference on CPU, specifically for macOS (Intel & Apple Silicon), providing a seamless "Wispr Flow" style experience entirely on your machine.

---

## ⚡ Key Features

- **Push-to-Talk**: Hold **RIGHT CMD** to record, release to transcribe.
- **Auto-Type Output**: Transcriptions are typed into your active cursor position automatically.
- **Interactive Model Toggling**: Switch between different Whisper models (Tiny, Base, Small, Large) instantly while the app is running using number keys **1-4**.
- **Privacy First**: Everything runs 100% locally. No audio or text ever leaves your machine.
- **Mac Optimized**: Uses a non-blocking CoreAudio callback and architecture-aware thread scheduling for maximum responsiveness.

---

## 📊 Benchmarks (Apple Silicon / Intel Mac)

We evaluated the performance on 5.5 minutes of real-world speech data from the **LibriSpeech (test-clean)** research dataset. These benchmarks reflect real-world usage on a modern Mac CPU:

| Model | WER (%) ↓ | RTF ↓ | Transcription Speed ↑ | Accuracy Note |
| :--- | :---: | :---: | :---: | :--- |
| **tiny.en** | 6.89% | 0.124 | **8.0x Real-time** | Fast, but misses some punctuation. |
| **base.en** | 6.45% | 0.219 | **4.6x Real-time** | **Perfect sweet spot (Default).** |
| **small.en** | 5.68% | 0.664 | 1.5x Real-time | Most accurate, but slight delay. |
| **distil-large-v3**| 2.5%* | ~1.6 | 0.6x Real-time | Extremely accurate, but slow on CPU. |

* *WER: Word Error Rate (Lower is better).*
* *RTF: Real-Time Factor. (0.1 means 10s of audio takes 1s to transcribe).*

---

## ⚙️ How It's Optimized

We achieved a **12x speed improvement** over standard Whisper by implementing the following:

1.  **Architecture-Aware Inference**: The backend dynamically detects your CPU (ARM64 vs x86_64). On Apple Silicon, it utilizes the **Accelerate framework** (float32) and avoids Efficiency-core bottlenecks by limiting to 4 threads. On Intel, it uses **AVX/VNNI** instructions via `int8` quantization.
2.  **CTranslate2 Backend**: Powered by `faster-whisper`, which is up to 4x faster than OpenAI's original implementation with lower memory usage.
3.  **Callback Audio Pipeline**: Uses a non-blocking PyAudio callback to ensure the audio stream never drops or hangs during hotkey transitions.
4.  **Zero-Latency Switching**: Models are swapped in memory using Python's `gc` (garbage collection) to prevent memory leaks while allowing instant transitions between speed and accuracy.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- macOS (CoreAudio supported)
- PortAudio (Install via Homebrew: `brew install portaudio`)

### 2. Installation
```bash
# Clone and enter the repo
git clone <this-repo-url>
cd parakeet-flow

# Setup virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Usage
Simply run the startup script:
```bash
./start.sh
```

- **Hold Right Command (⌘)**: Start speaking.
- **Release Right Command**: Text will be typed into your active app.
- **Press 1, 2, 3, or 4**: Switch models instantly in the terminal.

---

## 🛠️ Technical Architecture

- **`ear.py`**: The "Listener". Handles global hotkeys, raw audio capture via PyAudio callback, and the interactive terminal menu.
- **`brain.py`**: The "Inference Engine". A persistent background server that manages the loaded Whisper model, processes incoming audio buffers via a Unix socket, and auto-types the result via `pynput`.
- **`backend_faster_whisper.py`**: The "Optimized Driver". Handles the heavy lifting of loading and running CTranslate2 models with architecture-specific optimizations.

---

## ⚖️ License
MIT License. Free to use, modify, and distribute.
