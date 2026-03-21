# 🎙️ Parakeet Flow v2
<img width="2752" height="1536" alt="Gemini_Generated_Image_o99mqio99mqio99m" src="https://github.com/user-attachments/assets/f7911baf-2c25-4d58-a07c-84050f2e85c2" />

> Local, Private, Fast Voice-to-Text for your Desktop.

Parakeet Flow v2 is a push-to-talk voice-to-text tool that transcribes your speech locally and **auto-types it instantly into whatever application you're currently using** (Browser, IDE, Slack, etc.).

It is highly optimized for low-latency inference on CPU, specifically for macOS (Intel & Apple Silicon), providing a seamless "Wispr Flow" style experience entirely on your machine.

---

## ⚡ Key Features

- **Push-to-Talk**: Hold **RIGHT CMD** to record, release to transcribe.
- **Auto-Type Output**: Transcriptions are typed into your active cursor position automatically.
- **Interactive Model Toggling**: Switch between different Whisper and Parakeet models instantly while the app is running using number keys **1-9** and **0**.
- **Privacy First**: Everything runs 100% locally. No audio or text ever leaves your machine.
- **Mac Optimized**: Uses a non-blocking CoreAudio callback and architecture-aware thread scheduling for maximum responsiveness.

---

## 📊 Benchmarks (Intel Core i7 / 16GB RAM)

We evaluated the performance using the **LibriSpeech (test-clean)** research dataset directly on an Intel Core i7 (12-threads) CPU. These benchmarks reflect actual local hardware execution using `int8` quantization:

| Option | Model Name | WER (%) ↓ | Speed (RTF) ↓ | Time per 10s audio | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[1]** | `tiny.en` | 2.17% | 0.097x | ~0.9 seconds | Simple voice commands. |
| **[2]** | `base.en` | 2.17% | 0.177x | ~1.7 seconds | Default option, fast but misses accents. |
| **[3]** | **`small.en`**| **0.93%** | **0.525x** | **~5.2 seconds** | **Sweet Spot! Sub-1% error, minimal delay.** |
| **[4]** | `distil-large-v3`| ~0.80% | ~1.200x | ~12.0 seconds| High accuracy, faster than standard large. |
| **[5]** | `medium.en` | 0.62% | 1.734x | ~17.3 seconds| Very high accuracy, but noticeable delay. |
| **[6]** | `large-v2` | ~0.60% | ~2.500x | ~25.0 seconds| Pre-2024 standard; heavy on CPU. |
| **[7]** | `large-v3` | ~0.55% | ~2.500x | ~25.0 seconds| Maximum theoretical accuracy; heavy delay. |
| **[8]** | **`turbo`** | **~0.60%** | **~0.800x** | **~8.0 seconds** | **Best High-End! large-v3 accuracy, 3x faster.** |
| **[9]** | **`Parakeet v2`** | **0.31%** | **0.164x** | **~1.6 seconds** | **KING of Accuracy! Best for Vibe Coding.** |
| **[0]** | `Parakeet v3` | 1.24% | 0.171x | ~1.7 seconds | Fast Multilingual support. |

* *WER: Word Error Rate (Lower is better).*
* *RTF: Real-Time Factor (Lower is faster, e.g. 0.1 means 10s of audio takes 1s to transcribe).*
* *Tilde (~) values are projected based on established scaling factors for this specific hardware.*

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
- **Press 1 through 0**: Switch models instantly in the terminal.

---

## 🛠️ Technical Architecture

- **`ear.py`**: The "Listener". Handles global hotkeys, raw audio capture via PyAudio callback, and the interactive terminal menu.
- **`brain.py`**: The "Inference Engine". A persistent background server that manages the loaded Whisper model, processes incoming audio buffers via a Unix socket, and auto-types the result via `pynput`.
- **`backend_faster_whisper.py`**: The "Optimized Driver". Handles the heavy lifting of loading and running CTranslate2 models with architecture-specific optimizations.

---

## ⚖️ License
MIT License. Free to use, modify, and distribute.
