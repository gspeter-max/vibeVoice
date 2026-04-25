# 🎙️ Parakeet Flow v2
<img width="2752" height="1536" alt="Gemini_Generated_Image_o99mqio99mqio99m" src="https://github.com/user-attachments/assets/f7911baf-2c25-4d58-a07c-84050f2e85c2" />

> Local, Private, Fast Voice-to-Text for your Desktop.

Parakeet Flow v2 is a local voice-to-text tool that transcribes your speech and **auto-types it instantly into whatever application you're currently using** (Browser, IDE, Slack, etc.).

It is highly optimized for low-latency inference on CPU, specifically for macOS (Intel & Apple Silicon), providing a seamless "Wispr Flow" style experience entirely on your machine.

---

## ⚡ Key Features

- **Dual Control Methods**:
  - **Keyboard Shortcut**: Press **Right CMD** to start recording; a short release latches recording until the next press, while a longer hold stops on release
  - **Mouse Hold-to-Talk**: Hold **RIGHT mouse button** for 1 second to start recording, release to stop
- **Auto-Type Output**: Transcriptions are typed into your active cursor position automatically.
- **Interactive Model Toggling**: Switch between different Whisper and Parakeet models instantly while the app is running using number keys **1-9** and **0**.
- **Privacy First**: Everything runs 100% locally. No audio or text ever leaves your machine.
- **Mac Optimized**: Uses a non-blocking CoreAudio callback and architecture-aware thread scheduling for maximum responsiveness.

---

## 📊 Benchmarks (Intel Core i7 / 16GB RAM)

We evaluated the performance using the **LibriSpeech (test-clean)** research dataset directly on an Intel Core i7 (12-threads) CPU. These benchmarks reflect actual local hardware execution using `int8` quantization:

| Option | Model Name | WER (%) ↓ | Speed (RTF) ↓ | Time per 10s audio | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[1]** | `Conformer` | ~0.80% | ~0.100x | ~1.0 seconds | Balanced speed and accuracy. |
| **[2]** | `Moonshine` | ~0.70% | ~0.120x | ~1.2 seconds | Highly efficient for voice tasks. |
| **[3]** | **`Parakeet v2`** | **0.31%** | **0.164x** | **~1.6 seconds** | **KING of Accuracy! Best for Vibe Coding.** |
| **[4]** | `Parakeet v3` | 1.24% | 0.171x | ~1.7 seconds | Fast Multilingual support. |

* *WER: Word Error Rate (Lower is better).*
* *RTF: Real-Time Factor (Lower is faster, e.g. 0.1 means 10s of audio takes 1s to transcribe).*

---

## ⚙️ How It's Optimized

We achieved a **12x speed improvement** over standard Whisper by implementing the following:

1.  **Architecture-Aware Inference**: The backend dynamically detects your CPU (ARM64 vs x86_64). On Apple Silicon, it utilizes the **Accelerate framework** (float32) and avoids Efficiency-core bottlenecks by limiting to 4 threads. On Intel, it uses **AVX/VNNI** instructions via `int8` quantization.
2.  **Sherpa-ONNX Backend**: Powered by NVIDIA's NeMo models optimized for ONNX, which are significantly faster than Whisper-style architectures.
3.  **Callback Audio Pipeline**: Uses a non-blocking PyAudio callback to ensure the audio stream never drops or hangs during hotkey transitions.
4.  **Zero-Latency Switching**: Models are swapped in memory using Python's `gc` (garbage collection) to prevent memory leaks while allowing instant transitions between speed and accuracy.
5.  **Configurable Threading**: All models support adjustable thread counts via `PARAKEET_THREADS` for CPU optimization.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- macOS (CoreAudio supported)
- PortAudio (Install via Homebrew: `brew install portaudio`)

**Note on Models:** This project exclusively uses `sherpa-onnx` for high-performance inference. It is installed automatically with the project dependencies.

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

When the app starts, microphone choices are shown as simple menu numbers such as `0`, `1`, `2`. These are menu choices, not the raw PyAudio device indices.

**Control Recording (choose either method):**

**Method 1: Keyboard Shortcut**
- **Hold Right Command (⌘)**: Start recording
- **Release Right Command**: Stop recording, text will be typed into your active app

**Method 2: Mouse Hold-to-Talk**
- **Press and hold RIGHT mouse button** for 1 second → Start recording
- **Continue holding while speaking**
- **Release RIGHT mouse button** → Stop recording, text will be typed

**Model Switching:**
- **Press 1 through 4**: Switch models instantly in the terminal.

### 4. Advanced Configuration

#### Streaming Defaults
The live streaming path currently uses these defaults:

- `VAD_THRESHOLD=0.50`
- `VOICE_ACTIVITY_DETECTION_SILENCE_DETECTION_THRESHOLD_TIMEOUT=0.80`
- `VAD_SENSITIVITY_BOOST_FOR_SPEECH_DETECTION=1.0`
- `VAD_ENERGY_THRESHOLD=0.05`
- `VAD_ENERGY_RATIO=2.5`
- `OVERLAP_SECONDS=1.0`
- `MIN_CHUNK_SECONDS=8.0`

The capture path also applies a mild microphone gain before sending audio to ASR. The current in-code default is `1.2x`.

#### Thread Count Tuning
You can manually configure the number of threads used for transcription:

```bash
# Use specific number of threads (2, 4, 6, or 12)
PARAKEET_THREADS=6 ./start.sh

# Let system auto-detect (default, uses all cores)
./start.sh
```

Test different values to find the optimal setting for your hardware. The thread count is displayed at startup and in the brain log.

---

## 🛠️ Technical Architecture

- **`ear.py`**: The input/controller layer. It opens the microphone, listens for keyboard and mouse gestures, applies streaming VAD, cuts speech into chunks on silence, sends each chunk to Brain over a Unix socket, and sends HUD volume telemetry over UDP.
- **`brain.py`**: The inference server. In streaming session mode it decodes chunks as they arrive, deduplicates overlap against the previous chunk, waits for session commit, then pastes the stitched final text into the active app. It also still supports one-shot fallback decoding when raw audio is sent without chunk/session headers.
- **`hud.py`**: The always-on-top Qt HUD. It listens for state commands on TCP `57234` and volume/frequency packets on UDP `57235`.
- **`backend_parakeet.py`**: Optimized sherpa-onnx backend for NeMo-based models (Parakeet, Conformer, Moonshine).

---

## ⚖️ License
MIT License. Free to use, modify, and distribute.
