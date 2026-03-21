import time
import re
import string
import io
import librosa
from datasets import load_dataset
from faster_whisper import WhisperModel
import jiwer
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Number of samples to evaluate
NUM_SAMPLES = 50

print(f"Loading first {NUM_SAMPLES} samples from LibriSpeech (test-clean)...")
# We load the dataset with streaming=True and without automatic decoding to be safe
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

# Fetch and decode samples into memory
samples = []
print("Fetching and decoding samples...")
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    
    # The 'audio' field usually contains 'bytes' if not decoded, or 'array' if decoded.
    # If automatic decoding failed, we try to decode the bytes ourselves.
    audio_data = sample["audio"]
    
    if isinstance(audio_data, dict) and "array" in audio_data:
        # Already decoded (hopefully)
        audio_array = audio_data["array"].astype(np.float32)
        samplerate = audio_data.get("sampling_rate", 16000)
    elif isinstance(audio_data, dict) and "bytes" in audio_data:
        # Manual decode from bytes using librosa
        audio_bytes = io.BytesIO(audio_data["bytes"])
        audio_array, samplerate = librosa.load(audio_bytes, sr=16000)
    else:
        print(f"Warning: Sample {i} has unexpected format: {type(audio_data)}")
        continue
        
    # LibriSpeech text is uppercase with no punctuation, let's normalize to lowercase
    reference_text = sample["text"].lower()
    
    samples.append({
        "audio": audio_array,
        "reference": reference_text,
        "duration": len(audio_array) / float(samplerate)
    })

total_audio_duration = sum(s["duration"] for s in samples)
print(f"Loaded {len(samples)} samples. Total audio duration: {total_audio_duration:.2f} seconds.")

def normalize_text(text):
    """Basic text normalization to match LibriSpeech references (lowercase, no punctuation)"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_model(model_name):
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*60}")
    
    # Initialize model with Mac-optimized settings we discovered
    model = WhisperModel(model_name, device="cpu", compute_type="default", cpu_threads=4)
    
    # Warmup
    model.transcribe(samples[0]["audio"], language="en", vad_filter=True)
    
    predictions = []
    references = [s["reference"] for s in samples]
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        # We use beam_size=5 for consistency with standard faster-whisper benchmarks
        segments, _ = model.transcribe(sample["audio"], language="en", vad_filter=True, beam_size=5)
        pred_text = " ".join([seg.text for seg in segments])
        predictions.append(normalize_text(pred_text))
        
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate WER using jiwer
    wer = jiwer.wer(references, predictions)
    
    # Calculate Real-Time Factor (RTF)
    # RTF = processing_time / audio_duration
    rtf = inference_time / total_audio_duration
    
    print(f"Processed {len(samples)} samples.")
    print(f"Total Inference Time: {inference_time:.2f}s")
    print(f"Real-Time Factor (RTF): {rtf:.3f}x (lower is faster, < 1.0 is faster than real-time)")
    print(f"Word Error Rate (WER): {wer * 100:.2f}% (lower is better)")
    
    return {
        "model": model_name,
        "wer": wer * 100,
        "rtf": rtf,
        "time": inference_time
    }

models_to_test = ["tiny.en", "base.en", "small.en"]
results = []

for m in models_to_test:
    try:
        res = evaluate_model(m)
        results.append(res)
    except Exception as e:
        print(f"Error evaluating {m}: {e}")

print("\n\n" + "╔" + "═"*78 + "╗")
print("║" + " "*30 + "FINAL BENCHMARK SUMMARY" + " "*25 + "║")
print("╠" + "═"*17 + "╦" + "═"*13 + "╦" + "═"*13 + "╦" + "═"*31 + "╣")
print(f"║ {'Model':<15} ║ {'WER (%)':<11} ║ {'RTF':<11} ║ {'Transcription Speed':<29} ║")
print("╠" + "═"*17 + "╬" + "═"*13 + "╬" + "═"*13 + "╬" + "═"*31 + "╣")

for r in results:
    speed_label = f"{1.0/r['rtf']:.1f}x Real-time" if r['rtf'] > 0 else "N/A"
    print(f"║ {r['model']:<15} ║ {r['wer']:<11.2f} ║ {r['rtf']:<11.3f} ║ {speed_label:<29} ║")

print("╚" + "═"*17 + "╩" + "═"*13 + "╩" + "═"*13 + "╩" + "═"*31 + "╝")
print("* WER: Word Error Rate (lower is better)")
print("* RTF: Real-Time Factor. E.g., 0.1 means 10s of audio takes 1s to transcribe.")
