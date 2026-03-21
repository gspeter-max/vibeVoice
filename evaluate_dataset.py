import time
import re
import string
from datasets import load_dataset
from faster_whisper import WhisperModel
import jiwer
import numpy as np

# Number of samples to evaluate
NUM_SAMPLES = 50

print(f"Loading first {NUM_SAMPLES} samples from LibriSpeech (test-clean)...")
# We use streaming to avoid downloading the entire 30GB dataset
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

# Fetch samples into memory
samples = []
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    
    # Resample is usually handled by the dataset if it's 16kHz, LibriSpeech is 16kHz
    audio_array = sample["audio"]["array"].astype(np.float32)
    # LibriSpeech text is uppercase with no punctuation, let's normalize to lowercase
    reference_text = sample["text"].lower()
    
    samples.append({
        "audio": audio_array,
        "reference": reference_text,
        "duration": len(audio_array) / 16000.0
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
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")
    
    model = WhisperModel(model_name, device="cpu", compute_type="default", cpu_threads=4)
    
    # Warmup
    model.transcribe(samples[0]["audio"], language="en", vad_filter=True)
    
    predictions = []
    references = [s["reference"] for s in samples]
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        segments, _ = model.transcribe(sample["audio"], language="en", vad_filter=True, beam_size=5)
        pred_text = " ".join([seg.text for seg in segments])
        predictions.append(normalize_text(pred_text))
        
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate WER
    wer = jiwer.wer(references, predictions)
    
    # Calculate Real-Time Factor (RTF)
    # RTF < 1 means faster than real-time. RTF = processing_time / audio_duration
    rtf = inference_time / total_audio_duration
    
    print(f"Time taken: {inference_time:.2f}s for {total_audio_duration:.2f}s of audio")
    print(f"Real-Time Factor (RTF): {rtf:.3f}x (lower is faster)")
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
    res = evaluate_model(m)
    results.append(res)

print("\n\n" + "="*60)
print(f"{'Model':<15} | {'WER (%)':<10} | {'RTF':<10} | {'Total Time (s)':<15}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<15} | {r['wer']:<10.2f} | {r['rtf']:<10.3f} | {r['time']:<15.2f}")
print("="*60)
print("* WER: Word Error Rate (lower is better)")
print("* RTF: Real-Time Factor. E.g., 0.1 means 10s of audio takes 1s to transcribe (lower is faster)")
