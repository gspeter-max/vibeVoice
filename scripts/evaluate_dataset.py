import time
import re
import string
import io
import librosa
from datasets import load_dataset, Audio
from faster_whisper import WhisperModel
import jiwer
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Number of samples to evaluate
NUM_SAMPLES = 15

print(f"Loading first {NUM_SAMPLES} samples from LibriSpeech (test-clean)...")
try:
    # Load dataset with streaming
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    
    # CRITICAL: Cast the audio column to NOT decode automatically. 
    # This bypasses the torchcodec/ImportError issue.
    dataset = dataset.cast_column("audio", Audio(decode=False))
except Exception as e:
    print(f"Error initializing dataset: {e}")
    exit(1)

# Fetch and decode samples into memory
samples = []
print("Fetching and decoding samples manually...")
try:
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        
        audio_data = sample["audio"]
        
        # Since decode=False, audio_data['bytes'] contains the raw file content (WAV/FLAC/etc)
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_bytes = io.BytesIO(audio_data["bytes"])
            # librosa.load is very robust for various formats
            audio_array, samplerate = librosa.load(audio_bytes, sr=16000)
        else:
            # Fallback if casting didn't work as expected
            print(f"Warning: Sample {i} did not provide raw bytes as expected.")
            continue
            
        # LibriSpeech text is uppercase with no punctuation, let's normalize to lowercase
        reference_text = sample["text"].lower()
        
        samples.append({
            "audio": audio_array,
            "reference": reference_text,
            "duration": len(audio_array) / float(samplerate)
        })
except Exception as e:
    print(f"Error during sample fetching: {e}")
    exit(1)

if not samples:
    print("No samples were loaded. Check connection or dataset accessibility.")
    exit(1)

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
    
    import multiprocessing
    if "parakeet-tdt" in model_name:
        import backend_parakeet
        model = backend_parakeet.load_model(model_name)
    else:
        # Initialize model with Mac-optimized settings for Intel CPU
        model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=multiprocessing.cpu_count())
    
    # Warmup
    if "parakeet-tdt" in model_name:
        import backend_parakeet
        backend_parakeet.transcribe(model, samples[0]["audio"])
    else:
        model.transcribe(samples[0]["audio"], language="en", vad_filter=True)
    
    predictions = []
    references = [s["reference"] for s in samples]
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        if "parakeet-tdt" in model_name:
            import backend_parakeet
            pred_text = backend_parakeet.transcribe(model, sample["audio"])
        else:
            # We use beam_size=5 for consistency with standard faster-whisper benchmarks
            segments, _ = model.transcribe(sample["audio"], language="en", vad_filter=True, beam_size=5)
            pred_text = " ".join([seg.text for seg in segments])
        predictions.append(normalize_text(pred_text))
        
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate WER using jiwer
    wer = jiwer.wer(references, predictions)
    
    # Calculate Real-Time Factor (RTF)
    rtf = inference_time / total_audio_duration
    
    print(f"Processed {len(samples)} samples.")
    print(f"Total Inference Time: {inference_time:.2f}s")
    print(f"Real-Time Factor (RTF): {rtf:.3f}x (lower is faster)")
    print(f"Word Error Rate (WER): {wer * 100:.2f}% (lower is better)")
    
    return {
        "model": model_name,
        "wer": wer * 100,
        "rtf": rtf,
        "time": inference_time
    }

models_to_test = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large-v2",
    "large-v3",
    "Systran/faster-distil-whisper-large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3"
]
results = []

for m in models_to_test:
    try:
        res = evaluate_model(m)
        results.append(res)
    except Exception as e:
        print(f"Error evaluating {m}: {e}")

if not results:
    print("No models were successfully evaluated.")
    exit(1)

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
