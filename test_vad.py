import numpy as np
import os
import wave
from src.brain import SileroVAD

vad = SileroVAD(os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx"))

with wave.open("demo_01_current_arial.wav", "rb") as wf:
    raw = wf.readframes(wf.getnframes())

audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

chunk_size = 512
speech_chunks = []
hangover_chunks = 0 
max_hangover = int((16000 * 0.3) / chunk_size)

vad.reset()
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    if len(chunk) < chunk_size:
        break
        
    score = vad.is_speech(chunk)
    if score > 0.5:
        speech_chunks.append(chunk)
        hangover_chunks = max_hangover
    elif hangover_chunks > 0:
        speech_chunks.append(chunk)
        hangover_chunks -= 1

if len(speech_chunks) > 0:
    stripped = np.concatenate(speech_chunks)
    print(f"Original: {len(audio)/16000:.2f}s, Stripped: {len(stripped)/16000:.2f}s")
else:
    print("VAD kept nothing!")
