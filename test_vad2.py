import numpy as np
import os
import wave
from src.brain import SileroVAD

vad = SileroVAD(os.path.expanduser("~/.cache/parakeet-flow/vad/silero_vad.onnx"))

with wave.open("demo_01_current_arial.wav", "rb") as wf:
    raw = wf.readframes(wf.getnframes())

audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

chunk_size = 512
vad.reset()
scores = []
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    if len(chunk) < chunk_size:
        break
    score = vad.is_speech(chunk)
    scores.append(score)

print("Max score:", max(scores))
print("Mean score:", sum(scores)/len(scores))
