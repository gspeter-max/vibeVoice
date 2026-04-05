# Streaming Notes

The current problem in voice-to-text streaming is accuracy at utterance boundaries.
If the system cuts audio at fixed time intervals, it can split words in half.
That gives the model broken phonemes, repeated words, or hallucinated text.

## Decision

Use Silero VAD.

Silero VAD is the speech boundary detector for this project. It decides when the user is speaking and when there is enough silence to treat the current audio as a complete utterance.

## Why this is better than fixed chunks

- Fixed chunks are simple, but they cut words in the middle.
- Speech segments from VAD are better because they follow natural pauses.
- The model gets complete speech, so accuracy improves.
- Streaming still happens in real time, but transcription is triggered on speech boundaries instead of arbitrary time slices.

## Streaming flow

1. Capture microphone audio continuously.
2. Feed small audio frames into Silero VAD.
3. Keep buffering audio while speech is active.
4. When VAD detects a silence boundary, close the current utterance.
5. Send that full utterance to `brain.py`.
6. Transcribe the whole utterance.
7. Reset the buffer and start collecting the next phrase.

## What this means for the user

- The app still streams audio in the background.
- The transcript appears after a short pause, not after every tiny chunk.
- The result should be more accurate because the model sees full phrases instead of broken fragments.

## Recommended tuning

- Use short audio frames for VAD, around 20-30 ms.
- Treat short pauses as part of the same sentence.
- Treat longer silence as the end of a speech segment.
- Keep the final transcription step separate from the live audio capture step.

## Minimal implementation shape

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad_model = load_silero_vad()
audio_buffer = []

def on_audio_chunk(chunk):
    audio_buffer.append(chunk)

def on_silence_detected():
    full_audio = concatenate(audio_buffer)
    speech_segments = get_speech_timestamps(full_audio, vad_model)

    for segment in speech_segments:
        brain.transcribe(segment)

    audio_buffer.clear()
```

## Summary

Silero VAD is the right choice here because it keeps streaming active while avoiding bad chunk boundaries. The core idea is simple: stream continuously, but only transcribe when the audio contains a complete speech segment.
