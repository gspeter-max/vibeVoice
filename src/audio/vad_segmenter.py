"""
Simple helpers for voice activity detection.

This file does two jobs:

1. Run the Silero VAD model on small audio frames.
2. Keep track of one spoken utterance so the caller can know:
   - when speech started
   - when silence has lasted long enough
   - when it is time to flush the saved audio

This file works with 16 kHz mono PCM16 audio.
It does not do speech-to-text.
It only decides whether audio looks like speech or silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.settings import settings


@dataclass()
class SileroVAD:
    """ONNX-based Silero Voice Activity Detection (VAD) model wrapper."""

    def __init__(self, model_path: str):
        """Load the VAD model and initialize the recurrent state and context buffers."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            options,
            providers=["CPUExecutionProvider"],
        )

        state_input = next(
            (inp for inp in self.session.get_inputs() if inp.name == "state"),
            None,
        )
        if state_input:
            self._version = 5
            state_shape = (
                [dim if isinstance(dim, int) else 1 for dim in state_input.shape]
                if state_input.shape
                else [2, 1, 128]
            )
            self._state = np.zeros(state_shape, dtype=np.float32)
            # 64-sample rolling context window required by Silero V5 ONNX streaming.
            # Without it the model sees a cold start every frame and returns ~0.001.
            self._context = np.zeros((1, 64), dtype=np.float32)
        else:
            self._version = 3
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        """Reset the internal model recurrent state buffers to clear context."""
        if self._version == 5:
            self._state = np.zeros_like(self._state)
            self._context = np.zeros_like(self._context)
        else:
            self._h = np.zeros_like(self._h)
            self._c = np.zeros_like(self._c)

    def is_speech(
        self, audio_samples: np.ndarray, sample_rate: int = settings.rate
    ) -> float:
        """Analyze a 512-sample audio frame and return speech probability (0.0 to 1.0)."""
        if len(audio_samples) == 0:
            return 0.0

        # The model expects a fixed-size frame, so we normalize the length here.
        if len(audio_samples) < 512:
            audio_samples = np.pad(audio_samples, (0, 512 - len(audio_samples)))
        elif len(audio_samples) > 512:
            audio_samples = audio_samples[:512]

        if self._version == 5:
            # Silero V5 ONNX streaming expects the current 512 samples preceded
            # by 64 samples of context from the previous frame. We maintain that
            # rolling window in self._context and prepend it here.
            audio_samples_2d = audio_samples.reshape(1, -1)
            input_with_context = np.concatenate(
                [self._context, audio_samples_2d], axis=1
            )
            ort_inputs = {
                "input": input_with_context,
                "sr": np.array([sample_rate], dtype=np.int64),
                "state": self._state,
            }
            out, new_state = self.session.run(None, ort_inputs)
            self._state = new_state
            # Slide the context window forward: keep the last 64 samples.
            self._context = input_with_context[:, -64:].copy()
        else:
            ort_inputs = {
                "input": audio_samples.reshape(1, -1),
                "sr": np.array([sample_rate], dtype=np.int64),
                "h": self._h,
                "c": self._c,
            }
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c

        return float(out[0][0])


class SileroUtteranceGate:
    """Manages audio buffering and speech boundary detection using VAD and energy levels."""

    def __init__(
        self,
        vad_engine: Any,
        *,
        sample_rate: int = settings.rate,
        frame_samples: int = 512,
        voice_threshold: float = 0.5,
        silence_timeout_s: float = settings.silence_timeout_seconds,
        min_utterance_bytes: int = 8000,
        energy_threshold: float = 0.03,
        energy_ratio: float = 2.5,
    ):
        """Set up the gate with audio analysis buffers and sensitivity thresholds."""
        self.vad_engine = vad_engine
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.voice_threshold = voice_threshold
        self.silence_timeout_s = silence_timeout_s
        self.min_utterance_bytes = min_utterance_bytes
        self.energy_threshold = energy_threshold
        self.energy_ratio = energy_ratio
        self._buffer = bytearray()
        self._analysis_buffer = bytearray()
        self._raw_analysis_buffer = bytearray()
        self._speech_started = False
        self._last_voice_time = 0.0
        self._finalize_armed = False
        self._finalize_time = 0.0
        self._last_score = 0.0
        self._max_score = 0.0
        self._last_energy = 0.0
        self._last_dynamic_threshold = energy_threshold
        self._noise_floor = 0.0

    @property
    def reset(self) -> None:
        """Clear all buffered audio and reset speech detection timers."""
        self._buffer.clear()
        self._analysis_buffer.clear()
        self._raw_analysis_buffer.clear()
        self._speech_started = False
        self._last_voice_time = 0.0
        self._finalize_armed = False
        self._finalize_time = 0.0
        self._last_score = 0.0
        self._max_score = 0.0
        self._last_energy = 0.0
        self._last_dynamic_threshold = self.energy_threshold
        self._noise_floor = 0.0
        if self.vad_engine is not None and hasattr(self.vad_engine, "reset"):
            self.vad_engine.reset()

    def arm_finalize(self, now: float) -> None:
        """Arm the gate to stop recording and flush remaining audio after a short timeout."""
        self._finalize_armed = True
        self._finalize_time = now

    @property
    def has_speech_started(self) -> bool:
        """Return True if speech has been detected in the current utterance."""
        return self._speech_started

    def push(
        self, audio_chunk: bytes, now: float, analysis_chunk: bytes | None = None
    ) -> bool:
        """Process new audio bytes through VAD and energy checks. Return True if active speech is found."""
        if not audio_chunk:
            return False
        self._raw_analysis_buffer.extend(audio_chunk)
        self._analysis_buffer.extend(
            analysis_chunk if analysis_chunk is not None else audio_chunk
        )
        # Keep the full utterance audio so it can be flushed later.
        self._buffer.extend(audio_chunk)

        frame_bytes = self.frame_samples * 2
        speech_detected = False
        while (
            len(self._analysis_buffer) >= frame_bytes
            and len(self._raw_analysis_buffer) >= frame_bytes
        ):
            # Read one analysis frame and remove it from the queue.
            frame_bytes_data = bytes(self._analysis_buffer[:frame_bytes])
            raw_frame_bytes = bytes(self._raw_analysis_buffer[:frame_bytes])
            del self._analysis_buffer[:frame_bytes]
            del self._raw_analysis_buffer[:frame_bytes]

            # Convert PCM16 bytes into normalized float audio for the model.
            audio = (
                np.frombuffer(frame_bytes_data, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            raw_audio_for_energy_detection = (
                np.frombuffer(raw_frame_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            score = (
                1.0
                if self.vad_engine is None
                else self.vad_engine.is_speech(
                    audio,
                    sample_rate=self.sample_rate,
                )
            )
            frame_rms = float(np.sqrt(np.mean(raw_audio_for_energy_detection**2)))
            self._last_energy = frame_rms
            self._last_score = float(score)
            if score > self._max_score:
                self._max_score = float(score)

            # Build an energy threshold that can adapt to room noise.
            dynamic_threshold = max(
                self.energy_threshold,
                self._noise_floor * self.energy_ratio,
            )
            self._last_dynamic_threshold = dynamic_threshold
            speech_by_energy = frame_rms >= dynamic_threshold

            if score > self.voice_threshold or speech_by_energy:
                # Any positive speech frame moves the utterance into speech state.
                speech_detected = True
                self._speech_started = True
                self._last_voice_time = now
            else:
                # Frames that are not speech help us slowly learn background noise.
                self._noise_floor = (
                    frame_rms
                    if self._noise_floor == 0.0
                    else (0.95 * self._noise_floor) + (0.05 * frame_rms)
                )

        return speech_detected

    def should_finalize(self, now: float) -> bool:
        """Return True if the user has stopped speaking and the chunk is ready to send."""
        if self._speech_started:
            # Do not finalize very tiny audio fragments after speech starts.
            if len(self._buffer) < self.min_utterance_bytes:
                return False
            return (now - self._last_voice_time) >= self.silence_timeout_s

        return (
            self._finalize_armed
            and (now - self._finalize_time) >= self.silence_timeout_s
        )

    def silence_len(self, now: float) -> float:
        """Return the number of seconds that have passed since speech was last detected."""
        return now - self._last_voice_time

    def finalize_elapsed(self, now: float) -> float:
        """Return the elapsed time in seconds since the finalize countdown was armed."""
        return now - self._finalize_time

    @property
    def last_score(self) -> float:
        """Return the latest VAD probability score."""
        return self._last_score

    @property
    def max_score(self) -> float:
        """Return the peak VAD probability score achieved during this utterance."""
        return self._max_score

    @property
    def last_energy(self) -> float:
        """Return the RMS energy of the last processed audio frame."""
        return self._last_energy

    @property
    def last_dynamic_threshold(self) -> float:
        """Return the current dynamic energy threshold adjusted for background noise."""
        return self._last_dynamic_threshold

    @property
    def flush(self) -> bytes:
        """Return the entire buffered audio utterance as bytes and reset the gate."""
        audio = bytes(self._buffer)
        self.reset
        return audio
