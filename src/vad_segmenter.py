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

import numpy as np


class SileroVAD:
    """Small wrapper around the Silero ONNX model.

    This class loads the VAD model, keeps its internal state, and returns
    a speech score for one audio frame at a time.
    """

    def __init__(self, model_path: str):
        """Load the ONNX VAD model from disk and prepare its state buffers."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

        input_names = [inp.name for inp in self.session.get_inputs()]
        if "state" in input_names:
            self._version = 5
            state_shape = None
            for inp in self.session.get_inputs():
                if inp.name == "state":
                    state_shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
                    break
            if state_shape is None:
                state_shape = [2, 1, 128]
            self._state = np.zeros(state_shape, dtype=np.float32)
            # 64-sample rolling context window required by Silero V5 ONNX streaming.
            # Without it the model sees a cold start every frame and returns ~0.001.
            self._context = np.zeros((1, 64), dtype=np.float32)
        else:
            self._version = 3
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        """Clear the model state so the next utterance starts fresh."""
        if self._version == 5:
            self._state = np.zeros_like(self._state)
            self._context = np.zeros_like(self._context)
        else:
            self._h = np.zeros_like(self._h)
            self._c = np.zeros_like(self._c)

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """Return a speech score for one audio frame.

        The input frame is padded or trimmed to the exact size expected by the
        model. The returned value is a float score where larger usually means
        "more likely to be speech".
        """
        if len(audio_chunk) == 0:
            return 0.0

        # The model expects a fixed-size frame, so we normalize the length here.
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        elif len(audio_chunk) > 512:
            audio_chunk = audio_chunk[:512]

        if self._version == 5:
            # Silero V5 ONNX streaming expects the current 512 samples preceded
            # by 64 samples of context from the previous frame. We maintain that
            # rolling window in self._context and prepend it here.
            audio_chunk_2d = audio_chunk.reshape(1, -1)
            input_with_context = np.concatenate([self._context, audio_chunk_2d], axis=1)
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
                "input": audio_chunk.reshape(1, -1),
                "sr": np.array([sample_rate], dtype=np.int64),
                "h": self._h,
                "c": self._c,
            }
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c

        return float(out[0][0])


class SileroUtteranceGate:
    """Track one possible spoken utterance across many audio chunks.

    This class stores incoming audio, checks small frames for speech, keeps a
    running estimate of background noise, and decides when enough silence has
    happened to finalize the utterance.
    """

    def __init__(
        self,
        vad_engine,
        *,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        voice_threshold: float = 0.5,
        silence_timeout_s: float = 0.4,
        min_utterance_bytes: int = 8000,
        energy_threshold: float = 0.03,
        energy_ratio: float = 2.5,
    ):
        """Create a new utterance gate with VAD and silence settings.

        Args:
            vad_engine: Object with an `is_speech()` method, or `None`.
            sample_rate: Audio sample rate in Hz.
            frame_samples: Number of samples to analyze at one time.
            voice_threshold: Minimum model score that counts as speech.
            silence_timeout_s: How long silence must last before finalize.
            min_utterance_bytes: Minimum saved audio before finalize is allowed.
            energy_threshold: Fixed minimum RMS energy threshold.
            energy_ratio: Multiplier used with the learned noise floor.
        """
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

    def reset(self):
        """Clear all saved audio and speech state for a new utterance."""
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
        """Start the "finalize soon" timer when the caller wants to stop."""
        self._finalize_armed = True
        self._finalize_time = now

    def has_speech_started(self) -> bool:
        """Return True after the gate has seen speech in this utterance."""
        return self._speech_started

    def push(
            self, 
            pcm16_bytes: bytes, 
            now: float, 
            analysis_pcm16_bytes : bytes | None = None
        ) -> bool:
        """Add new PCM16 audio and check whether it contains speech.

        This method:
        - saves the incoming audio into the utterance buffer
        - splits analysis audio into fixed-size frames
        - runs VAD and energy checks on each frame
        - updates speech timing and noise-floor state

        Returns:
            True if this call detected speech in at least one frame.
            False if this call did not detect speech.
        """
        if not pcm16_bytes:
            return False
        self._raw_analysis_buffer.extend(pcm16_bytes)
        if analysis_pcm16_bytes is not None:
            self._analysis_buffer.extend(analysis_pcm16_bytes)
        else:
            self._analysis_buffer.extend(pcm16_bytes)
        # Keep the full utterance audio so it can be flushed later.
        self._buffer.extend(pcm16_bytes)

        frame_bytes = self.frame_samples * 2
        speech_detected = False
        while len(self._analysis_buffer) >= frame_bytes and len(self._raw_analysis_buffer) >= frame_bytes:
            # Read one analysis frame and remove it from the queue.
            frame_bytes_data = bytes(self._analysis_buffer[:frame_bytes])
            raw_frame_bytes = bytes(self._raw_analysis_buffer[:frame_bytes])
            del self._analysis_buffer[:frame_bytes]
            del self._raw_analysis_buffer[:frame_bytes]

            # Convert PCM16 bytes into normalized float audio for the model.
            audio = np.frombuffer(frame_bytes_data, dtype=np.int16).astype(np.float32) / 32768.0
            raw_audio_for_energy_detection = (
                np.frombuffer(raw_frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            score = 1.0 if self.vad_engine is None else self.vad_engine.is_speech(audio, sample_rate=self.sample_rate)
            frame_rms = float(np.sqrt(np.mean(raw_audio_for_energy_detection ** 2)))
            self._last_energy = frame_rms
            self._last_score = float(score)
            if score > self._max_score:
                self._max_score = float(score)

            # Build an energy threshold that can adapt to room noise.
            dynamic_threshold = max(self.energy_threshold, self._noise_floor * self.energy_ratio)
            self._last_dynamic_threshold = dynamic_threshold
            speech_by_energy = frame_rms >= dynamic_threshold

            if score > self.voice_threshold or speech_by_energy:
                # Any positive speech frame moves the utterance into speech state.
                speech_detected = True
                self._speech_started = True
                self._last_voice_time = now
            else:
                # Frames that are not speech help us slowly learn background noise.
                if self._noise_floor == 0.0:
                    self._noise_floor = frame_rms
                else:
                    self._noise_floor = (0.95 * self._noise_floor) + (0.05 * frame_rms)

        return speech_detected

    def should_finalize(self, now: float) -> bool:
        """Return True when the utterance is ready to be closed.

        There are two cases:
        - speech was detected earlier, and now silence has lasted long enough
        - no speech was detected, but the caller armed finalize and waited long enough
        """
        if self._speech_started:
            # Do not finalize very tiny audio fragments after speech starts.
            if len(self._buffer) < self.min_utterance_bytes:
                return False
            return (now - self._last_voice_time) >= self.silence_timeout_s

        return (
            self._finalize_armed
            and (now - self._finalize_time) >= self.silence_timeout_s
        )

    def silence_elapsed(self, now: float) -> float:
        """Return how many seconds have passed since the last speech frame."""
        return now - self._last_voice_time

    def finalize_elapsed(self, now: float) -> float:
        """Return how many seconds have passed since finalize was armed."""
        return now - self._finalize_time

    def last_score(self) -> float:
        """Return the most recent VAD model score."""
        return self._last_score

    def max_score(self) -> float:
        """Return the highest VAD score seen in the current utterance."""
        return self._max_score

    def last_energy(self) -> float:
        """Return the RMS energy of the most recent analyzed frame."""
        return self._last_energy

    def last_dynamic_threshold(self) -> float:
        """Return the current adaptive energy threshold."""
        return self._last_dynamic_threshold

    def flush(self) -> bytes:
        """Return the saved utterance audio and reset all gate state."""
        audio = bytes(self._buffer)
        self.reset()
        return audio
