"""Silero VAD helpers for speech-boundary detection and utterance buffering."""

from __future__ import annotations

import numpy as np


class SileroVAD:
    def __init__(self, model_path: str):
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
        else:
            self._version = 3
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        if self._version == 5:
            self._state = np.zeros_like(self._state)
        else:
            self._h = np.zeros_like(self._h)
            self._c = np.zeros_like(self._c)

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        if len(audio_chunk) == 0:
            return 0.0

        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        elif len(audio_chunk) > 512:
            audio_chunk = audio_chunk[:512]

        if self._version == 5:
            ort_inputs = {
                "input": audio_chunk.reshape(1, -1),
                "sr": np.array([sample_rate], dtype=np.int64),
                "state": self._state,
            }
            out, new_state = self.session.run(None, ort_inputs)
            self._state = new_state
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
        self._buffer.clear()
        self._analysis_buffer.clear()
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
        self._finalize_armed = True
        self._finalize_time = now

    def has_speech_started(self) -> bool:
        return self._speech_started

    def push(self, pcm16_bytes: bytes, now: float) -> bool:
        if not pcm16_bytes:
            return False

        self._buffer.extend(pcm16_bytes)
        self._analysis_buffer.extend(pcm16_bytes)

        frame_bytes = self.frame_samples * 2
        speech_detected = False
        while len(self._analysis_buffer) >= frame_bytes:
            frame_bytes_data = bytes(self._analysis_buffer[:frame_bytes])
            del self._analysis_buffer[:frame_bytes]

            audio = np.frombuffer(frame_bytes_data, dtype=np.int16).astype(np.float32) / 32768.0
            score = 1.0 if self.vad_engine is None else self.vad_engine.is_speech(audio, sample_rate=self.sample_rate)
            frame_rms = float(np.sqrt(np.mean(audio ** 2)))
            self._last_energy = frame_rms
            self._last_score = float(score)
            if score > self._max_score:
                self._max_score = float(score)
            dynamic_threshold = max(self.energy_threshold, self._noise_floor * self.energy_ratio)
            self._last_dynamic_threshold = dynamic_threshold
            speech_by_energy = frame_rms >= dynamic_threshold

            if score > self.voice_threshold or speech_by_energy:
                speech_detected = True
                self._speech_started = True
                self._last_voice_time = now
            else:
                if self._noise_floor == 0.0:
                    self._noise_floor = frame_rms
                else:
                    self._noise_floor = (0.95 * self._noise_floor) + (0.05 * frame_rms)

        return speech_detected

    def should_finalize(self, now: float) -> bool:
        if self._speech_started:
            if len(self._buffer) < self.min_utterance_bytes:
                return False
            return (now - self._last_voice_time) >= self.silence_timeout_s

        return (
            self._finalize_armed
            and (now - self._finalize_time) >= self.silence_timeout_s
        )

    def silence_elapsed(self, now: float) -> float:
        return now - self._last_voice_time

    def finalize_elapsed(self, now: float) -> float:
        return now - self._finalize_time

    def last_score(self) -> float:
        return self._last_score

    def max_score(self) -> float:
        return self._max_score

    def last_energy(self) -> float:
        return self._last_energy

    def last_dynamic_threshold(self) -> float:
        return self._last_dynamic_threshold

    def flush(self) -> bytes:
        audio = bytes(self._buffer)
        self.reset()
        return audio
