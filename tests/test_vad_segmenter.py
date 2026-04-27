import numpy as np
from src.audio.vad_segmenter import SileroUtteranceGate, SileroVAD


class FakeVAD:
    def __init__(self, scores):
        self._scores = iter(scores)

    def is_speech(self, _audio_chunk, sample_rate=16000):
        return next(self._scores)


class RecordingVAD:
    def __init__(self, score):
        self.score = score
        self.seen_audio_chunks = []

    def is_speech(self, audio_chunk, sample_rate=16000):
        self.seen_audio_chunks.append(audio_chunk.copy())
        return self.score


def test_gate_flushes_only_after_voice_then_silence():
    gate = SileroUtteranceGate(
        vad_engine=FakeVAD([0.95, 0.91]),
        voice_threshold=0.5,
        silence_timeout_s=0.2,
        min_utterance_bytes=16,
        frame_samples=4,
    )

    gate.push(b"\x01\x00" * 4, now=0.0)
    gate.push(b"\x02\x00" * 4, now=0.1)

    assert gate.should_finalize(now=0.2) is False
    assert gate.should_finalize(now=0.31) is True
    assert gate.flush() == (b"\x01\x00" * 4) + (b"\x02\x00" * 4)


def test_gate_can_finalize_after_release_without_speech():
    gate = SileroUtteranceGate(
        vad_engine=FakeVAD([0.0, 0.0]),
        voice_threshold=0.5,
        silence_timeout_s=0.2,
        min_utterance_bytes=16,
        frame_samples=4,
    )

    gate.push(b"\x00\x00" * 4, now=0.0)
    gate.arm_finalize(now=0.0)

    assert gate.should_finalize(now=0.1) is False
    assert gate.should_finalize(now=0.2) is True


def test_gate_falls_back_to_energy_when_silero_score_flatlines():
    gate = SileroUtteranceGate(
        vad_engine=FakeVAD([0.001, 0.001, 0.001, 0.001]),
        voice_threshold=0.5,
        silence_timeout_s=0.2,
        min_utterance_bytes=16,
        frame_samples=4,
    )

    loud = (b"\x00\x10" * 4)
    quiet = (b"\x00\x01" * 4)

    assert gate.push(loud, now=0.0) is True
    assert gate.has_speech_started() is True
    assert gate.push(quiet, now=0.1) is False
    assert gate.should_finalize(now=0.31) is True


def test_gate_keeps_raw_audio_but_uses_analysis_audio_for_detection():
    vad_engine = RecordingVAD(score=0.95)
    gate = SileroUtteranceGate(
        vad_engine=vad_engine,
        voice_threshold=0.5,
        silence_timeout_s=0.2,
        min_utterance_bytes=8,
        frame_samples=4,
    )

    raw_pcm16_bytes = b"\x01\x00" * 4
    analysis_pcm16_bytes = b"\x02\x00" * 4

    assert gate.push(
        raw_pcm16_bytes,
        now=0.0,
        analysis_pcm16_bytes=analysis_pcm16_bytes,
    ) is True
    assert gate.flush() == raw_pcm16_bytes
    assert vad_engine.seen_audio_chunks[0].tolist() == [2 / 32768.0] * 4


def test_gate_uses_current_raw_frame_for_energy_fallback_not_first_frame_forever():
    gate = SileroUtteranceGate(
        vad_engine=FakeVAD([0.001, 0.001]),
        voice_threshold=0.5,
        silence_timeout_s=0.2,
        min_utterance_bytes=16,
        frame_samples=4,
    )

    loud_raw_pcm16_bytes = b"\x00\x10" * 4
    quiet_raw_pcm16_bytes = b"\x00\x01" * 4
    boosted_quiet_analysis_pcm16_bytes = b"\x00\x06" * 4

    assert gate.push(loud_raw_pcm16_bytes, now=0.0) is True
    assert gate.push(
        quiet_raw_pcm16_bytes,
        now=0.1,
        analysis_pcm16_bytes=boosted_quiet_analysis_pcm16_bytes,
    ) is False


def test_silero_v5_wrapper_prepends_64_sample_context_for_onnx_input(monkeypatch):
    seen_inputs = []

    class FakeInput:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return [
                FakeInput("input", [None, None]),
                FakeInput("state", [2, None, 128]),
                FakeInput("sr", []),
            ]

        def run(self, _output_names, ort_inputs):
            seen_inputs.append(ort_inputs["input"].copy())
            return np.array([[0.9]], dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)

    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)

    vad = SileroVAD("fake.onnx")
    chunk = np.ones(512, dtype=np.float32) * 0.25
    vad.is_speech(chunk, sample_rate=16000)

    assert seen_inputs[0].shape == (1, 576)
    assert np.allclose(seen_inputs[0][0, :64], 0.0)
    assert np.allclose(seen_inputs[0][0, 64:], 0.25)


def test_silero_v5_wrapper_updates_context_after_each_call(monkeypatch):
    seen_inputs = []

    class FakeInput:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return [
                FakeInput("input", [None, None]),
                FakeInput("state", [2, None, 128]),
                FakeInput("sr", []),
            ]

        def run(self, _output_names, ort_inputs):
            seen_inputs.append(ort_inputs["input"].copy())
            return np.array([[0.9]], dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)

    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)

    vad = SileroVAD("fake.onnx")
    first = np.ones(512, dtype=np.float32) * 0.10
    second = np.ones(512, dtype=np.float32) * 0.20

    vad.is_speech(first, sample_rate=16000)
    vad.is_speech(second, sample_rate=16000)

    assert np.allclose(seen_inputs[1][0, :64], 0.10)
    assert np.allclose(seen_inputs[1][0, 64:], 0.20)


def test_silero_v5_wrapper_reset_clears_context(monkeypatch):
    seen_inputs = []

    class FakeInput:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return [
                FakeInput("input", [None, None]),
                FakeInput("state", [2, None, 128]),
                FakeInput("sr", []),
            ]

        def run(self, _output_names, ort_inputs):
            seen_inputs.append(ort_inputs["input"].copy())
            return np.array([[0.9]], dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)

    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)

    vad = SileroVAD("fake.onnx")
    vad.is_speech(np.ones(512, dtype=np.float32) * 0.10, sample_rate=16000)
    vad.reset()
    vad.is_speech(np.ones(512, dtype=np.float32) * 0.20, sample_rate=16000)

    assert np.allclose(seen_inputs[1][0, :64], 0.0)
