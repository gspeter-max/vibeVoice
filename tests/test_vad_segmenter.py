from src.vad_segmenter import SileroUtteranceGate


class FakeVAD:
    def __init__(self, scores):
        self._scores = iter(scores)

    def is_speech(self, _audio_chunk, sample_rate=16000):
        return next(self._scores)


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
