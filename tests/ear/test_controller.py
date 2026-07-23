"""Unit tests for src/audio/ear_runtime/controller.py (Ear controller class)."""

from unittest.mock import MagicMock, patch
import pytest

from src.audio.ear_runtime.controller import Ear


class FakePyAudioInstance:
    def get_device_count(self):
        return 1

    def get_device_info_by_index(self, index):
        return {"index": 0, "name": "Built-in Microphone", "maxInputChannels": 2, "defaultSampleRate": 44100.0}

    def get_default_input_device_info(self):
        return {"index": 0, "name": "Built-in Microphone"}

    def terminate(self):
        pass


def test_ear_controller_initialization():
    fake_pa = FakePyAudioInstance()
    ear = Ear(pyaudio_lib=fake_pa, input_device_index=0)
    assert ear.input_device_index == 0
    assert ear.active_mic_name == "Built-in Microphone"
    assert ear.is_recording is False


def test_ear_tracks_current_model():
    fake_pa = FakePyAudioInstance()
    ear = Ear(pyaudio_lib=fake_pa, input_device_index=0)
    assert ear.current_model == "parakeet-tdt-0.6b-v3"
    ear.current_model = "parakeet-tdt-0.6b-v2"
    assert ear.current_model == "parakeet-tdt-0.6b-v2"
