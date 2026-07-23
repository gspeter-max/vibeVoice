"""Unit tests for src/audio/ear_runtime/devices.py (Microphone device selection)."""

from unittest.mock import MagicMock, patch
import pytest

from src.audio.ear_runtime.devices import (
    default_input_device_index,
    resolve_input_device_index,
    select_mic,
)


class FakePyAudioForDevices:
    def get_device_count(self):
        return 3

    def get_device_info_by_index(self, index):
        devices = {
            0: {"index": 0, "name": "Built-in Microphone", "maxInputChannels": 2, "defaultSampleRate": 44100.0},
            1: {"index": 1, "name": "USB Headset Mic", "maxInputChannels": 1, "defaultSampleRate": 48000.0},
            2: {"index": 2, "name": "Output Speaker", "maxInputChannels": 0, "defaultSampleRate": 44100.0},
        }
        return devices[index]

    def get_default_input_device_info(self):
        return {"index": 0, "name": "Built-in Microphone"}


def test_default_input_device_index_returns_default():
    assert default_input_device_index(FakePyAudioForDevices()) == 0


def test_resolve_input_device_index_prefers_explicit_index():
    assert resolve_input_device_index(FakePyAudioForDevices(), 1) == 1


def test_resolve_input_device_index_uses_valid_environment_value(monkeypatch):
    monkeypatch.setenv("VIBEVOICE_MIC_INDEX", "1")
    assert resolve_input_device_index(FakePyAudioForDevices(), None) == 1


def test_resolve_input_device_index_falls_back_to_default_when_environment_is_invalid(monkeypatch):
    monkeypatch.setenv("VIBEVOICE_MIC_INDEX", "invalid")
    assert resolve_input_device_index(FakePyAudioForDevices(), None) == 0


def test_select_mic_returns_selected_device_index(monkeypatch):
    fake_pa = FakePyAudioForDevices()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    selected = select_mic(fake_pa)
    assert selected == 1


def test_select_mic_returns_default_device_index_when_choice_is_blank(monkeypatch):
    fake_pa = FakePyAudioForDevices()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    selected = select_mic(fake_pa)
    assert selected == 0
