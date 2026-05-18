"""Microphone and input-device helpers for the Ear runtime."""

from __future__ import annotations

import os

from src import log


def default_input_device_index(pyaudio_instance) -> int:
    """Return the active system default microphone index.

    The Ear runtime resolves the microphone choice once during startup.
    Keeping the default-device lookup here prevents controller.py from
    re-owning low-level input-device logic.
    """

    return pyaudio_instance.get_default_input_device_info().get("index")


def resolve_input_device_index(pyaudio_instance, input_device_index) -> int:
    """Resolve the microphone index from explicit input, env var, or default.

    Resolution order stays intentionally simple and literal:
    1. An explicit constructor argument wins.
    2. `VIBEVOICE_MIC_INDEX` is used when it contains a valid integer.
    3. The active system default input device is used as the fallback.
    """

    if input_device_index is not None:
        return input_device_index

    environment_microphone_index = os.environ.get("VIBEVOICE_MIC_INDEX")
    if environment_microphone_index is not None:
        try:
            return int(environment_microphone_index)
        except ValueError:
            pass

    return default_input_device_index(pyaudio_instance)


def select_mic(pyaudio_instance):
    """Let the user choose one input device from the available microphones.

    The function shows only devices that can record input audio. The visible
    numbering is contiguous, even when the underlying device indexes are not.
    Pressing Enter keeps the system default input device.
    """

    log.info("\n🎤  SELECT YOUR MICROPHONE:")
    log.info("─" * 30)
    selectable_device_indexes: list[int] = []
    default_device_info = pyaudio_instance.get_default_input_device_info()
    default_device_index = default_device_info.get("index")
    default_choice_index = 0

    for device_index in range(pyaudio_instance.get_device_count()):
        device_info = pyaudio_instance.get_device_info_by_index(device_index)
        if device_info.get("maxInputChannels") > 0:
            device_name = device_info.get("name")
            visible_choice_index = len(selectable_device_indexes)
            is_default_device = " (DEFAULT)" if device_index == default_device_index else ""
            log.info(f" [{visible_choice_index}] {device_name}{is_default_device}")
            selectable_device_indexes.append(device_index)
            if device_index == default_device_index:
                default_choice_index = visible_choice_index

    log.info("─" * 30)
    while True:
        try:
            chosen_text = input(
                f"Select Mic Index [default {default_choice_index}]: "
            ).strip()
            if not chosen_text:
                return default_device_index

            visible_choice_index = int(chosen_text)
            if 0 <= visible_choice_index < len(selectable_device_indexes):
                return selectable_device_indexes[visible_choice_index]
            log.info("❌ Invalid index.")
        except ValueError:
            log.info("❌ Please enter a valid number.")
