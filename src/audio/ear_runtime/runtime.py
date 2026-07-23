"""Runtime bootstrap entry point for the Ear process."""

from __future__ import annotations

import os
import socket
import sys
import termios
import time

import pyaudio

from src import log
from src.audio.ear_runtime.controller import Ear
from src.audio.ear_runtime.devices import select_mic
from src.audio.ear_runtime.menu import TerminalMenu
from src.audio.ear_runtime.recording import LogState, start_recording_state
from src.audio.vad_segmenter import SileroUtteranceGate, SileroVAD
from src.input.hotkeys import InputTrigger, RecordingCallbacks
from src.streaming.capture_session import CaptureSession
from src.ui.hud_client import change_ui_status, ui_wave_input
from src.utils.settings import settings


def start_ear():
    """Start the Ear runtime using the controller class."""

    env_mic_index = os.environ.get("VIBEVOICE_MIC_INDEX")
    if env_mic_index is not None:
        try:
            selected_mic_index = int(env_mic_index)
            log.debug(f"[Ear] Using microphone index {selected_mic_index} from .env")
        except ValueError:
            temporary_pyaudio = pyaudio.PyAudio()
            selected_mic_index = select_mic(temporary_pyaudio)
            temporary_pyaudio.terminate()
    else:
        temporary_pyaudio = pyaudio.PyAudio()
        selected_mic_index = select_mic(temporary_pyaudio)
        temporary_pyaudio.terminate()

    ear = Ear(input_device_index=selected_mic_index)

    session = CaptureSession(
        sample_rate=settings.rate,
        overlap_seconds=settings.overlap_seconds,
    )

    try:
        vad_engine = SileroVAD(settings.vad_model_path)
        log.info("[system] VAD initialized")
    except Exception as e:
        vad_engine = None
        log.warning(f"[Ear] VAD load failed: {e}")

    utr_gate = SileroUtteranceGate(
        vad_engine,
        voice_threshold=settings.vad_score_threshold,
        silence_timeout_s=settings.silence_timeout_seconds,
        energy_threshold=settings.vad_energy_threshold,
        energy_ratio=settings.vad_energy_ratio,
    )

    log_state = LogState()
    menu = TerminalMenu(ear_instance=ear)
    menu.start()

    def _start_recording_wrapper():
        start_recording_state(ear, session, utr_gate, log_state, ear.telemetry_enabled)
        ear.cmd_press_time = time.time()
        change_ui_status("listen")
        ui_wave_input(ear)

    def _stop_recording_wrapper(stop_session: bool):
        ear.stop_and_send(session, utr_gate, log_state, stop_session=stop_session)
        ear.toggle_active = False

    def _toggle_recording_wrapper():
        ear.toggle_active = True
        log.info("[ear]    toggle mode active")
        _start_recording_wrapper()

    input_trigger = InputTrigger(
        callbacks=RecordingCallbacks(
            on_start=_start_recording_wrapper,
            on_stop=_stop_recording_wrapper,
            on_toggle=_toggle_recording_wrapper,
        )
    )
    input_trigger.start()

    log.info("[ear]    mouse listener started")

    backend_label = {
        "parakeet": "Parakeet TDT v3",
        "nemotron": "Nemotron",
    }.get(settings.backend, settings.backend)

    log.info(f"[ear]    ready using {backend_label} on {ear.active_mic_name}")

    try:
        ear.record_loop(
            input_trigger=input_trigger, utr_gate=utr_gate, session=session, log_state=log_state
        )
    except KeyboardInterrupt:
        log.info("[ear]    shutting down")
    finally:
        menu.stop()
        ear.cleanup()
        if sys.stdin.isatty():
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                termios.tcgetattr(sys.stdin.fileno()),
            )


if __name__ == "__main__":
    start_ear()
