"""Runtime bootstrap entry point for the Ear process."""

from __future__ import annotations

import os
import socket
import sys
import termios
import time

import pyaudio

from src import log
from src.audio.ear_runtime.controller import Ear, TerminalMenu
from src.audio.ear_runtime.devices import select_mic
from src.audio.ear_runtime.recording import start_recording_state
from src.input.hotkeys import InputTrigger
from src.ipc.client import open_checked_raw_audio_stream_to_brain
from src.ui.hud_client import start_hud_command_thread, start_volume_sender_thread
from src.utils.settings import settings

def start_ear():
    """Start the Ear runtime using the controller class."""

    env_mic_index = os.environ.get("VIBEVOICE_MIC_INDEX")
    if env_mic_index is not None:
        try:
            selected_mic_index = int(env_mic_index)
            log.info(f"[Ear] Using microphone index {selected_mic_index} from .env")
        except ValueError:
            temporary_pyaudio = pyaudio.PyAudio()
            selected_mic_index = select_mic(temporary_pyaudio)
            temporary_pyaudio.terminate()
    else:
        temporary_pyaudio = pyaudio.PyAudio()
        selected_mic_index = select_mic(temporary_pyaudio)
        temporary_pyaudio.terminate()

    ear = Ear(input_device_index=selected_mic_index)

    menu = TerminalMenu(ear_instance=ear)
    menu.start()

    def _start_recording_wrapper(from_hold: bool):
        if settings.is_no_streaming_mode:
            raw_stream_socket = open_checked_raw_audio_stream_to_brain(
                timeout_seconds=5.0,
                socket_path=settings.socket_path,
                socket_factory=socket.socket,
            )
            if raw_stream_socket is None:
                return
            with ear._brain_sock_lock:
                ear._brain_sock = raw_stream_socket

        start_recording_state(ear, from_hold=from_hold)
        ear._cmd_press_time = time.time()
        start_hud_command_thread("listen", socket_factory=socket.socket)
        start_volume_sender_thread(
            ear,
            volume_port=settings.vol_port,
            socket_factory=socket.socket,
        )

    def _stop_recording_wrapper(stop_session: bool):
        ear._stop_and_send(stop_session=stop_session)
        ear._toggle_active = False

    def _toggle_recording_wrapper():
        ear._toggle_active = True
        log.info("\r\n⏸️  Toggle mode — tap Right CMD again to stop")
        _start_recording_wrapper(from_hold=False)

    input_trigger = InputTrigger(
        on_start_recording=_start_recording_wrapper,
        on_stop_recording=_stop_recording_wrapper,
        on_toggle_recording=_toggle_recording_wrapper,
    )
    input_trigger.start_listening()

    log.info("[Ear] 🖱️  Mouse listener started - Hold RIGHT button for 1s to record")

    backend_label = {
        "parakeet": "Parakeet TDT v3",
        "nemotron": "Nemotron",
    }.get(settings.backend, settings.backend)

    log.info("─" * 60)
    log.info(f"🎙️  VIBEVOICE PRO | {backend_label} | {ear.active_mic_name}")
    log.info("Hotkey: RIGHT CMD (hold) | Mouse: RIGHT BUTTON (hold)")
    log.info("─" * 60)
    log.info("Ready. Press hotkey to record.")

    try:
        ear.record_loop(input_trigger=input_trigger)
    except KeyboardInterrupt:
        log.info("\r\n\nShutting down Ear...")
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
