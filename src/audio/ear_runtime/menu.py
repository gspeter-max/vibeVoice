"""Terminal menu and self-test helpers for the Ear runtime."""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from src import log
from src.ipc.client import send_message_to_brain
from src.ipc.protocol import format_switch_model_message
from src.utils.settings import settings


def send_switch_command(model_name, ear_instance=None):
    """Send a model-switch command to Brain and keep Ear model state aligned."""

    log.info(f"\n🔄 Switching Brain to use: {model_name}...\n")
    if ear_instance:
        ear_instance.current_model = model_name

    sent = send_message_to_brain(
        format_switch_model_message(model_name),
        socket_path=settings.socket_path,
    )
    if not sent:
        log.info("\n❌ Failed to send switch command\n")


def run_self_test(sample_rate: int = settings.rate):
    """Send one second of synthetic audio to Brain to test the input path."""

    log.info("\n🧪 Running SELF-TEST (synthetic audio)...\n")
    duration_seconds = 1.0
    frequency_hz = 440.0
    time_axis = np.linspace(
        0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    audio_data = (np.sin(2 * np.pi * frequency_hz * time_axis) * 32767).astype(
        np.int16
    ).tobytes()

    max_retries = 3
    retry_delay_seconds = 1
    for attempt_index in range(max_retries):
        if not os.path.exists(settings.socket_path):
            if attempt_index < max_retries - 1:
                log.info(
                    f"\r⏳ Socket not ready, retrying in {retry_delay_seconds}s... "
                    f"(attempt {attempt_index + 1}/{max_retries})\n"
                )
                time.sleep(retry_delay_seconds)
                continue
            log.info(f"\r❌ Self-test failed: Socket not found at {settings.socket_path}\n")
            log.info("   Is Brain running? Check this terminal for Brain output.\n")
            return

        if send_message_to_brain(audio_data, socket_path=settings.socket_path):
            log.info("\r✅ Self-test audio sent to Brain\n")
            return

        if attempt_index < max_retries - 1:
            log.info(
                f"\r⏳ Brain busy, retrying in {retry_delay_seconds}s... "
                f"(attempt {attempt_index + 1}/{max_retries})\n"
            )
            time.sleep(retry_delay_seconds)
        else:
            log.info("\r❌ Self-test failed: Brain not accepting connections\n")
            log.info("   Brain might be loading model. Check this terminal for Brain output.\n")


class TerminalMenu(threading.Thread):
    """Background terminal input loop for model switching and self-test actions."""

    def __init__(self, ear_instance=None):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()
        self.ear = ear_instance

    def run(self):
        if not sys.stdin.isatty():
            return

        old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    pressed_key = sys.stdin.read(1)
                    if pressed_key in "12345":
                        choice_index = int(pressed_key) - 1
                        active_models = settings.active_stt_models
                        if choice_index < len(active_models):
                            send_switch_command(
                                active_models[choice_index],
                                self.ear,
                            )
                    elif pressed_key.lower() == "t":
                        threading.Thread(target=run_self_test, daemon=True).start()
                    elif pressed_key == "\x03":
                        os.kill(os.getpid(), 2)
                        break
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old_settings)

    def stop(self):
        """Request the background menu thread to stop."""

        self._stop.set()
