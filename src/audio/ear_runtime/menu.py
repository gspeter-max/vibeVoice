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
from src.ipc.client import send_message
from src.ipc.protocol import fmt_switch
from src.utils.settings import settings


def send_switch_command(model_name, ear_instance=None):
    """Send a model-switch command to Brain and keep Ear model state aligned.

    Args:
        model_name: The name of the speech-to-text model to switch to.
        ear_instance: Optional Ear runtime instance to sync the local model state.
    """
    log.debug(f"\n🔄 Switching Brain to use: {model_name}...\n")
    if ear_instance:
        ear_instance.current_model = model_name

    sent = send_message(
        fmt_switch(model_name),
    )
    if not sent:
        log.debug("\n❌ Failed to send switch command\n")


def self_test(sample_rate: int = settings.rate):
    """Send one second of synthetic audio to Brain to test the input path.

    Generates a 440Hz sine wave (1.0 second duration) to verify that the
    IPC communication channel and Brain inference are receiving data properly.

    Args:
        sample_rate: Audio sample rate in Hz. Defaults to settings.rate.
    """
    log.info("[system] running self-test")
    duration_seconds = 1.0
    frequency_hz = 440.0
    time_axis = np.linspace(
        0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    audio_data = (
        (np.sin(2 * np.pi * frequency_hz * time_axis) * 32767)
        .astype(np.int16)
        .tobytes()
    )

    max_retries = 3
    retry_delay_seconds = 1
    for attempt_index in range(max_retries):
        if not os.path.exists(settings.ear_to_brain_socket_path):
            if attempt_index < max_retries - 1:
                log.debug(
                    f"\r⏳ Socket not ready, retrying in {retry_delay_seconds}s... "
                    f"(attempt {attempt_index + 1}/{max_retries})\n"
                )
                time.sleep(retry_delay_seconds)
                continue
            log.warning("[system] self-test failed: brain socket not found")
            return

        if send_message(audio_data):
            log.info("[system] self-test audio sent to brain")
            return

        if attempt_index < max_retries - 1:
            log.debug(
                f"\r⏳ Brain busy, retrying in {retry_delay_seconds}s... "
                f"(attempt {attempt_index + 1}/{max_retries})\n"
            )
            time.sleep(retry_delay_seconds)
        else:
            log.warning("[system] self-test failed: brain not responding")


class TerminalMenu(threading.Thread):
    def __init__(self, ear_instance=None):
        """Background terminal input loop for model switching and self-test actions.
        Runs as a daemon thread to monitor standard input for specific keystrokes,
        enabling runtime model switching and self-tests without blocking main execution.

        Args:
            ear_instance: Optional active Ear instance used to track model changes.
        """
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()
        self.ear = ear_instance

    def run(self):
        """Monitor standard input for user commands using raw termios terminal control.

        Saves the current terminal settings, puts the terminal into non-canonical
        (cbreak) mode to intercept raw keypresses, and polls standard input in a loop.
        Restores settings when stopped or interrupted.
        """
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
                        threading.Thread(target=self_test, daemon=True).start()

        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old_settings)

    def stop(self):
        """Request the background menu thread to stop."""
        self._stop.set()
