"""Ear-facing HUD client helpers.

This module contains only the lightweight socket clients that Ear uses to tell
the HUD about state changes and live microphone levels.
"""

from __future__ import annotations

import socket
import threading
import time

from src import log
from src.ipc.client import SocketConfig, send_message
from src.utils.settings import settings
from src.ipc.socket_utils import create_socket


def send_hud_command(
    command_text: str,
    timeout: float = 0.2,
) -> bool:
    """Send one HUD state command over TCP.

    The command vocabulary stays unchanged. This helper only handles the local
    socket send and returns `True` on success or `False` on failure.
    """
    cfg = SocketConfig(address=settings.brain_to_hud_socket_path, timeout=timeout)
    succ = send_message(message_bytes=command_text.encode("utf-8"), cfg=cfg)
    return succ


def change_ui_status(
    command_text: str,
    timeout: float = 0.2,
):
    """Start one daemon thread that sends a single HUD command.

    Ear triggers these fire-and-forget HUD state changes from multiple places.
    Keeping the thread launch here avoids repeating the same small threading
    boilerplate in the runtime controller.
    """

    sender_thread = threading.Thread(
        target=send_hud_command,
        kwargs={
            "command_text": command_text,
            "timeout": timeout,
        },
        daemon=True,
    )
    sender_thread.start()
    return sender_thread


def ui_wave_input(
    ear_state,
    send_interval_seconds: float = 0.04,
):
    """Start the background UDP sender that streams Ear volume information.

    The `ear_state` object must expose `_lock`, `is_recording`, `last_rms`,
    and `last_frequency_bands`, which matches the current Ear runtime fields.
    The helper returns the created thread so tests can join it deterministically.
    """

    def _sender():
        while True:
            with ear_state.lock:
                if not ear_state.is_recording:
                    break
                rms = ear_state.last_rms
                frequency_bands = ear_state.last_frequency_bands

            try:
                message = (
                    f"vol:{rms:.4f},bass:{frequency_bands['bass']:.3f},"
                    f"mid:{frequency_bands['mid']:.3f},treble:{frequency_bands['treble']:.3f}"
                )
                send_hud_command(message)
            except OSError as error:
                log.debug("[Ear] Failed to send volume: %s", error)
            time.sleep(send_interval_seconds)

    sender_thread = threading.Thread(target=_sender, daemon=True)
    sender_thread.start()
    return sender_thread
