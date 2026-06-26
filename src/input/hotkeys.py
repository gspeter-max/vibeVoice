"""Keyboard and mouse shortcut helpers for starting and stopping recording."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from pynput import keyboard, mouse

from src.utils.settings import settings


def _is_rcmd(key: Any) -> bool:
    """Check if the pressed key matches Right Command."""
    return (
        key == keyboard.Key.cmd_r
        or getattr(key, "name", None) == "cmd_r"
        or getattr(key, "vk", None) == settings.right_cmd_vk
    )


class InputTrigger:
    """Handles keyboard and mouse inputs for speech recording control.

    Attributes:
        _on_start: Wrapper function called to start recording.
        _on_stop: Wrapper function called to stop recording.
        _on_toggle: Wrapper function called to toggle recording mode.
        _lock: Threading lock for thread-safe state mutations.
        _cmd_held: True if the right command key is currently held.
        _toggle_active: True if toggle mode is active.
        _last_release: Timestamp of the last right command release.
        _double_tap: Time threshold in seconds for detecting double taps.
        _hold_seconds: Time threshold in seconds to trigger hold recording.
        _timer: Timer thread to delay hold recording trigger.
        _rec_hold: True if recording was triggered by holding right command.
        _mouse_start: Timestamp when right mouse button was pressed.
        is_mouse_held: True if right mouse button is held down.
        _rec_mouse_hold: True if recording was triggered by holding right mouse button.
        _kb_listener: pynput Keyboard listener thread instance.
        _mouse_listener: pynput Mouse listener thread instance.
    """

    _on_start: Callable[[bool], None]
    _on_stop: Callable[[bool], None]
    _on_toggle: Callable[[], None]
    _lock: threading.Lock
    _cmd_held: bool
    _toggle_active: bool
    _last_release: float
    _double_tap: float
    _hold_seconds: float
    _timer: Optional[threading.Timer]
    _rec_hold: bool
    _mouse_start: float
    is_mouse_held: bool
    _rec_mouse_hold: bool
    _kb_listener: Optional[keyboard.Listener]
    _mouse_listener: Optional[mouse.Listener]

    def __init__(
        self,
        on_start_recording: Callable[[bool], None],
        on_stop_recording: Callable[[bool], None],
        on_toggle_recording: Callable[[], None],
        hold_threshold_seconds: float = settings.recording_button_hold_threshold,
    ) -> None:
        """Initialize the input trigger with callbacks and timing thresholds.

        Args:
            on_start_recording: Callback to initiate recording.
            on_stop_recording: Callback to terminate recording.
            on_toggle_recording: Callback to toggle recording state.
            hold_threshold_seconds: Duration before trigger counts as hold.
        """
        self._on_start = on_start_recording
        self._on_stop = on_stop_recording
        self._on_toggle = on_toggle_recording
        self._lock = threading.Lock()
        self._cmd_held = False
        self._toggle_active = False
        self._last_release = 0.0
        self._double_tap = 0.3
        self._hold_seconds = hold_threshold_seconds
        self._timer = None
        self._rec_hold = False
        self._mouse_start = 0.0
        self.is_mouse_held = False
        self._rec_mouse_hold = False
        self._kb_listener = None
        self._mouse_listener = None

    def start(self) -> None:
        """Initialize and start keyboard and mouse background listeners."""
        with self._lock:
            if self._kb_listener is None:
                self._kb_listener = keyboard.Listener(
                    on_press=lambda k: self._key_press(k, time.time()),
                    on_release=lambda k: self._key_release(k, time.time()),
                )
                if hasattr(self._kb_listener, "start"):
                    self._kb_listener.start()

            if self._mouse_listener is None:
                self._mouse_listener = mouse.Listener(on_click=self._mouse_click)
                if hasattr(self._mouse_listener, "start"):
                    self._mouse_listener.start()

    def stop(self) -> None:
        """Stop background listeners and cancel active timers."""
        with self._lock:
            if self._kb_listener:
                if hasattr(self._kb_listener, "stop"):
                    self._kb_listener.stop()
                self._kb_listener = None
            if self._mouse_listener:
                if hasattr(self._mouse_listener, "stop"):
                    self._mouse_listener.stop()
                self._mouse_listener = None
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def _trigger_hold(self) -> None:
        """Invoke start wrapper if right command key is still held."""
        with self._lock:
            if self._cmd_held and not self._toggle_active:
                self._rec_hold = True
                self._on_start(from_hold=True)

    def _key_press(self, key: Any, current_time: float) -> None:
        """Handle OS key press events.

        Args:
            key: Keyboard key object.
            current_time: Release event timestamp.
        """
        if not _is_rcmd(key):
            return

        with self._lock:
            if self._cmd_held:
                return

            self._cmd_held = True

            if self._toggle_active:
                self._toggle_active = False
                self._on_stop(stop_session=True)
                return

            time_since_last_release = current_time - self._last_release

            if time_since_last_release <= self._double_tap:
                if self._timer:
                    self._timer.cancel()
                    self._timer = None

                self._toggle_active = True
                self._on_toggle()
            else:
                if self._timer:
                    self._timer.cancel()

                self._timer = threading.Timer(
                    self._hold_seconds,
                    self._trigger_hold,
                )
                self._timer.start()

    def _key_release(self, key: Any, current_time: float) -> None:
        """Handle OS key release events.

        Args:
            key: Keyboard key object.
            current_time: Release event timestamp.
        """
        if not _is_rcmd(key):
            return

        with self._lock:
            self._cmd_held = False
            self._last_release = current_time

            if self._timer:
                self._timer.cancel()
                self._timer = None

            if self._toggle_active:
                return

            if self._rec_hold:
                self._rec_hold = False
                self._on_stop(stop_session=True)

    def _mouse_click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        """Handle mouse click and release events.

        Args:
            x: Cursor x coordinate.
            y: Cursor y coordinate.
            button: Clicked mouse button.
            pressed: True if pressed, False if released.
        """
        if button != getattr(mouse.Button, "right", None):
            return

        with self._lock:
            if pressed:
                self._mouse_start = time.time()
                self.is_mouse_held = True
            else:
                self.is_mouse_held = False
                if self._rec_mouse_hold:
                    self._on_stop(stop_session=True)
                    self._rec_mouse_hold = False

    def check_mouse_hold(self) -> bool:
        """Check if mouse right button has been held beyond threshold.

        Returns:
            True if recording was triggered, False otherwise.
        """
        with self._lock:
            if self.is_mouse_held and not self._rec_mouse_hold:
                if time.time() - self._mouse_start >= 1.0:
                    self._rec_mouse_hold = True
                    self._on_start(from_hold=True)
                    return True
        return False
