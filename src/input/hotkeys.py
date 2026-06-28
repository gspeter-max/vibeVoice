"""Keyboard and mouse shortcut helpers for starting and stopping recording."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
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


@dataclass
class RecordingCallbacks:
    """Holds all external callbacks that define what happens on recording events."""

    on_start: Callable[[bool], None]
    on_stop: Callable[[bool], None]
    on_toggle: Callable[[], None]


@dataclass
class KeyState:
    """Tracks the internal state of the Right Command key."""

    held: bool = False
    toggle_active: bool = False
    last_release: float = 0.0
    rec_hold: bool = False
    double_tap: float = 0.3
    timer: Optional[threading.Timer] = field(default=None)


@dataclass
class MouseState:
    """Tracks the internal state of the right mouse button."""

    start: float = 0.0
    held: bool = False
    rec_hold: bool = False


class InputTrigger:
    """Handles keyboard and mouse inputs for speech recording control.

    Attributes:
        callbacks: External recording event callbacks.
        key: Right Command key runtime state.
        mouse: Right mouse button runtime state.
    """

    def __init__(
        self,
        callbacks: RecordingCallbacks,
        hold_threshold_seconds: float = settings.recording_button_hold_threshold,
    ) -> None:
        """Initialize the input trigger with callbacks and timing thresholds.

        Args:
            callbacks: Grouped recording event callbacks.
            hold_threshold_seconds: Duration in seconds before a keypress counts as a hold.
        """
        self.callbacks = callbacks
        self.key = KeyState()
        self.mouse = MouseState()
        self._hold_seconds = hold_threshold_seconds
        self._lock = threading.Lock()
        self._kb_listener: Optional[keyboard.Listener] = None
        self._mouse_listener: Optional[mouse.Listener] = None

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
            if self.key.timer:
                self.key.timer.cancel()
                self.key.timer = None

    def _trigger_hold(self) -> None:
        """Invoke start callback if right command key is still held."""
        with self._lock:
            if self.key.held and not self.key.toggle_active:
                self.key.rec_hold = True
                self.callbacks.on_start(from_hold=True)

    def _key_press(self, key: Any, current_time: float) -> None:
        """Handle OS key press events.

        Args:
            key: Keyboard key object.
            current_time: Press event timestamp.
        """
        if not _is_rcmd(key):
            return

        with self._lock:
            if self.key.held:
                return

            self.key.held = True

            if self.key.toggle_active:
                self.key.toggle_active = False
                self.callbacks.on_stop(stop_session=True)
                return

            time_since_last_release = current_time - self.key.last_release

            if time_since_last_release <= self.key.double_tap:
                if self.key.timer:
                    self.key.timer.cancel()
                    self.key.timer = None

                self.key.toggle_active = True
                self.callbacks.on_toggle()
            else:
                if self.key.timer:
                    self.key.timer.cancel()

                self.key.timer = threading.Timer(
                    self._hold_seconds,
                    self._trigger_hold,
                )
                self.key.timer.start()

    def _key_release(self, key: Any, current_time: float) -> None:
        """Handle OS key release events.

        Args:
            key: Keyboard key object.
            current_time: Release event timestamp.
        """
        if not _is_rcmd(key):
            return

        with self._lock:
            self.key.held = False
            self.key.last_release = current_time

            if self.key.timer:
                self.key.timer.cancel()
                self.key.timer = None

            if self.key.toggle_active:
                return

            if self.key.rec_hold:
                self.key.rec_hold = False
                self.callbacks.on_stop(stop_session=True)

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
                self.mouse.start = time.time()
                self.mouse.held = True
            else:
                self.mouse.held = False
                if self.mouse.rec_hold:
                    self.callbacks.on_stop(stop_session=True)
                    self.mouse.rec_hold = False

    def check_mouse_hold(self) -> bool:
        """Check if right mouse button has been held beyond the threshold.

        Returns:
            True if recording was triggered, False otherwise.
        """
        with self._lock:
            if self.mouse.held and not self.mouse.rec_hold:
                if time.time() - self.mouse.start >= 1.0:
                    self.mouse.rec_hold = True
                    self.callbacks.on_start(from_hold=True)
                    return True
        return False
