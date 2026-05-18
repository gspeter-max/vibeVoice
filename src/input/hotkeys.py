"""Keyboard and mouse shortcut helpers for starting and stopping recording."""

import time
import threading
from typing import Callable

from src.utils.settings import settings

# We try to import pynput, which is a library that listens to the keyboard and mouse.
# If it's not installed or can't load, we create "fallback" classes so the code doesn't crash.
try:
    from pynput import keyboard, mouse
except ImportError:
    # If pynput is missing, we create empty objects that look like pynput
    # so the rest of the code can still run (even if it won't actually hear keys).
    class _FallbackKey:
        cmd_r = "cmd_r"

    class _FallbackListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            """Match pynput's listener API without doing any work."""
            return None

        def stop(self):
            """Match pynput's listener API without doing any work."""
            return None

    class _FallbackKeyboardModule:
        Key = _FallbackKey()
        Listener = _FallbackListener
    keyboard = _FallbackKeyboardModule()

    class _FallbackMouseButton:
        right = "right"
    class _FallbackMouseModule:
        Button = _FallbackMouseButton()
        Listener = _FallbackListener
    mouse = _FallbackMouseModule()

def _is_right_cmd(key) -> bool:
    """Return `True` when the key event represents Right Command."""
    # 1. Standard check
    _key_class = getattr(keyboard, 'Key', None)
    if _key_class and key == getattr(_key_class, 'cmd_r', None):
        return True

    # 2. Name-based check (for some OS versions)
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True

    # 3. Hardware-code check (for macOS specifically)
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == settings.right_cmd_vk:
        return True

    return False

class InputTrigger:
    """
    The 'Brain' for handling keyboard and mouse shortcuts.

    Features:
    - Smart filtering: Ignores noisy "auto-repeat" signals from the OS.
    - Thread Safety: Uses a Lock to prevent crashes when multi-tasking.
    - Delayed Start: Waits 0.3s before triggering Push-to-Talk to prevent accidental flashes.
    - Double Tap Toggle: Tapping twice quickly locks the recording on.
    """
    def __init__(
        self,
        on_start_recording: Callable[[bool], None],
        on_stop_recording: Callable[[bool], None],
        on_toggle_recording: Callable[[], None],
        hold_threshold_seconds: float = settings.recording_button_hold_threshold,
    ):
        # Functions to call
        self._on_start_recording = on_start_recording
        self._on_stop_recording = on_stop_recording
        self._on_toggle_recording = on_toggle_recording

        # Security & State
        self._state_lock = threading.Lock()

        # Keyboard tracking
        self._is_cmd_key_currently_held = False  # Guards against OS auto-repeat noise
        self._is_toggle_mode_active = False

        # Double-tap and delayed hold tracking
        self._last_cmd_release_time = 0.0
        self._double_tap_threshold = 0.3
        self._hold_threshold_seconds = hold_threshold_seconds
        self._delayed_start_timer: threading.Timer | None = None
        self._is_recording_due_to_hold = False

        # Mouse tracking
        self._mouse_press_start_time = 0.0
        self.is_mouse_button_held_down = False
        self._is_recording_due_to_mouse_hold = False

        # Placeholders for listener threads
        self._keyboard_listener_thread = None
        self._mouse_listener_thread = None

    def start_listening(self):
        """Spins up the background threads to start watching for inputs."""
        with self._state_lock:
            if self._keyboard_listener_thread is None:
                self._keyboard_listener_thread = keyboard.Listener(
                    on_press=lambda k: self._handle_key_press(k, time.time()),
                    on_release=lambda k: self._handle_key_release(k, time.time())
                )
                if hasattr(self._keyboard_listener_thread, "start"):
                    self._keyboard_listener_thread.start()

            if self._mouse_listener_thread is None:
                self._mouse_listener_thread = mouse.Listener(
                    on_click=self._handle_mouse_click
                )
                if hasattr(self._mouse_listener_thread, "start"):
                    self._mouse_listener_thread.start()

    def stop_listening(self):
        """Shuts down the background listeners and timers."""
        with self._state_lock:
            if self._keyboard_listener_thread:
                if hasattr(self._keyboard_listener_thread, "stop"):
                    self._keyboard_listener_thread.stop()
                self._keyboard_listener_thread = None
            if self._mouse_listener_thread:
                if hasattr(self._mouse_listener_thread, "stop"):
                    self._mouse_listener_thread.stop()
                self._mouse_listener_thread = None
            if self._delayed_start_timer:
                self._delayed_start_timer.cancel()
                self._delayed_start_timer = None

    def _trigger_hold_recording(self):
        """Called by the background timer 0.3s after the user presses the key."""
        with self._state_lock:
            # If the user is STILL physically holding the key, and we aren't in toggle mode...
            if self._is_cmd_key_currently_held and not self._is_toggle_mode_active:
                self._is_recording_due_to_hold = True
                self._on_start_recording(from_hold=True)

    def _handle_key_press(self, key, current_time: float):
        """Processes a 'Key Down' event from the OS."""
        if not _is_right_cmd(key):
            return

        with self._state_lock:
            # Guard against the "Holding down" repeat signals from the OS
            if self._is_cmd_key_currently_held:
                return

            self._is_cmd_key_currently_held = True

            # If we are already recording in Toggle Mode, a single tap turns it off.
            if self._is_toggle_mode_active:
                self._is_toggle_mode_active = False
                self._on_stop_recording(stop_session=True)
                return

            # Check how long it has been since we last let go of the key
            time_since_last_release = current_time - self._last_cmd_release_time

            if time_since_last_release <= self._double_tap_threshold:
                # It's a double tap!
                if self._delayed_start_timer:
                    self._delayed_start_timer.cancel()
                    self._delayed_start_timer = None

                self._is_toggle_mode_active = True
                self._on_toggle_recording()
            else:
                # It's the first press. Do NOT start recording yet.
                # Start a 0.3s timer. If they hold it that long, it's a push-to-talk.
                if self._delayed_start_timer:
                    self._delayed_start_timer.cancel()

                # Start hold-to-talk only after the configured hold threshold.
                self._delayed_start_timer = threading.Timer(
                    self._hold_threshold_seconds,
                    self._trigger_hold_recording,
                )
                self._delayed_start_timer.start()

    def _handle_key_release(self, key, current_time: float):
        """Processes a 'Key Up' event from the OS."""
        if not _is_right_cmd(key):
            return

        with self._state_lock:
            self._is_cmd_key_currently_held = False
            self._last_cmd_release_time = current_time

            # They let go! Cancel the delayed timer so it doesn't fire.
            # If it was just a quick tap (like CMD+C), nothing will have recorded.
            if self._delayed_start_timer:
                self._delayed_start_timer.cancel()
                self._delayed_start_timer = None

            # If we are in toggle mode, releasing the key means nothing. Just keep recording.
            if self._is_toggle_mode_active:
                return

            # If the delayed timer fired and we WERE recording from a hold, stop it now.
            if self._is_recording_due_to_hold:
                self._is_recording_due_to_hold = False
                self._on_stop_recording(stop_session=True)

    def _handle_mouse_click(self, x, y, button, pressed):
        """Processes mouse clicks (Down or Up)."""
        if button != getattr(mouse.Button, 'right', None):
            return

        with self._state_lock:
            if pressed:
                # Button Pushed
                self._mouse_press_start_time = time.time()
                self.is_mouse_button_held_down = True
            else:
                # Button Let Go
                self.is_mouse_button_held_down = False
                if self._is_recording_due_to_mouse_hold:
                    self._on_stop_recording(stop_session=True)
                    self._is_recording_due_to_mouse_hold = False

    def check_mouse_hold_threshold(self) -> bool:
        """A 'Polling' function to check if the mouse has been held long enough."""
        with self._state_lock:
            if self.is_mouse_button_held_down and not self._is_recording_due_to_mouse_hold:
                if time.time() - self._mouse_press_start_time >= 1.0:
                    self._is_recording_due_to_mouse_hold = True
                    self._on_start_recording(from_hold=True)
                    return True
        return False
