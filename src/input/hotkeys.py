import time
from typing import Callable

# We try to import pynput, which is a library that listens to the keyboard and mouse.
# If it's not installed or can't load, we create "fallback" classes so the code doesn't crash.
try:
    from pynput import keyboard, mouse
except Exception:
    # If pynput is missing, we create empty objects that look like pynput
    # so the rest of the code can still run (even if it won't actually hear keys).
    class _FallbackKey:
        cmd_r = "cmd_r"
    class _FallbackKeyboardModule:
        Key = _FallbackKey()
        Listener = lambda *args, **kwargs: None
    keyboard = _FallbackKeyboardModule()
    
    class _FallbackMouseButton:
        right = "right"
    class _FallbackMouseModule:
        Button = _FallbackMouseButton()
        Listener = lambda *args, **kwargs: None
    mouse = _FallbackMouseModule()

# This is a specific code (Virtual Key) for the Right Command key on macOS.
# We use this as a backup check if the name 'cmd_r' is missing.
_RIGHT_COMMAND_VIRTUAL_KEY = 54

def _is_right_cmd(key) -> bool:
    """
    This function checks if the key that was pressed is the 'Right Command' key.
    
    Args:
        key: The key object provided by the pynput listener.
        
    Returns:
        True if it is the Right Command key, False otherwise.
    """
    # 1. Check if the key object is exactly the standard 'cmd_r' defined by pynput.
    if key == getattr(keyboard.Key, 'cmd_r', None):
        return True
    
    # 2. Check if the key has a name property equal to 'cmd_r'.
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True
    
    # 3. Check if the key has a virtual key code (vk) equal to 54.
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == _RIGHT_COMMAND_VIRTUAL_KEY:
        return True
        
    return False

class InputTrigger:
    """
    The InputTrigger class listens for specific keyboard and mouse actions.
    It acts like a "remote control" for the application's recording state.
    
    Logic:
    - Tap Right CMD: Starts recording and stays on (Toggle Mode).
    - Hold Right CMD: Starts recording, but stops when you let go (Hold Mode).
    - Hold Right Mouse: Starts recording after 1 second, stops when let go.
    """
    def __init__(
        self, 
        on_start_recording: Callable[[bool], None], 
        on_stop_recording: Callable[[bool], None],
        on_toggle_recording: Callable[[], None],
        hold_threshold_seconds: float = 0.4
    ):
        """
        Sets up the trigger with the functions it should call when actions happen.
        
        Args:
            on_start_recording: Called when recording should begin. 
                                It takes a boolean 'from_hold'.
            on_stop_recording: Called when recording should stop.
                               It takes a boolean 'stop_session'.
            on_toggle_recording: Called when we enter 'Toggle Mode' (short tap).
            hold_threshold_seconds: How long (in seconds) to differentiate a tap from a hold.
        """
        # We store these "callback" functions to use them later.
        self._on_start_recording = on_start_recording
        self._on_stop_recording = on_stop_recording
        self._on_toggle_recording = on_toggle_recording
        
        # This is our timer limit (default 0.4 seconds).
        self._hold_threshold = hold_threshold_seconds
        
        # Variables to track the state of the keys and mouse.
        self._cmd_press_start_time = 0.0
        self._is_toggle_mode_active = False
        
        self._mouse_press_start_time = 0.0
        self.is_mouse_button_held_down = False
        self._is_recording_due_to_mouse_hold = False
        
        # These will hold the background threads that listen for inputs.
        self._keyboard_listener_thread = None
        self._mouse_listener_thread = None

    def start_listening(self):
        """
        Starts the background processes that watch for key presses and mouse clicks.
        """
        # Create and start the keyboard listener.
        self._keyboard_listener_thread = keyboard.Listener(
            on_press=lambda k: self._handle_key_press(k, time.time()), 
            on_release=lambda k: self._handle_key_release(k, time.time())
        )
        self._keyboard_listener_thread.start()
        
        # Create and start the mouse listener.
        self._mouse_listener_thread = mouse.Listener(
            on_click=self._handle_mouse_click
        )
        self._mouse_listener_thread.start()

    def stop_listening(self):
        """
        Stops the background processes. Useful for cleaning up when the app closes.
        """
        if self._keyboard_listener_thread:
            self._keyboard_listener_thread.stop()
        if self._mouse_listener_thread:
            self._mouse_listener_thread.stop()

    def _handle_key_press(self, key, current_time: float):
        """
        Called automatically whenever ANY key is pressed.
        
        Args:
            key: The key that was pressed.
            current_time: The exact timestamp of the press.
        """
        # We only care about the Right Command key.
        if not _is_right_cmd(key):
            return

        # If we are already in Toggle Mode, pressing the key again means "Stop".
        if self._is_toggle_mode_active:
            self._is_toggle_mode_active = False
            self._on_stop_recording(stop_session=True)
            return

        # Record the time the key was pressed down.
        self._cmd_press_start_time = current_time
        
        # Tell the app to start recording immediately.
        self._on_start_recording(from_hold=False)

    def _handle_key_release(self, key, current_time: float):
        """
        Called automatically whenever ANY key is released.
        
        Args:
            key: The key that was released.
            current_time: The exact timestamp of the release.
        """
        # We only care about the Right Command key.
        if not _is_right_cmd(key):
            return
            
        # If we just entered toggle mode, we don't do anything on release.
        if self._is_toggle_mode_active: 
            return

        # Calculate how long the key was held down.
        duration_held = current_time - self._cmd_press_start_time
        
        if duration_held >= self._hold_threshold:
            # LONG PRESS: The user held it down. Stop recording now that they let go.
            self._on_stop_recording(stop_session=True)
        else:
            # SHORT PRESS: The user just tapped it. Enter Toggle Mode.
            self._is_toggle_mode_active = True
            self._on_toggle_recording()

    def _handle_mouse_click(self, x, y, button, pressed):
        """
        Called automatically whenever a mouse button is pressed or released.
        
        Args:
            x, y: Coordinates (not used).
            button: Which button was clicked.
            pressed: True if pressed down, False if released.
        """
        # We only care about the Right Mouse Button.
        if button != getattr(mouse.Button, 'right', None):
            return
            
        if pressed:
            # User pushed the button down.
            self._mouse_press_start_time = time.time()
            self.is_mouse_button_held_down = True
        else:
            # User let go of the button.
            self.is_mouse_button_held_down = False
            
            # If we were recording because of this hold, stop it now.
            if self._is_recording_due_to_mouse_hold:
                self._on_stop_recording(stop_session=True)
                self._is_recording_due_to_mouse_hold = False
                
    def check_mouse_hold_threshold(self) -> bool:
        """
        This function should be called repeatedly in a loop.
        It checks if the mouse has been held down for more than 1 second.
        
        Returns:
            True if recording was just triggered by a long mouse hold.
        """
        # If mouse is down, and we haven't started recording yet...
        if self.is_mouse_button_held_down and not self._is_recording_due_to_mouse_hold:
            # ...and if it's been more than 1.0 second...
            if time.time() - self._mouse_press_start_time >= 1.0:
                # ...start recording!
                self._is_recording_due_to_mouse_hold = True
                self._on_start_recording(from_hold=True)
                return True
        return False
