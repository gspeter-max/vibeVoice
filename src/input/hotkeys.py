import time
import threading
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
    Checks if a keyboard event is actually the 'Right Command' key.
    
    Why this exists: 
    Different computers (Mac, Windows, Linux) and different versions of Python 
    report the 'Right Command' key in different ways. This function acts as 
    a "universal translator" to make sure we always catch it.

    Step-by-step logic:
    1. Check if the key object is the standard 'cmd_r' object from pynput.
    2. If not, check if the key has a text name equal to 'cmd_r'.
    3. If still not found, check if it matches the macOS 'Virtual Key' code (54).
    
    Returns:
        True if it's our target key, False otherwise.
    """
    # 1. Standard check
    if key == getattr(keyboard.Key, 'cmd_r', None):
        return True
    
    # 2. Name-based check (for some OS versions)
    if hasattr(key, 'name') and getattr(key, 'name', None) == 'cmd_r':
        return True
    
    # 3. Hardware-code check (for macOS specifically)
    if hasattr(key, 'vk') and getattr(key, 'vk', None) == _RIGHT_COMMAND_VIRTUAL_KEY:
        return True
        
    return False

class InputTrigger:
    """
    The 'Brain' for handling keyboard and mouse shortcuts.
    
    It watches the Right Command key and Right Mouse button and tells 
    the application when to start or stop recording based on how long 
    the user holds them.
    
    Features:
    - Smart filtering: Ignores noisy "auto-repeat" signals from the OS.
    - Thread Safety: Uses a Lock to prevent crashes when multi-tasking.
    - Simple API: Just provide 3 functions (callbacks) and it does the rest.
    """
    def __init__(
        self, 
        on_start_recording: Callable[[bool], None], 
        on_stop_recording: Callable[[bool], None],
        on_toggle_recording: Callable[[], None],
        hold_threshold_seconds: float = 0.4
    ):
        """
        Initializes the trigger with custom behavior.
        
        Step-by-step setup:
        1. Store the functions (callbacks) that we will call when things happen.
        2. Set the 'hold threshold' (0.4s) which separates a 'tap' from a 'hold'.
        3. Create a 'Safety Lock' (threading.Lock) to keep our data safe.
        4. Prepare empty variables to track key/mouse state.
        """
        # Functions to call
        self._on_start_recording = on_start_recording
        self._on_stop_recording = on_stop_recording
        self._on_toggle_recording = on_toggle_recording
        
        # Timing settings
        self._hold_threshold = hold_threshold_seconds
        
        # Security & State
        self._state_lock = threading.Lock()
        
        # Keyboard tracking
        self._cmd_press_start_time = 0.0
        self._is_cmd_key_currently_held = False # Guards against OS auto-repeat noise
        self._is_toggle_mode_active = False
        
        # Mouse tracking
        self._mouse_press_start_time = 0.0
        self.is_mouse_button_held_down = False
        self._is_recording_due_to_mouse_hold = False
        
        # Placeholders for listener threads
        self._keyboard_listener_thread = None
        self._mouse_listener_thread = None

    def start_listening(self):
        """
        Spins up the background threads to start watching for inputs.
        
        Step-by-step logic:
        1. Grab the Safety Lock.
        2. Create a Keyboard Listener if it doesn't exist.
        3. Create a Mouse Listener if it doesn't exist.
        4. Start both threads so they run in the background.
        """
        with self._state_lock:
            if self._keyboard_listener_thread is None:
                self._keyboard_listener_thread = keyboard.Listener(
                    on_press=lambda k: self._handle_key_press(k, time.time()), 
                    on_release=lambda k: self._handle_key_release(k, time.time())
                )
                self._keyboard_listener_thread.start()
            
            if self._mouse_listener_thread is None:
                self._mouse_listener_thread = mouse.Listener(
                    on_click=self._handle_mouse_click
                )
                self._mouse_listener_thread.start()

    def stop_listening(self):
        """
        Shuts down the background listeners.
        
        Step-by-step logic:
        1. Grab the Safety Lock.
        2. Stop and clear the keyboard listener.
        3. Stop and clear the mouse listener.
        """
        with self._state_lock:
            if self._keyboard_listener_thread:
                self._keyboard_listener_thread.stop()
                self._keyboard_listener_thread = None
            if self._mouse_listener_thread:
                self._mouse_listener_thread.stop()
                self._mouse_listener_thread = None

    def _handle_key_press(self, key, current_time: float):
        """
        Processes a 'Key Down' event from the OS.
        
        Step-by-step logic:
        1. Ignore if the key is not the Right Command.
        2. Grab the Safety Lock.
        3. Check 'Auto-repeat': If the key is already held, STOP (OS is being noisy).
        4. Mark the key as 'Physically Held'.
        5. Check 'Toggle Mode': If we are already recording, STOP the session and exit.
        6. Start Recording: Save the start time and tell the app to begin.
        """
        if not _is_right_cmd(key):
            return

        with self._state_lock:
            # Step 3: Guard against the "Holding down" repeat signals
            if self._is_cmd_key_currently_held:
                return
            
            # Step 4: Update physical state
            self._is_cmd_key_currently_held = True

            # Step 5: Handle "Toggle Off" (User tapped it again while recording)
            if self._is_toggle_mode_active:
                self._is_toggle_mode_active = False
                self._on_stop_recording(stop_session=True)
                return

            # Step 6: Begin a new session
            self._cmd_press_start_time = current_time
            self._on_start_recording(from_hold=False)

    def _handle_key_release(self, key, current_time: float):
        """
        Processes a 'Key Up' event from the OS.
        
        Step-by-step logic:
        1. Ignore if the key is not the Right Command.
        2. Grab the Safety Lock.
        3. Mark the key as 'Physically Released'.
        4. Ignore if we just entered 'Toggle Mode' (logic handled in press).
        5. Calculate duration: (Current Time - Original Press Time).
        6. Decide:
           - If held >= 0.4s: Long Hold. Tell the app to STOP immediately.
           - If held < 0.4s: Short Tap. Enter 'Toggle Mode' (stay recording).
        """
        if not _is_right_cmd(key):
            return
            
        with self._state_lock:
            # Step 3: Update physical state
            self._is_cmd_key_currently_held = False
            
            # Step 4: Guard check
            if self._is_toggle_mode_active: 
                return

            # Step 5 & 6: Timing calculation
            duration_held = current_time - self._cmd_press_start_time
            
            if duration_held >= self._hold_threshold:
                # User held it down: Stop now that they let go.
                self._on_stop_recording(stop_session=True)
            else:
                # User just tapped it: Switch to 'Stay On' (Toggle) mode.
                self._is_toggle_mode_active = True
                self._on_toggle_recording()

    def _handle_mouse_click(self, x, y, button, pressed):
        """
        Processes mouse clicks (Down or Up).
        
        Step-by-step logic:
        1. Ignore if it's not the Right Mouse Button.
        2. Grab the Safety Lock.
        3. If Pressed: Record the start time and set 'Mouse Down' flag.
        4. If Released: Set 'Mouse Up' flag. If we were recording because 
           of this hold, stop it immediately.
        """
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
        """
        A 'Polling' function to check if the mouse has been held long enough.
        
        Step-by-step logic:
        1. Grab the Safety Lock.
        2. Check: Is the mouse button currently down AND are we NOT yet recording?
        3. Check Time: Has it been >= 1.0 second since the press started?
        4. Action: Start recording and return True. Otherwise return False.
        """
        with self._state_lock:
            # Step 2: Basic conditions
            if self.is_mouse_button_held_down and not self._is_recording_due_to_mouse_hold:
                # Step 3: Timing check
                if time.time() - self._mouse_press_start_time >= 1.0:
                    # Step 4: Fire the start command
                    self._is_recording_due_to_mouse_hold = True
                    self._on_start_recording(from_hold=True)
                    return True
        return False
