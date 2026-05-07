import sys
import math
import time
import random
import platform
import logging
import socket
import threading
from typing import Optional
from ctypes import c_void_p

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QScreen

# Global activation policy setup for macOS
if platform.system() == "Darwin":
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        ns_app = NSApplication.sharedApplication()
        ns_app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except ImportError:
        pass

# Constants for the indicator dimensions (Native Menu Bar size)
INDICATOR_WIDTH = 68
INDICATOR_HEIGHT = 26

HUD_HOST, HUD_PORT = "127.0.0.1", 57234

# State definitions
STATE_HIDDEN = "HIDDEN"
STATE_LISTENING = "LISTENING"
STATE_THINKING = "THINKING"
STATE_PROCESSING = "PROCESS"
STATE_DONE = "DONE"


class HudCommandBridge(QObject):
    """
    Thread-safe bridge between the TCP server thread and the Qt main thread.

    Qt's Signal/Slot system automatically detects when a signal is emitted
    from a different thread than the receiver lives in. When that happens,
    Qt uses QueuedConnection internally — the call is posted to the
    receiver's event loop and executed safely on the main thread.

    Without this bridge, calling QTimer.singleShot from a daemon thread
    creates a timer in that thread (which has no Qt event loop), so the
    callback never fires and HUD state never changes.
    """
    command_received = Signal(str)


class RoundedRectangularIndicatorWidget(QWidget):
    """
    A persistent, non-intrusive floating indicator that renders vertical bar waveforms.
    It is designed to stay on top of all windows (including the Dock) without accepting input focus.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Fundamental Window Flags
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |       # Always above other windows
            Qt.FramelessWindowHint |        # No title bar or borders
            Qt.Tool |                       # Floating tool level (macOS/Linux)
            Qt.WindowDoesNotAcceptFocus |   # Keyboard ignores this window
            Qt.WindowTransparentForInput    # Mouse clicks go through to windows behind
        )
        
        # 2. Attributes for persistence and non-activation
        self.setAttribute(Qt.WA_TranslucentBackground)  # Transparent corners
        self.setAttribute(Qt.WA_ShowWithoutActivating)  # Don't steal focus on show
        
        if platform.system() == "Darwin":
            self.setAttribute(Qt.WA_MacAlwaysShowToolWindow) # Stay visible when app is background
            
        self.setFixedSize(INDICATOR_WIDTH, INDICATOR_HEIGHT)
        
        # Platform Specific Hardening
        self._apply_platform_specific_hardening()
        
        # Internal rendering state
        self._interface_state = STATE_HIDDEN
        self._base_amplitude = 1.0

        # Smooth transition: rendered amplitude and pill size lerp toward target each frame
        # to avoid jarring instant state changes (inspired by Wispr Flow)
        self._smooth_amplitude = 0.0
        self._smooth_width = 44.0   # Idle width
        self._smooth_height = 20.0  # Idle height
        self._last_frame_time = time.time()

        # Per-bar random phase offsets create organic, non-synchronized motion.
        # Each bar oscillates at slightly different timing, like real audio.
        self._bar_phase_offsets = [random.uniform(0, math.pi * 2) for _ in range(10)]

        # Time-based animation start (frame-rate independent)
        self._animation_start_time = time.time()

        # Colors (Solid Black with White Outline)
        self._border_color = QColor(255, 255, 255, 100)
        self._background_color = QColor(0, 0, 0, 255)

    def _apply_platform_specific_hardening(self):
        """Applies OS-level settings to ensure the indicator is non-intrusive and always on top."""
        sys_name = platform.system()
        
        if sys_name == "Darwin":
            # macOS: Ensure the window follows the user into full-screen 'Spaces'
            # AND elevate the window level to stay above the Dock.
            try:
                import objc
                from AppKit import NSWindowCollectionBehaviorCanJoinAllSpaces, NSStatusWindowLevel
                
                # Get the underlying NSWindow for the widget
                ptr = self.winId()
                # Use pyobjc to set the collection behavior and level
                # winId() on macOS provides the NSView pointer. We need the NSWindow.
                ns_view = objc.objc_object(c_void_p=ptr)
                ns_window = ns_view.window()
                
                if ns_window:
                    # Behavior: Join all spaces (full screen support)
                    ns_window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces)
                    
                    # Level: Elevate to Status Level (above Dock)
                    ns_window.setLevel_(NSStatusWindowLevel)
                else:
                    logging.debug("macOS hardening failed: Could not retrieve NSWindow from NSView.")
                
            except Exception as e:
                logging.debug(f"macOS hardening failed: {e}")
                
        elif sys_name == "Windows":
            # Windows: Set WS_EX_NOACTIVATE to prevent focus theft on click
            try:
                import ctypes
                
                GWL_EXSTYLE = -20
                WS_EX_NOACTIVATE = 0x08000000
                
                hwnd = self.winId()
                get_window_long = ctypes.windll.user32.GetWindowLongW
                set_window_long = ctypes.windll.user32.SetWindowLongW
                
                current_style = get_window_long(hwnd, GWL_EXSTYLE)
                set_window_long(hwnd, GWL_EXSTYLE, current_style | WS_EX_NOACTIVATE)
            except Exception as e:
                logging.debug(f"Windows hardening failed: {e}")

    def update_interface_state(self, state: str, amplitude: float = 1.0):
        """Updates the internal state and amplitude for the drawing logic."""
        self._interface_state = state
        self._base_amplitude = amplitude
        self.update()

    def paintEvent(self, event):
        """Renders the rounded rectangular background, outline, and vertical bars."""
        current_time = time.time()
        elapsed = current_time - self._animation_start_time
        dt = max(0.001, current_time - self._last_frame_time)
        self._last_frame_time = current_time

        # Target pill dimensions based on state
        if self._interface_state == STATE_HIDDEN:
            target_w = 44.0
            target_h = 8.0  # <--- CHANGE THIS: Height when NOT recording (idle)
        else:
            target_w = 68.0
            target_h = 26.0  # <--- CHANGE THIS: Height when RECORDING (active)

        # Smooth size lerp
        smooth_speed_size = 12.0 if target_w > self._smooth_width else 6.0
        lerp_factor_size = 1.0 - math.exp(-dt * smooth_speed_size)
        self._smooth_width += (target_w - self._smooth_width) * lerp_factor_size
        self._smooth_height += (target_h - self._smooth_height) * lerp_factor_size

        # Center the drawn pill inside the fixed 68x26 widget window
        from PySide6.QtCore import QRectF
        pill_x = (self.width() - self._smooth_width) / 2.0
        pill_y = (self.height() - self._smooth_height) / 2.0
        pill_rect = QRectF(pill_x, pill_y, self._smooth_width, self._smooth_height)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw the Pill Background (Solid Black)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._background_color))
        radius = self._smooth_height / 2.0
        painter.drawRoundedRect(pill_rect, radius, radius)
        
        # 2. Draw the "Slightly White" Outline
        outline_pen = QPen(self._border_color, 1)
        painter.setPen(outline_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(pill_rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        
        # 3. Draw the Vertical Bars
        self._draw_vertical_bars(painter, elapsed, dt, pill_rect)
        
    def _draw_vertical_bars(self, painter: QPainter, elapsed: float, dt: float, pill_rect):
        """
        Draws vertical bars with organic, Wispr Flow-inspired animation.

        Premium techniques used:
        1. Time-based animation (frame-rate independent)
        2. Smooth amplitude lerping between states (no jarring snaps)
        3. Multi-frequency layered sin waves per bar (organic motion)
        4. Center-weighted bell curve (center bars taller than edges)
        5. Per-bar opacity modulation (subtle breathing effect)
        """
        mid_y = pill_rect.center().y()

        # Bar layout — 6 centered bars, 2px wide with 2px gap
        # Thin bars with full pill-shaped rounding (radius = width/2) look like
        # smooth capsules, matching the Wispr Flow visual reference
        num_bars = 10
        bar_spacing = 4    # 2px bar + 2px gap
        bar_width = 2
        bar_rounding = bar_width / 2.0  # Full pill shape — perfectly round tips
        total_width = (num_bars - 1) * bar_spacing + bar_width
        start_x = pill_rect.center().x() - (total_width / 2.0)

        # Target amplitude based on current HUD state
        if self._interface_state == STATE_HIDDEN:
            target_amp = 0.0    # 0 = bars fade out and disappear completely
        elif self._interface_state == STATE_LISTENING:
            target_amp = 14.0 * self._base_amplitude   # Taller, prominent waves
        elif self._interface_state == STATE_THINKING:
            # Gentle breathing oscillation while AI processes
            target_amp = 3.5 * (math.sin(elapsed * 1.5) * 0.3 + 1.0)
        elif self._interface_state == STATE_PROCESSING:
            target_amp = 5.0
        else:  # DONE
            target_amp = 2.0

        # Smooth amplitude lerp using exponential decay (frame-rate independent).
        # Rise = fast (user starts speaking, instant feedback needed).
        # Fall = slow (gentle fade-out feels premium, not jarring).
        smooth_speed = 8.0 if target_amp > self._smooth_amplitude else 3.0
        lerp_factor = 1.0 - math.exp(-dt * smooth_speed)
        self._smooth_amplitude += (target_amp - self._smooth_amplitude) * lerp_factor

        amp = self._smooth_amplitude
        mid_index = (num_bars - 1) / 2.0

        painter.setPen(Qt.NoPen)

        for i in range(num_bars):
            # Coherent Traveling Wave — a smooth sine wave that travels across the bars,
            # creating a beautiful liquid "flow" effect instead of chaotic random spikes.
            # Speed = elapsed * 8.0. Wavelength = i * 0.8.
            # Using (sin + 1)/2 makes the bottoms of the wave perfectly rounded and smooth,
            # completely eliminating the sharp "bouncing ball" bottoms of an abs(sin) wave.
            wave_val = (math.sin(elapsed * 8.0 - i * 0.8) + 1.0) / 2.0

            # Center-weighted bell curve — center bars reach full height,
            # edge bars are ~45% shorter (mimics real audio spectrum shape)
            center_weight = 1.0 - abs(i - mid_index) / mid_index * 0.45

            # Final bar height
            bar_h = 1.5 + (amp * wave_val * center_weight)
            bar_h = min(bar_h, pill_rect.height() - 6)  # Stay within pill bounds

            # Fade out opacity when amp approaches 0 so dots completely disappear
            opacity_multiplier = min(1.0, amp * 2.0)
            if opacity_multiplier <= 0.01:
                continue

            # Per-bar opacity modulation — bars pulse between 180-255 alpha.
            # Creates a subtle "breathing" effect that makes the UI feel alive.
            base_alpha = 180 + 75 * abs(math.sin(elapsed * 2.0 + i * 0.5))
            alpha = int(base_alpha * opacity_multiplier)

            painter.setBrush(QBrush(QColor(255, 255, 255, alpha)))

            x = start_x + i * bar_spacing
            y = mid_y - (bar_h / 2)

            # drawRoundedRect with radius = bar_width/2 creates capsule/pill shape
            painter.drawRoundedRect(x, y, bar_width, bar_h, bar_rounding, bar_rounding)

class OscillatingInterfaceController:
    """
    Manages the lifecycle, positioning, and state transitions of the indicator widget.
    Handles adaptive frame rate — 60 FPS when active, 10 FPS when idle to save CPU.
    """
    def __init__(self):
        self.widget = RoundedRectangularIndicatorWidget()

        # Animation timer — starts at low FPS (idle), ramps up when active
        self.timer = QTimer()
        self.timer.timeout.connect(self.widget.update)
        self.timer.start(100)  # 10 FPS when idle — saves CPU

        # Positioning
        self._position_at_bottom_center()
        self.widget.show()

    def _position_at_bottom_center(self):
        """Positions the widget at the horizontal center, absolute bottom of the screen."""
        screen = QApplication.primaryScreen()
        if not screen:
            return

        geo = screen.geometry()
        x = geo.x() + (geo.width() - INDICATOR_WIDTH) // 2
        y = geo.y() + geo.height() - INDICATOR_HEIGHT

        self.widget.move(x, y)

    def _set_animation_speed(self, interval_ms: int):
        """
        Adjusts the repaint timer interval for adaptive frame rate.
        Active states get 60 FPS (16ms) for smooth animation.
        Idle state gets 10 FPS (100ms) to save CPU.
        """
        if self.timer.interval() != interval_ms:
            self.timer.setInterval(interval_ms)

    @Slot(str, float)
    def on_interface_command(self, command: str, amplitude: float = 1.0):
        """Handles incoming commands to change the indicator's behavior."""
        cmd = command.lower()
        if cmd == "listen":
            self.widget.update_interface_state(STATE_LISTENING, amplitude)
            self._set_animation_speed(16)  # 60 FPS for smooth waveform
        elif cmd in ["think", "process"]:
            self.widget.update_interface_state(STATE_THINKING if cmd == "think" else STATE_PROCESSING)
            self._set_animation_speed(16)  # 60 FPS for smooth waveform
        elif cmd == "done":
            self.widget.update_interface_state(STATE_DONE)
            self._set_animation_speed(16)  # Keep smooth during fade-out
            QTimer.singleShot(1500, self._return_to_idle)
        elif cmd == "hide":
            self._return_to_idle()

    def _return_to_idle(self):
        """Resets the indicator to the subtle idle state."""
        self.widget.update_interface_state(STATE_HIDDEN)
        # Delay the FPS drop so the smooth fade-out animation completes at 60 FPS
        QTimer.singleShot(600, lambda: self._set_animation_speed(100))

class HudServer(threading.Thread):
    """
    TCP server that listens for state commands from Ear and Brain.

    Runs in a daemon thread. When a command arrives (e.g. "listen", "process"),
    it emits a Qt Signal through the HudCommandBridge. Qt automatically
    queues the signal and delivers it on the main thread where the
    controller can safely update the widget.
    """
    def __init__(self, controller):
        super().__init__(daemon=True)
        self.controller = controller

        # Bridge lives in the main thread (created here in __init__ which
        # runs on the main thread). Signal emission from the daemon thread
        # is thread-safe — Qt detects cross-thread and uses QueuedConnection.
        self.command_bridge = HudCommandBridge()
        self.command_bridge.command_received.connect(controller.on_interface_command)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HUD_HOST, HUD_PORT))
            except Exception as e:
                print(f"HUD Server bind error: {e}")
                return
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if not data:
                        continue
                    command = data.decode().strip()
                    # Emit signal — Qt queues it to the main thread automatically
                    self.command_bridge.command_received.emit(command)

def initialize_hud() -> OscillatingInterfaceController:
    """Helper to instantiate the controller."""
    return OscillatingInterfaceController()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = initialize_hud()
    server = HudServer(controller)
    server.start()
    sys.exit(app.exec())
