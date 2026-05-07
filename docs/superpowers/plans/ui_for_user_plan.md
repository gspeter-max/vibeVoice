 For agentic workers: REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps  │
│ use checkbox (- [ ]) syntax for tracking.                                                                  │
│                                                                                                            │
│ Goal information for freshAgent :                                                                          │
│      - Convert the current macOS-only menu bar HUD (src/ui/hud.py) into a cross-platform (macOS/Linux)     │
│        PySide6 floating "Pill" window.                                                                     │
│      - Replace the discrete rectangular bars with a continuous, anti-aliased "Whisper Flow" cubic Bezier   │
│        liquid waveform.                                                                                    │
│      - The window must be a frameless, transparent-background capsule (pill) anchored to the bottom-center │
│        of the screen.                                                                                      │
│      - CRITICAL: Do NOT implement dragging/interactivity in this MVP phase. The focus is purely on the     │
│        cross-platform rendering and liquid waveform math.                                                  │
│      - CRITICAL: for agent, you should not assume anything, you should strictly follow the plan.    │
│        Ask questions if you don't understand anything.                                                     │
│      - File Context:                                                                                       │
│          - src/ui/hud.py: The target file. We will completely rewrite the MenuBarWaveformView and          │
│            MenuBarWaveformController logic, stripping out AppKit and replacing it with pure PySide6        │
│            QWidget and QPainterPath drawing.                                                               │
│                                                                                                            │
│ Architecture:                                                                                              │
│  - UI Shell: A frameless QWidget acting as the main window (Qt.FramelessWindowHint |                       │
│    Qt.WindowStaysOnTopHint | Qt.Tool).                                                                     │
│  - Transparency: Handled via self.setAttribute(Qt.WA_TranslucentBackground) to allow the dark, rounded     
│    pill background to float naturally over other windows.                                                  
│  - Rendering Engine: Override paintEvent(self, event). Use QPainter with high-quality antialiasing.        
│  - Waveform Math: Generate 3 layers of QPainterPath using cubicTo() to connect sine-wave generated points  │
│    for the liquid effect.                                                                                  │
│                                                                                                            │
│ Important Rule to follow :                                                                                 │
│  - CRITICAL:  add detailed docs in functions and explain the code and logic in comments.                   │
│  - (CRITICAL make the code function name and variable name clear and easily to understand instand of short │
│    and confusing names                                                                                     │
│      - so 5 year old child easily understand                                                               │
│      - do not put any imagination and analogy to understand for 5 year old child                           │
│      - write code function name and docs and code like this developer get hightest speed to read the code  │
│      - Explain like a fresher                                                                              │
│      - Write docs in your step-by-step simple style.                                                       │
│      - Make the docs in function and file headers human-readable and literal.                              │
│                                                                                                            │
│ ---                                                                                                        │
│                                                                                                            │
│ Task 1 : Setup and Initialization                                                                          │
│                                                                                                            │
│  - [ ] read GEMINI.md file                                                                                 │
│  - CRITICAL:  add detailed docs in functions and explain the code and logic in comments.                   │
│  - (CRITICAL make the code function name and variable name clear not easily to understand instand of short │
│    and confusing names                                                                                     │
│                                                                                                            │
│ ---                                                                                                        │
│                                                                                                            │
│ Task 2: Pure PySide6 Window Shell Refactor                                                                 │
│                                                                                                            │
│ Files:                                                                                                     │
│  - Modify: src/ui/hud.py                                                                                   │
│                                                                                                            │
│  - [ ] Step 1: Remove AppKit Dependencies and Setup Imports                                                │
│   Remove all macOS AppKit imports. Add necessary PySide6.QtGui imports.                                    │
│                                                                                                            │
│   1 import sys                                                                                             │
│   2 import os                                                                                              │
│   3 import math                                                                                            │
│   4 import time                                                                                            │
│   5 import socket                                                                                          │
│   6 import subprocess                                                                                      │
│   7 import random                                                                                          │
│   8 from src import log                                                                                    │
│   9                                                                                                        │
│  10 from PySide6.QtCore import QObject, QTimer, QThread, Signal, Qt, QRectF, QPointF                       │
│  11 from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush                                 │
│  12 from PySide6.QtWidgets import QApplication, QWidget                                                    │
│  13                                                                                                        │
│  14 # Remove AppKit try/except block entirely                                                              │
│                                                                                                            │
│  - [ ] Step 2: Create the LiquidPillWidget Class                                                           │
│   Replace MenuBarWaveformView with a pure PySide6 QWidget that handles the frameless window and background │
│ painting.                                                                                                  │
│                                                                                                            │
│   1 class LiquidPillWidget(QWidget):                                                                       │
│   2     """                                                                                                │
│   3     A cross-platform, floating, transparent 'Pill' shaped window.                                      │
│   4     This replaces the macOS-specific menu bar view.                                                    │
│   5     """                                                                                                │
│   6     def __init__(self, controller):                                                                    │
│   7         super().__init__()                                                                             │
│   8         self.controller = controller                                                                   │
│   9                                                                                                        │
│  10         # Configure window to be frameless, always on top, and not show in taskbar                     │
│  11         self.setWindowFlags(                                                                           │
│  12             Qt.FramelessWindowHint |                                                                   │
│  13             Qt.WindowStaysOnTopHint |                                                                  │
│  14             Qt.Tool                                                                                    │
│  15         )                                                                                              │
│  16         # Allow transparency for the rounded pill corners                                              │
│  17         self.setAttribute(Qt.WA_TranslucentBackground)                                                 │
│  18                                                                                                        │
│  19         # Fixed MVP dimensions                                                                         │
│  20         self.setFixedSize(160, 40)                                                                     │
│  21         self.position_bottom_center()                                                                  │
│  22                                                                                                        │
│  23     def position_bottom_center(self):                                                                  │
│  24         """Places the pill at the bottom center of the primary screen."""                              │
│  25         screen = QApplication.primaryScreen().geometry()                                               │
│  26         x = (screen.width() - self.width()) // 2                                                       │
│  27         y = screen.height() - self.height() - 50 # 50px padding from bottom                            │
│  28         self.move(x, y)                                                                                │
│  29                                                                                                        │
│  30     def paintEvent(self, event):                                                                       │
│  31         """                                                                                            │
│  32         Main drawing routine called every frame.                                                       │
│  33         Draws the dark pill background, then delegates waveform drawing to the controller.             │
│  34         """                                                                                            │
│  35         painter = QPainter(self)                                                                       │
│  36         painter.setRenderHint(QPainter.Antialiasing)                                                   │
│  37                                                                                                        │
│  38         # 1. Draw the dark, semi-transparent pill background                                           │
│  39         pill_rect = QRectF(0, 0, self.width(), self.height())                                          │
│  40         painter.setBrush(QColor(20, 20, 20, 200)) # Dark gray, 80% opacity                             │
│  41         painter.setPen(Qt.NoPen)                                                                       │
│  42         # Radius is half the height to make a perfect pill shape                                       │
│  43         radius = self.height() / 2.0                                                                   │
│  44         painter.drawRoundedRect(pill_rect, radius, radius)                                             │
│  45                                                                                                        │
│  46         # 2. Draw the liquid waves                                                                     │
│  47         self.controller.draw_liquid_waves(painter, self.width(), self.height())                        │
│                                                                                                            │
│  - [ ] Step 3: Update MenuBarWaveformController initialization                                             │
│   Rename to LiquidHudController and initialize the new widget.                                             │
│                                                                                                            │
│   1 class LiquidHudController(QObject):                                                                    │
│   2     def __init__(self):                                                                                │
│   3         super().__init__()                                                                             │
│   4                                                                                                        │
│   5         # ... keep existing IPC, state, and timer initialization ...                                   │
│   6                                                                                                        │
│   7         self.widget = LiquidPillWidget(self)                                                           │
│   8         self.widget.show()                                                                             │
│   9                                                                                                        │
│  10         # Replace _ensure_menu_bar_item logic                                                          │
│  11         log.info("[HUD] Liquid Pill HUD ready ✓")                                                      │
│                                                                                                            │
│ ---                                                                                                        │
│                                                                                                            │
│ Task 3: Liquid Bezier Wave Math & Rendering                                                                │
│                                                                                                            │
│ Files:                                                                                                     │
│  - Modify: src/ui/hud.py                                                                                   │
│                                                                                                            │
│  - [ ] Step 1: Implement draw_liquid_waves in the Controller                                               │
│   Replace _draw_menu_bar_waveform with the new continuous path logic.                                      │
│                                                                                                            │
│   1     def draw_liquid_waves(self, painter: QPainter, width: float, height: float):                       │
│   2         """                                                                                            │
│   3         Calculates and draws the continuous, overlapping cubic Bezier paths                            │
│   4         that create the 'Whisper Flow' liquid aesthetic.                                               │
│   5         """                                                                                            │
│   6         if self._state == HIDDEN:                                                                      │
│   7             return                                                                                     │
│   8                                                                                                        │
│   9         t = self._t                                                                                    │
│  10         v = self._voice_smooth                                                                         │
│  11                                                                                                        │
│  12         # Base amplitude multipliers based on state                                                    │
│  13         base_amplitude = 2.0                                                                           │
│  14         if self._state == LISTENING:                                                                   │
│  15             base_amplitude = 4.0 + (v * 15.0) # Swells significantly when speaking                     │
│  16         elif self._state == THINKING:                                                                  │
│  17             base_amplitude = 3.0 + math.sin(t * 3.0) * 1.0                                             │
│  18                                                                                                        │
│  19         center_y = height / 2.0                                                                        │
│  20                                                                                                        │
│  21         # We will draw 3 overlapping layers for depth (Core, Glow, Ghost)                              │
│  22         layers = [                                                                                     │
│  23             {"opacity": 255, "speed": 1.0, "freq": 0.05, "amp_mod": 1.0, "width": 1.5},  # Core        │
│     Spine                                                                                                  │
│  24             {"opacity": 100, "speed": 1.3, "freq": 0.07, "amp_mod": 0.8, "width": 3.0},  # Ghost       │
│  25             {"opacity": 40,  "speed": 0.8, "freq": 0.03, "amp_mod": 1.2, "width": 6.0},  # Glow        │
│  26         ]                                                                                              │
│  27                                                                                                        │
│  28         # How many points to calculate across the pill width                                           │
│  29         num_points = 20                                                                                │
│  30         step_x = width / (num_points - 1)                                                              │
│  31                                                                                                        │
│  32         for layer in layers:                                                                           │
│  33             path = QPainterPath()                                                                      │
│  34                                                                                                        │
│  35             # Start path on the left edge, vertically centered                                         │
│  36             path.moveTo(0, center_y)                                                                   │
│  37                                                                                                        │
│  38             prev_x = 0                                                                                 │
│  39             prev_y = center_y                                                                          │
│  40                                                                                                        │
│  41             for i in range(1, num_points):                                                             │
│  42                 x = i * step_x                                                                         │
│  43                                                                                                        │
│  44                 # Math: Sum of sines for organic, non-repeating movement                               │
│  45                 # We use distance from center to taper the ends of the wave                            │
│  46                 dist_from_center = abs((x / width) - 0.5) * 2.0 # 0 at center, 1 at edges              │
│  47                 taper = max(0.0, 1.0 - (dist_from_center ** 2)) # Quadratic falloff                    │
│  48                                                                                                        │
│  49                 wave_math = math.sin((x * layer["freq"]) + (t * layer["speed"] * 5.0))                 │
│  50                 y = center_y + (wave_math * base_amplitude * layer["amp_mod"] * taper)                 │
│  51                                                                                                        │
│  52                 # Calculate Control Points for the Cubic Bezier curve                                  │
│  53                 # To make it smooth, control points sit halfway between X coordinates                  │
│  54                 cp_x = (prev_x + x) / 2.0                                                              │
│  55                                                                                                        │
│  56                 # cubicTo(controlPoint1X, controlPoint1Y, controlPoint2X, controlPoint2Y,              │
│     endPointX, endPointY)                                                                                  │
│  57                 path.cubicTo(cp_x, prev_y, cp_x, y, x, y)                                              │
│  58                                                                                                        │
│  59                 prev_x = x                                                                             │
│  60                 prev_y = y                                                                             │
│  61                                                                                                        │
│  62             # Draw this specific layer                                                                 │
│  63             alpha = int(layer["opacity"] * self._status_bar_alpha())                                   │
│  64             pen = QPen(QColor(255, 255, 255, alpha))                                                   │
│  65             pen.setWidthF(layer["width"])                                                              │
│  66             pen.setCapStyle(Qt.RoundCap)                                                               │
│  67                                                                                                        │
│  68             painter.setPen(pen)                                                                        │
│  69             painter.setBrush(Qt.NoBrush)                                                               │
│  70             painter.drawPath(path)                                                                     │
│                                                                                                            │
│  - [ ] Step 2: Update the _tick method                                                                     │
│   Remove the discrete bar logic (the for i in range(NUM_BARS): loop) and simply call self.widget.update()  │
│ to trigger a repaint with the new _t value.                                                                │
│                                                                                                            │
│   1     def _tick(self):                                                                                   │
│   2         self._t = time.time() - self._t0                                                               │
│   3                                                                                                        │
│   4         if time.time() - self._last_vol_t > 0.15:                                                      │
│   5             self._voice_raw *= 0.80                                                                    │
│   6                                                                                                        │
│   7         target = self._voice_raw if self._state == LISTENING else 0.0                                  │
│   8         spd    = 0.38 if target > self._voice_smooth else 0.08                                         │
│   9         self._voice_smooth += (target - self._voice_smooth) * spd                                      │
│  10                                                                                                        │
│  11         # The mathematical state updates are now handled directly inside draw_liquid_waves             │
│  12         # We just need to tell the widget to redraw itself.                                            │
│  13         self.widget.update()                                                                           │
│  14                                                                                                        │
│  15         if self._state == HIDDEN:                                                                      │
│  16             self._timer.stop()                                                                         │
│                                                                                                            │
│ ---                                                                                                        │
│                                                                                                            │
│ Task 4: Final Wiring and Cleanup                                                                           │
│                                                                                                            │
│  - [ ] Step 1: Update Main execution block                                                                 │
│   Update the bottom of hud.py to instantiate the new controller.                                           │
│                                                                                                            │
│  1 if __name__ == "__main__":                                                                              │
│  2     app = QApplication(sys.argv)                                                                        │
│  3     app.setQuitOnLastWindowClosed(False)                                                                │
│  4                                                                                                         │
│  5     # We no longer need NSApplicationActivationPolicyAccessory since we aren't                          │
│  6     # fighting the macOS dock directly with AppKit. Pure Qt handles it via Qt.Tool.                     │
│  7                                                                                                         │
│  8     hud = LiquidHudController()                                                                         │
│  9     sys.exit(app.exec())                                                                                │
│                                                                                                            │
│  - [ ] Step 2: Remove dead code                                                                            │
│   Delete compute_menu_bar_waveform_layout, _triangle_wave, and the old NUM_BARS constants at the top of    │
│ the file as they are no longer used.                                                                       │
│                                                                                                            │
│ Self-Review (before sharing the plan)                                                                      │
│  - [x] Does this plan remove macOS-only dependencies? Yes.                                                 │
│  - [x] Does it implement the continuous Bezier curve? Yes, via QPainterPath.cubicTo.                       │
│  - [x] Are the names clear and documented literally? Yes (LiquidPillWidget, draw_liquid_waves).            │
│  - [x] Did we intentionally omit the dragging feature for this MVP phase? Yes. 