

# Fixing the Invisible HUD on Intel Mac + Siri-Style Orb

There are **three likely causes** why you see nothing. Let me walk through diagnosis and then give you a complete Siri-style orb replacement.

---

## Step 1 — Find the Problem

Run these checks **one at a time** in your terminal:

### Check A: Is `pynput` even detecting your key press?

```bash
# Save and run this — press RIGHT Command key
cd /path/to/your/project
./.venv/bin/python - <<'EOF'
from pynput import keyboard
def on_press(key):
    print(f"PRESSED:  {key!r}  type={type(key).__name__}")
def on_release(key):
    print(f"RELEASED: {key!r}")
    if key == keyboard.Key.esc:
        return False
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Press any key... (ESC to quit)")
    print("Look for Key.cmd_r when you press RIGHT Command")
    listener.join()
EOF
```

**If you see NOTHING printed when pressing keys** → your Terminal app doesn't have **Accessibility permission**. This is the #1 cause. Fix:

> **System Preferences → Security & Privacy → Privacy → Accessibility**  
> Add **Terminal.app** (or **iTerm2.app**)  
> Also check **Input Monitoring** tab and add it there too.  
> Then **restart Terminal**.

**If it prints `Key.cmd` instead of `Key.cmd_r`** → your pynput version doesn't distinguish left/right. I'll fix this in the code below.

### Check B: Is the HUD even running?

```bash
cat hud.log
```

If it shows an import error like `ModuleNotFoundError: No module named 'PySide6'`:
```bash
./.venv/bin/pip install PySide6
```

If it shows `IPC bind failed` → port 57234 is already in use:
```bash
lsof -i :57234
# Kill whatever is on that port, then restart
```

### Check C: Does the HUD render at all?

```bash
# Run HUD directly with demo mode
./.venv/bin/python hud.py --demo
```

If you see **nothing appear on screen** → Qt rendering issue on Intel Mac. The fix is the environment variable in Section 2.

### Check D: Can IPC reach the HUD?

```bash
# In one terminal, run HUD:
./.venv/bin/python hud.py

# In another terminal, send a command:
./.venv/bin/python -c "
import socket
s = socket.socket()
s.connect(('127.0.0.1', 57234))
s.sendall(b'listen')
s.close()
print('Sent listen command')
"
```

---

## Step 2 — All-In-One Diagnostic Script

Save this as **`test_diagnostic.py`** and run it:

```python
#!/usr/bin/env python3
"""
test_diagnostic.py — Run this to find exactly what's broken
"""
import sys, os, socket, subprocess

def check(label, ok, fix=""):
    status = "✅" if ok else "❌"
    print(f"  {status} {label}")
    if not ok and fix:
        print(f"     → FIX: {fix}")
    return ok

print("\n🔍 PARAKEET FLOW — DIAGNOSTIC\n" + "═" * 50)

# 1. Python version
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v.major == 3 and v.minor >= 9,
      "Need Python 3.9+")

# 2. PySide6
try:
    import PySide6
    from PySide6.QtWidgets import QApplication
    check(f"PySide6 {PySide6.__version__} installed", True)
except ImportError:
    check("PySide6 installed", False, "pip install PySide6")

# 3. pynput
try:
    from pynput import keyboard
    check("pynput installed", True)
    has_cmd_r = hasattr(keyboard.Key, 'cmd_r')
    check(f"pynput has Key.cmd_r: {has_cmd_r}", True)
except ImportError:
    check("pynput installed", False, "pip install pynput")

# 4. pyaudio
try:
    import pyaudio
    check("pyaudio installed", True)
except ImportError:
    check("pyaudio installed", False, "pip install pyaudio")

# 5. Socket path
sock_exists = os.path.exists("/tmp/parakeet.sock")
check(f"Brain socket exists: {sock_exists}", sock_exists,
      "Brain isn't running. Run ./start.sh first")

# 6. HUD port
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    s.connect(('127.0.0.1', 57234))
    s.close()
    check("HUD port 57234 reachable", True)
except Exception:
    check("HUD port 57234 reachable", False,
          "HUD isn't running. Check hud.log for errors")

# 7. macOS accessibility (rough check)
if sys.platform == "darwin":
    # Check if we can create a keyboard listener without error
    try:
        from pynput import keyboard as kb
        results = []
        def _test_press(key):
            results.append(key)
            return False
        listener = kb.Listener(on_press=_test_press)
        listener.start()
        import time
        time.sleep(0.5)
        listener.stop()
        check("pynput listener created (Accessibility likely OK)", True)
    except Exception as e:
        check("pynput listener", False,
              f"Grant Accessibility permission to Terminal: {e}")

# 8. Qt environment
qt_layer = os.environ.get("QT_MAC_WANTS_LAYER", "not set")
check(f"QT_MAC_WANTS_LAYER={qt_layer}", qt_layer == "1",
      "Add: export QT_MAC_WANTS_LAYER=1 to start.sh")

# 9. hud.log check
if os.path.exists("hud.log"):
    with open("hud.log") as f:
        content = f.read().strip()
    if content:
        has_error = any(w in content.lower() for w in ["error", "traceback", "failed"])
        if has_error:
            check("hud.log (errors found!)", False, f"Contents:\n{content[-500:]}")
        else:
            check(f"hud.log OK: {content[:100]}", True)
    else:
        check("hud.log (empty — HUD may not have started)", False,
              "Run: python hud.py --demo to test")
else:
    check("hud.log exists", False, "HUD was never launched")

print("\n" + "═" * 50)
print("Run with: ./.venv/bin/python test_diagnostic.py\n")
```

---

## Step 3 — The Fixes

### Fix 1: macOS Accessibility (MOST COMMON)

```
System Preferences → Security & Privacy → Privacy
  → Accessibility: add Terminal.app ✓
  → Input Monitoring: add Terminal.app ✓
Restart Terminal after adding.
```

### Fix 2: Qt Rendering on Intel Mac

Add these environment variables to **`start.sh`** (right after `set -euo pipefail`):

```bash
# Fix Qt rendering on Intel Macs
export QT_MAC_WANTS_LAYER=1
```

### Fix 3: Command Key Detection

Some pynput versions on macOS don't distinguish `cmd_r` vs `cmd`. Update the hotkey check in **`ear.py`** — I'll include this fix in the updated ear.py snippet below.

---

## Step 4 — Complete Siri-Style Orb HUD

Replace your **`hud.py`** entirely with this:

```python
"""
hud.py — Siri-Style Animated Orb HUD for Parakeet Flow v2
==========================================================
A floating circular orb with colorful blob animation, inspired by Apple Siri.

States:
  LISTENING   → colorful blobs orbit and pulse actively (like Siri listening)
  PROCESSING  → blobs spin faster + spinning gradient arc
  DONE        → green glow + animated checkmark, then auto-fade

IPC Control (TCP on port 57234):
  "listen"   → show listening orb
  "process"  → show processing animation
  "done"     → show checkmark (auto-hides after 1.4s)
  "hide"     → hide immediately

Run standalone demo:
  python hud.py --demo
"""

import sys
import os
import math
import time
import socket

# Fix Qt rendering on Intel Macs — must be set BEFORE importing PySide6
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QThread, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPainterPath, QPen, QBrush,
    QRadialGradient, QLinearGradient,
)

# ── Layout Constants ───────────────────────────────────────────────────────────
ORB_DIAMETER = 160
PADDING      = 24          # room for outer glow
WINDOW_SIZE  = ORB_DIAMETER + PADDING * 2   # 208

IPC_PORT = 57234

HIDDEN     = "hidden"
LISTENING  = "listening"
PROCESSING = "processing"
DONE       = "done"


# ── Blob definition ───────────────────────────────────────────────────────────
class Blob:
    """A colourful gradient sphere that orbits the centre."""
    __slots__ = ("r", "g", "b", "phase", "orbit_frac", "size_frac", "speed")

    def __init__(self, r, g, b, phase, orbit_frac, size_frac, speed):
        self.r = r
        self.g = g
        self.b = b
        self.phase = phase
        self.orbit_frac = orbit_frac   # orbit radius as fraction of orb radius
        self.size_frac  = size_frac    # blob radius as fraction of orb radius
        self.speed      = speed        # angular speed multiplier


# Apple-inspired palette — large overlapping blobs create colour blending
BLOBS = [
    Blob(10,  132, 255,  0.00,  0.20, 0.78, 0.70),   # Blue
    Blob(180,  80, 222,  1.05,  0.25, 0.68, 1.00),   # Purple
    Blob(255,  45,  85,  2.10,  0.18, 0.74, 0.85),   # Pink
    Blob(80,  200, 255,  3.15,  0.22, 0.64, 1.15),   # Cyan
    Blob(50,  215,  85,  4.20,  0.15, 0.60, 0.95),   # Green
    Blob(100,  80, 220,  5.25,  0.28, 0.70, 1.30),   # Indigo
]


# ── IPC server thread ─────────────────────────────────────────────────────────
class IPCServer(QThread):
    command = Signal(str)

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", IPC_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            print(f"[HUD] IPC listening on port {IPC_PORT}")
            while not self.isInterruptionRequested():
                try:
                    conn, _ = srv.accept()
                    data = conn.recv(256).decode().strip()
                    conn.close()
                    if data:
                        self.command.emit(data)
                        if data == "quit":
                            break
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[HUD] IPC error: {e}")
        except Exception as e:
            print(f"[HUD] ❌ Failed to bind port {IPC_PORT}: {e}")
        finally:
            srv.close()


# ── Siri Orb HUD ──────────────────────────────────────────────────────────────
class SiriOrbHUD(QWidget):

    def __init__(self):
        super().__init__()

        # ── Window flags ───────────────────────────────────────────────────
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)

        # Centre-top of screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - WINDOW_SIZE // 2,
            screen.top() + 28,
        )

        # ── Animation state ───────────────────────────────────────────────
        self._state          = HIDDEN
        self._t0             = time.time()
        self._t              = 0.0
        self._fade           = 0.0
        self._fade_dir       = 0       # +1 fade in, -1 fade out
        self._intensity      = 0.0     # current animation energy (0–1)
        self._target_intens  = 0.5
        self._spin_offset    = 0.0     # extra rotation for PROCESSING
        self._check_progress = 0.0     # checkmark draw progress

        # ── 60 fps timer ──────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        # ── IPC ───────────────────────────────────────────────────────────
        self._ipc = IPCServer(self)
        self._ipc.command.connect(self._on_command)
        self._ipc.start()

        print("[HUD] Siri Orb initialised ✓")

    # ── Public API ─────────────────────────────────────────────────────────

    def show_listening(self):
        self._enter(LISTENING)
        self._target_intens = 1.0

    def show_processing(self):
        self._enter(PROCESSING)
        self._target_intens = 0.7

    def show_done(self):
        self._enter(DONE)
        self._check_progress = 0.0
        self._target_intens  = 0.0
        QTimer.singleShot(1400, self.hide_hud)

    def hide_hud(self):
        self._fade_dir = -1

    # ── Internals ──────────────────────────────────────────────────────────

    def _enter(self, state):
        self._state    = state
        self._t0       = time.time()
        self._t        = 0.0
        self._fade_dir = +1
        if not self.isVisible():
            self._fade = 0.0
            self.setWindowOpacity(0.0)
            self.show()
            self.raise_()              # ensure it's on top
        self._timer.start()

    def _tick(self):
        self._t = time.time() - self._t0

        # Fade envelope
        if self._fade_dir == +1:
            self._fade = min(1.0, self._fade + 0.065)
            self.setWindowOpacity(self._fade)
        elif self._fade_dir == -1:
            self._fade = max(0.0, self._fade - 0.05)
            self.setWindowOpacity(self._fade)
            if self._fade <= 0:
                self._timer.stop()
                self.hide()
                self._state = HIDDEN
                return

        # Smooth intensity lerp
        self._intensity += (self._target_intens - self._intensity) * 0.08

        # Per-state updates
        if self._state == PROCESSING:
            self._spin_offset += 4.5 * 0.016
        elif self._state == DONE:
            self._check_progress = min(1.0, self._t * 2.5)

        self.update()

    def _on_command(self, cmd: str):
        c = cmd.strip().lower()
        if   c == "listen":  self.show_listening()
        elif c == "process": self.show_processing()
        elif c == "done":    self.show_done()
        elif c == "hide":    self.hide_hud()

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        ws  = WINDOW_SIZE
        cx  = ws / 2.0
        cy  = ws / 2.0
        rad = ORB_DIAMETER / 2.0

        # ── 1. Outer glow (drawn before clip) ─────────────────────────────
        self._draw_outer_glow(p, cx, cy, rad)

        # ── 2. Circular clip for the orb ──────────────────────────────────
        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), rad, rad)
        p.setClipPath(clip)

        # ── 3. Dark glass background ──────────────────────────────────────
        p.fillPath(clip, QBrush(QColor(8, 8, 14, 225)))

        # Subtle top sheen
        sheen = QLinearGradient(QPointF(cx, cy - rad), QPointF(cx, cy))
        sheen.setColorAt(0.0, QColor(255, 255, 255, 18))
        sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillPath(clip, QBrush(sheen))

        # ── 4. State-specific content ─────────────────────────────────────
        if self._state == LISTENING:
            self._draw_blobs(p, cx, cy, rad, speed_mult=1.0)
        elif self._state == PROCESSING:
            self._draw_blobs(p, cx, cy, rad, speed_mult=2.5)
            self._draw_spinner(p, cx, cy, rad)
        elif self._state == DONE:
            self._draw_done_glow(p, cx, cy, rad)
            self._draw_checkmark(p, cx, cy, rad)

        # ── 5. Rim ────────────────────────────────────────────────────────
        p.setClipping(False)
        rim_alpha = 55
        if self._state == LISTENING:
            hue = (self._t * 25) % 360
            rim_c = QColor.fromHsvF(hue / 360.0, 0.4, 1.0)
            rim_c.setAlpha(rim_alpha)
        elif self._state == DONE:
            rim_c = QColor(50, 215, 75, int(rim_alpha * self._check_progress))
        else:
            rim_c = QColor(255, 255, 255, rim_alpha)

        p.setPen(QPen(rim_c, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rad, rad)

        p.end()

    # ── Outer glow ─────────────────────────────────────────────────────────

    def _draw_outer_glow(self, p: QPainter, cx, cy, rad):
        """Soft coloured halo behind the orb, changes with state."""
        if self._state == HIDDEN:
            return

        intensity = self._intensity
        pulse     = 0.5 + 0.5 * math.sin(self._t * 2.0)

        if self._state == LISTENING:
            r, g, b = 30, 100, 255
        elif self._state == PROCESSING:
            r, g, b = 110, 60, 255
        elif self._state == DONE:
            r, g, b = 50, 215, 75
        else:
            r, g, b = 80, 80, 120

        a = int((20 + 25 * pulse) * intensity)

        glow = QRadialGradient(QPointF(cx, cy), rad + PADDING)
        glow.setColorAt(0.5,  QColor(r, g, b, a))
        glow.setColorAt(0.75, QColor(r, g, b, a // 2))
        glow.setColorAt(1.0,  QColor(r, g, b, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(glow))

    # ── Colour blobs (Siri waveform) ───────────────────────────────────────

    def _draw_blobs(self, p: QPainter, cx, cy, rad, speed_mult=1.0):
        t   = self._t
        inten = self._intensity

        for blob in BLOBS:
            # Organic orbital angle with wobble
            angle = (t * blob.speed * speed_mult
                     + blob.phase
                     + self._spin_offset
                     + 0.3 * math.sin(t * 0.73 + blob.phase * 2.1))

            # Elliptical orbit that breathes
            orb_rx = blob.orbit_frac * rad * inten
            orb_ry = blob.orbit_frac * rad * inten * 0.85
            orb_rx *= 1.0 + 0.25 * math.sin(t * 1.3 + blob.phase * 0.8)
            orb_ry *= 1.0 + 0.25 * math.cos(t * 1.6 + blob.phase * 1.1)

            bx = cx + math.cos(angle) * orb_rx
            by = cy + math.sin(angle) * orb_ry

            # Blob size pulses with multiple frequencies
            sz = blob.size_frac * rad
            sz *= (0.65 + 0.50 * (math.sin(t * 2.0 + blob.phase * 1.3) ** 2))
            sz *= (0.45 + 0.55 * inten)
            sz  = max(4.0, sz)

            # Alpha envelope
            a_core = int(160 + 70 * inten)
            a_mid  = int(70  + 40 * inten)
            a_edge = int(20  + 15 * inten)

            grad = QRadialGradient(QPointF(bx, by), sz)
            grad.setColorAt(0.00, QColor(blob.r, blob.g, blob.b, a_core))
            grad.setColorAt(0.30, QColor(blob.r, blob.g, blob.b, a_mid))
            grad.setColorAt(0.65, QColor(blob.r, blob.g, blob.b, a_edge))
            grad.setColorAt(1.00, QColor(blob.r, blob.g, blob.b, 0))

            p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(grad))

        # Bright core (white/blue centre)
        core_pulse = 0.5 + 0.5 * math.sin(t * 3.0)
        core_sz    = 10 + 8 * core_pulse * inten
        core_a     = int(35 + 55 * core_pulse * inten)

        core = QRadialGradient(QPointF(cx, cy), max(1, core_sz))
        core.setColorAt(0.0, QColor(210, 225, 255, core_a))
        core.setColorAt(0.4, QColor(160, 190, 255, core_a // 3))
        core.setColorAt(1.0, QColor(120, 150, 255, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(core))

    # ── Processing spinner arc ─────────────────────────────────────────────

    def _draw_spinner(self, p: QPainter, cx, cy, rad):
        arc_r = rad * 0.88
        arc_rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)

        # Background ring
        p.setPen(QPen(QColor(255, 255, 255, 18), 2.5,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), arc_r, arc_r)

        # Animated gradient arc
        angle = self._t * 320
        segs  = 24
        span  = 140
        seg_d = span / segs

        for i in range(segs):
            f = i / segs
            a_start = -angle - i * seg_d

            alpha = int(220 * (1.0 - f * 0.75))
            r = int(80  + f * 175)
            g = int(140 + f * 60)
            b = 255

            p.setPen(QPen(QColor(r, g, b, alpha), 2.8,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(arc_rect, int(a_start * 16), int(-seg_d * 16))

        # Pulsing centre dot
        pulse = 0.5 + 0.5 * math.sin(self._t * 5.2)
        dr = 2.5 + pulse * 1.8
        da = int(150 + pulse * 100)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(100, 180, 255, da)))
        p.drawEllipse(QPointF(cx, cy), dr, dr)

    # ── Done green glow ────────────────────────────────────────────────────

    def _draw_done_glow(self, p: QPainter, cx, cy, rad):
        prog = self._check_progress
        pulse = 0.5 + 0.5 * math.sin(self._t * 4.0)

        glow = QRadialGradient(QPointF(cx, cy), rad * 0.85)
        glow.setColorAt(0.0, QColor(50, 215, 75, int((60 + 30 * pulse) * prog)))
        glow.setColorAt(0.5, QColor(50, 215, 75, int(25 * prog)))
        glow.setColorAt(1.0, QColor(50, 215, 75, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(glow))

    # ── Animated checkmark ─────────────────────────────────────────────────

    def _draw_checkmark(self, p: QPainter, cx, cy, rad):
        prog = self._check_progress
        if prog <= 0:
            return

        sz = rad * 0.36
        p1 = QPointF(cx - sz * 0.50, cy + sz * 0.05)
        p2 = QPointF(cx - sz * 0.10, cy + sz * 0.45)
        p3 = QPointF(cx + sz * 0.55, cy - sz * 0.45)

        alpha = int(255 * min(1.0, prog * 1.5))
        pen = QPen(QColor(50, 215, 75, alpha), 3.8,
                   Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # First stroke: down-left to bottom
        s1 = min(1.0, prog * 2.0)
        if s1 > 0:
            end1 = QPointF(
                p1.x() + (p2.x() - p1.x()) * s1,
                p1.y() + (p2.y() - p1.y()) * s1,
            )
            p.drawLine(p1, end1)

        # Second stroke: bottom to up-right
        s2 = max(0.0, min(1.0, prog * 2.0 - 1.0))
        if s2 > 0:
            end2 = QPointF(
                p2.x() + (p3.x() - p2.x()) * s2,
                p2.y() + (p3.y() - p2.y()) * s2,
            )
            p.drawLine(p2, end2)


# ── IPC client helper ─────────────────────────────────────────────────────────
def send_command(cmd: str):
    """Send a command to the running HUD (for use from other scripts)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("127.0.0.1", IPC_PORT))
        s.sendall(cmd.encode())
        s.close()
    except Exception as e:
        print(f"HUD send failed: {e}")


# ── Demo ───────────────────────────────────────────────────────────────────────
def _demo(hud: SiriOrbHUD):
    """Cycle through all states so you can see the animations."""
    print("[HUD] 🎬 Demo: listen → process → done")
    QTimer.singleShot(500,   hud.show_listening)
    QTimer.singleShot(5000,  hud.show_processing)
    QTimer.singleShot(8500,  hud.show_done)
    QTimer.singleShot(11000, lambda: (print("[HUD] Demo finished"), sys.exit(0)))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    hud = SiriOrbHUD()

    if "--demo" in sys.argv:
        _demo(hud)
    else:
        print("[HUD] Waiting for IPC commands (listen / process / done / hide)")

    sys.exit(app.exec())
```

---

## Step 5 — Small Fix in `ear.py` (Key Detection)

Some macOS + pynput versions report `Key.cmd` for **both** left and right Command. Replace the hotkey check at the top and the `on_press`/`on_release` methods:

```python
# At the top of ear.py, replace the HOTKEY line:
HOTKEY_PRIMARY   = keyboard.Key.cmd_r    # Right Command
HOTKEY_FALLBACK  = keyboard.Key.cmd      # Fallback if cmd_r not detected

# Then update on_press:
def on_press(self, key):
    if key not in (HOTKEY_PRIMARY, HOTKEY_FALLBACK):
        return
    with self._lock:
        if self.is_recording:
            return
        self.is_recording = True
        self.frames = []
        self.last_rms = 0.0

    print("\r\n" + "─" * 50, flush=True)
    print(f"\r🎙️  RECORDING ({self.active_mic_name}) — release key to stop", flush=True)
    self._send_hud("listen")

    try:
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()
    except Exception as e:
        print(f"\r❌ Mic error: {e}", flush=True)
        with self._lock:
            self.is_recording = False

def on_release(self, key):
    if key not in (HOTKEY_PRIMARY, HOTKEY_FALLBACK):
        return
    # ... rest stays the same
```

---

## Step 6 — Updated `start.sh`

Add the Qt fix at the top:

```bash
#!/usr/bin/env bash
set -euo pipefail

export BACKEND="${BACKEND:-faster_whisper}"
export KMP_DUPLICATE_LIB_OK=TRUE

# ═══ Intel Mac Qt fixes ═══
export QT_MAC_WANTS_LAYER=1

VENV_PYTHON="./.venv/bin/python"

# ... rest of your start.sh stays exactly the same ...
```

---

## Quick Test Flow

After applying all fixes:

```bash
# 1. Test the HUD alone (you should see the orb animate!)
./.venv/bin/python hud.py --demo

# 2. If that works, test with IPC:
# Terminal 1:
./.venv/bin/python hud.py

# Terminal 2:
./.venv/bin/python -c "
import socket, time
for cmd in ['listen', 'process', 'done']:
    s = socket.socket()
    s.connect(('127.0.0.1', 57234))
    s.sendall(cmd.encode())
    s.close()
    print(f'Sent: {cmd}')
    time.sleep(3)
"

# 3. If both work, run the full system:
./start.sh
```

---

## Summary of What Was Wrong

| Symptom | Cause | Fix |
|---------|-------|-----|
| Nothing at all when pressing Cmd | macOS Accessibility not granted | Add Terminal to Accessibility + Input Monitoring |
| Terminal shows "RECORDING" but no orb | HUD crashed or Qt rendering bug | `export QT_MAC_WANTS_LAYER=1` + check `hud.log` |
| Only Left Cmd works, not Right | pynput version issue | Accept both `Key.cmd_r` and `Key.cmd` |
| HUD port bind error | Stale process on port 57234 | `kill $(lsof -ti :57234)` then restart |

The **#1 most likely cause** on a fresh Intel Mac is the accessibility permission. Grant it, restart Terminal, and the Siri orb should appear.