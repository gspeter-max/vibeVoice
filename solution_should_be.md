

# Siri Glass Orb HUD — Complete Rewrite

Looking at your Siri screenshots, the key visual elements you're missing are:

1. **Glass sphere depth** — edge vignette that makes it look 3D
2. **Specular highlight** — the bright white reflection spot on glass
3. **Voice-reactive blobs** — calm when silent, active when speaking
4. **Proper Siri color palette** — overlapping blue/pink/teal/purple

Here's the complete solution:

## `hud.py` — Full Replacement

```python
"""
hud.py — Apple Siri–Style Glass Orb HUD
=========================================
A floating glass sphere with colorful animated blobs inside,
matching Apple Siri's visual design as closely as possible.

Visual behaviour:
  LISTENING + silent  → calm, gently drifting blobs (glass marble at rest)
  LISTENING + voice   → energetic swirling blobs (Siri actively listening)
  PROCESSING          → calming blobs + gradient spinner arc
  DONE                → green internal glow + animated checkmark → fade

IPC:
  TCP 57234 → "listen" | "process" | "done" | "hide"
  UDP 57235 → "vol:0.XXX"  (real-time mic RMS from ear.py)

Test:  python hud.py --demo
"""

import sys
import os
import math
import time
import socket

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QThread, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPainterPath, QPen, QBrush,
    QRadialGradient, QLinearGradient,
)

# ── Layout ─────────────────────────────────────────────────────────────────────
ORB_DIAMETER = 120          # glass sphere size (logical px)
PADDING      = 24           # room for outer glow
WINDOW_SIZE  = ORB_DIAMETER + PADDING * 2   # 168

IPC_PORT = 57234            # TCP commands
VOL_PORT = 57235            # UDP volume data

HIDDEN     = "hidden"
LISTENING  = "listening"
PROCESSING = "processing"
DONE       = "done"


# ── Siri blob config ──────────────────────────────────────────────────────────
class SiriBlob:
    """One colourful sphere that floats inside the glass orb."""
    __slots__ = ("r", "g", "b", "phase", "orbit", "size", "speed")
    def __init__(self, r, g, b, phase, orbit, size, speed):
        self.r, self.g, self.b = r, g, b
        self.phase = phase       # orbital phase offset (radians)
        self.orbit = orbit       # orbit radius  (fraction of orb radius)
        self.size  = size        # blob radius   (fraction of orb radius)
        self.speed = speed       # angular speed multiplier

# Colours taken directly from Apple Siri reference
BLOBS = [
    SiriBlob( 10,  70, 215,  0.00, 0.18, 0.72, 0.50),  # Deep Blue   (large, dominant)
    SiriBlob(225,  38, 115,  1.90, 0.22, 0.62, 0.72),  # Pink/Magenta
    SiriBlob( 30, 195, 210,  3.55, 0.15, 0.56, 0.62),  # Teal/Cyan
    SiriBlob(140,  42, 218,  5.05, 0.20, 0.50, 0.82),  # Purple/Indigo
    SiriBlob( 48, 185, 105,  2.55, 0.16, 0.44, 0.92),  # Green (subtle accent)
]


# ── TCP IPC server thread ─────────────────────────────────────────────────────
class IPCServer(QThread):
    command = Signal(str)

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", IPC_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            print(f"[HUD] TCP IPC on :{IPC_PORT}", flush=True)
            while not self.isInterruptionRequested():
                try:
                    conn, _ = srv.accept()
                    data = conn.recv(256).decode().strip()
                    conn.close()
                    if data:
                        self.command.emit(data)
                except socket.timeout:
                    continue
                except Exception:
                    pass
        except Exception as e:
            print(f"[HUD] ❌ TCP bind failed: {e}", flush=True)
        finally:
            srv.close()


# ── UDP volume listener thread ─────────────────────────────────────────────────
class VolumeListener(QThread):
    volume = Signal(float)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", VOL_PORT))
            sock.settimeout(0.1)
            print(f"[HUD] UDP volume on :{VOL_PORT}", flush=True)
            while not self.isInterruptionRequested():
                try:
                    data, _ = sock.recvfrom(64)
                    txt = data.decode().strip()
                    if txt.startswith("vol:"):
                        self.volume.emit(float(txt[4:]))
                except socket.timeout:
                    continue
                except Exception:
                    pass
        except Exception as e:
            print(f"[HUD] ❌ UDP bind failed: {e}", flush=True)
        finally:
            sock.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SIRI ORB HUD WIDGET
# ══════════════════════════════════════════════════════════════════════════════
class SiriOrbHUD(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)

        # Position: bottom-centre, above Dock
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - WINDOW_SIZE // 2,
            screen.bottom() - WINDOW_SIZE - 30,
        )

        # ── Animation state ───────────────────────────────────────────────
        self._state       = HIDDEN
        self._t0          = time.time()
        self._t           = 0.0
        self._fade        = 0.0
        self._fade_dir    = 0

        # Voice reactivity
        self._voice_raw   = 0.0     # latest from ear.py
        self._voice_smooth = 0.0    # smoothed for animation (fast attack, slow decay)
        self._voice_peak  = 0.0     # peak hold for dramatic effects
        self._last_vol_t  = 0.0     # timestamp of last volume update

        # State-specific
        self._spin_angle  = 0.0
        self._check_prog  = 0.0

        # ── Timer (~60 fps) ───────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        # ── IPC ───────────────────────────────────────────────────────────
        self._ipc = IPCServer(self)
        self._ipc.command.connect(self._on_command)
        self._ipc.start()

        self._vol = VolumeListener(self)
        self._vol.volume.connect(self._on_volume)
        self._vol.start()

        print("[HUD] Siri Orb ready ✓", flush=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def show_listening(self):
        self._enter(LISTENING)

    def show_processing(self):
        self._enter(PROCESSING)

    def show_done(self):
        self._enter(DONE)
        self._check_prog = 0.0
        QTimer.singleShot(1500, self.hide_hud)

    def hide_hud(self):
        self._fade_dir = -1

    # ── IPC handlers ───────────────────────────────────────────────────────

    def _on_command(self, cmd: str):
        c = cmd.strip().lower()
        if   c == "listen":  self.show_listening()
        elif c == "process": self.show_processing()
        elif c == "done":    self.show_done()
        elif c == "hide":    self.hide_hud()

    def _on_volume(self, val: float):
        # Normalise raw RMS to 0–1 range (speech RMS ≈ 0.01–0.15)
        self._voice_raw  = min(1.0, val * 6.0)
        self._last_vol_t = time.time()

    # ── State machine ─────────────────────────────────────────────────────

    def _enter(self, state):
        self._state    = state
        self._t0       = time.time()
        self._t        = 0.0
        self._fade_dir = +1
        if not self.isVisible():
            self._fade = 0.0
            self.setWindowOpacity(0.0)
            self.show()
            self.raise_()
        self._timer.start()

    def _tick(self):
        dt = 0.016
        self._t = time.time() - self._t0

        # ── Fade envelope ──────────────────────────────────────────────
        if self._fade_dir == +1:
            self._fade = min(1.0, self._fade + 0.07)
            self.setWindowOpacity(self._fade)
        elif self._fade_dir == -1:
            self._fade = max(0.0, self._fade - 0.05)
            self.setWindowOpacity(self._fade)
            if self._fade <= 0:
                self._timer.stop()
                self.hide()
                self._state = HIDDEN
                return

        # ── Voice smoothing ────────────────────────────────────────────
        # Decay raw if no updates for 200ms
        if time.time() - self._last_vol_t > 0.2:
            self._voice_raw *= 0.88

        # Fast attack, slow decay
        target = self._voice_raw if self._state == LISTENING else 0.0
        if target > self._voice_smooth:
            self._voice_smooth += (target - self._voice_smooth) * 0.35
        else:
            self._voice_smooth += (target - self._voice_smooth) * 0.06

        # Peak hold
        if self._voice_smooth > self._voice_peak:
            self._voice_peak = self._voice_smooth
        else:
            self._voice_peak *= 0.97

        # ── Per-state update ───────────────────────────────────────────
        if self._state == PROCESSING:
            self._spin_angle += 320 * dt
        elif self._state == DONE:
            self._check_prog = min(1.0, self._t * 2.5)

        self.update()

    # ══════════════════════════════════════════════════════════════════════
    #  RENDERING
    # ══════════════════════════════════════════════════════════════════════

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        ws  = WINDOW_SIZE
        cx  = ws / 2.0
        cy  = ws / 2.0
        rad = ORB_DIAMETER / 2.0

        # 1. Outer glow (behind the sphere)
        self._paint_outer_glow(p, cx, cy, rad)

        # 2. Clip to sphere
        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), rad, rad)
        p.setClipPath(clip)

        # 3. Dark glass base
        p.fillPath(clip, QBrush(QColor(6, 6, 18, 235)))

        # 4. State content
        if self._state in (LISTENING, PROCESSING):
            self._paint_blobs(p, cx, cy, rad)
        if self._state == PROCESSING:
            self._paint_spinner(p, cx, cy, rad)
        if self._state == DONE:
            self._paint_done_glow(p, cx, cy, rad)

        # 5. Vignette (edge darkening → 3D sphere depth)
        self._paint_vignette(p, cx, cy, rad)

        # 6. Specular highlight (glass reflection)
        self._paint_specular(p, cx, cy, rad)

        # 7. Done checkmark (on top of everything inside sphere)
        if self._state == DONE:
            self._paint_checkmark(p, cx, cy, rad)

        # 8. Glass rim (outside clip)
        p.setClipping(False)
        self._paint_rim(p, cx, cy, rad)

        p.end()

    # ── 1. Outer glow ─────────────────────────────────────────────────────

    def _paint_outer_glow(self, p: QPainter, cx, cy, rad):
        if self._state == HIDDEN:
            return

        v = self._voice_smooth
        pulse = 0.5 + 0.5 * math.sin(self._t * 2.0)

        if self._state == LISTENING:
            # Glow colour shifts with voice level
            r = int(15 + 25 * v)
            g = int(60 + 50 * v)
            b = int(200 + 30 * v)
        elif self._state == PROCESSING:
            r, g, b = 100, 55, 240
        elif self._state == DONE:
            r, g, b = 50, 215, 75
        else:
            r, g, b = 40, 40, 80

        # Glow intensity scales with voice
        base_alpha = 12 + 28 * max(v, 0.15)
        a = int((base_alpha + 18 * pulse * max(v, 0.2)))

        glow_r = rad + PADDING + 4 * v
        glow = QRadialGradient(QPointF(cx, cy), glow_r)
        glow.setColorAt(0.45, QColor(r, g, b, a))
        glow.setColorAt(0.70, QColor(r, g, b, a // 2))
        glow.setColorAt(1.0,  QColor(r, g, b, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(glow))

    # ── 4. Colour blobs (the Siri effect) ──────────────────────────────────

    def _paint_blobs(self, p: QPainter, cx, cy, rad):
        t = self._t
        v = self._voice_smooth       # 0 = silent, 1 = loud
        pk = self._voice_peak

        # Base idle animation (always present, very subtle)
        idle_speed = 0.18
        idle_orbit = 0.06
        idle_size  = 0.55

        for blob in BLOBS:
            # ── Orbital motion ─────────────────────────────────────────
            # Idle: very slow drift, tiny orbit
            # Speaking: faster, wider orbit
            speed = idle_speed + v * blob.speed * 1.8
            orbit_r = (idle_orbit + v * blob.orbit * 0.85) * rad

            angle = (t * speed
                     + blob.phase
                     + 0.4 * math.sin(t * 0.6 + blob.phase * 2.0) * (0.3 + v))

            # Slightly elliptical orbit for organic feel
            aspect = 0.80 + 0.15 * math.sin(t * 0.35 + blob.phase)
            bx = cx + math.cos(angle) * orbit_r
            by = cy + math.sin(angle) * orbit_r * aspect

            # Add secondary wobble when speaking
            if v > 0.05:
                wobble = v * 4.0
                bx += math.sin(t * 2.3 + blob.phase * 1.7) * wobble
                by += math.cos(t * 1.9 + blob.phase * 2.3) * wobble * 0.7

            # ── Blob size ──────────────────────────────────────────────
            base_sz = (idle_size + v * (blob.size - idle_size + 0.15)) * rad

            # Breathing: gentle when idle, dramatic when speaking
            breath_amp = 0.04 + v * 0.20
            breath = 1.0 + breath_amp * math.sin(
                t * (1.2 + v * 2.5) + blob.phase * 1.3)

            # Peak-reactive pop
            pop = 1.0 + pk * 0.12

            sz = max(6.0, base_sz * breath * pop)

            # ── Alpha envelope ─────────────────────────────────────────
            # Idle: softer, more transparent
            # Speaking: vivid, more opaque
            a_core = int(100 + 120 * max(v, 0.25))
            a_mid  = int(50  + 60  * max(v, 0.15))
            a_edge = int(15  + 25  * v)

            grad = QRadialGradient(QPointF(bx, by), sz)
            grad.setColorAt(0.00, QColor(blob.r, blob.g, blob.b, a_core))
            grad.setColorAt(0.25, QColor(blob.r, blob.g, blob.b, a_mid))
            grad.setColorAt(0.55, QColor(blob.r, blob.g, blob.b, a_edge))
            grad.setColorAt(1.00, QColor(blob.r, blob.g, blob.b, 0))
            p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(grad))

        # ── Central core glow ──────────────────────────────────────────
        # White/blue hot centre — brighter when speaking
        core_pulse = 0.5 + 0.5 * math.sin(t * 3.2)
        core_energy = 0.25 + 0.75 * max(v, 0.1)
        core_sz = max(2.0, (8 + 14 * core_energy * core_pulse))
        core_a  = int((25 + 65 * core_energy) * core_pulse)

        core = QRadialGradient(QPointF(cx, cy), core_sz)
        core.setColorAt(0.0, QColor(220, 230, 255, core_a))
        core.setColorAt(0.3, QColor(180, 200, 255, core_a // 3))
        core.setColorAt(1.0, QColor(140, 170, 255, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(core))

    # ── 5. Vignette — the secret to the 3D glass look ─────────────────────

    def _paint_vignette(self, p: QPainter, cx, cy, rad):
        vig = QRadialGradient(QPointF(cx, cy), rad)
        vig.setColorAt(0.00, QColor(0, 0, 0, 0))
        vig.setColorAt(0.45, QColor(0, 0, 0, 0))
        vig.setColorAt(0.72, QColor(2, 2, 12, 80))
        vig.setColorAt(0.88, QColor(2, 2, 12, 160))
        vig.setColorAt(1.00, QColor(2, 2, 12, 210))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(vig))

    # ── 6. Specular highlight — glass surface reflection ───────────────────

    def _paint_specular(self, p: QPainter, cx, cy, rad):
        # Main highlight: upper-left area (like light from above)
        hx = cx - rad * 0.18
        hy = cy - rad * 0.32

        # Slight wobble based on voice (light shifts with internal energy)
        v = self._voice_smooth
        hx += math.sin(self._t * 0.7) * 1.5 * v
        hy += math.cos(self._t * 0.5) * 1.0 * v

        h_size = rad * 0.48

        highlight = QRadialGradient(QPointF(hx, hy), h_size)
        highlight.setColorAt(0.00, QColor(255, 255, 255, 115))
        highlight.setColorAt(0.18, QColor(255, 255, 255, 55))
        highlight.setColorAt(0.45, QColor(220, 235, 255, 15))
        highlight.setColorAt(1.00, QColor(200, 220, 255, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(highlight))

        # Small sharp reflection dot (the "sparkle")
        dot_x = hx + rad * 0.05
        dot_y = hy + rad * 0.02
        dot = QRadialGradient(QPointF(dot_x, dot_y), rad * 0.10)
        dot.setColorAt(0.0, QColor(255, 255, 255, 180))
        dot.setColorAt(0.4, QColor(255, 255, 255, 40))
        dot.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(dot))

        # Subtle bottom-right secondary reflection
        h2x = cx + rad * 0.28
        h2y = cy + rad * 0.35
        h2 = QRadialGradient(QPointF(h2x, h2y), rad * 0.22)
        h2.setColorAt(0.0, QColor(255, 255, 255, 18))
        h2.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(h2))

    # ── 8. Glass rim ───────────────────────────────────────────────────────

    def _paint_rim(self, p: QPainter, cx, cy, rad):
        # Colour-shifting rim during listening
        if self._state == LISTENING:
            v = self._voice_smooth
            hue = (self._t * 22 + v * 40) % 360
            rim_c = QColor.fromHsvF(hue / 360.0, 0.35 + 0.2 * v, 1.0)
            rim_c.setAlpha(int(35 + 40 * v))
        elif self._state == DONE:
            prog = self._check_prog
            rim_c = QColor(50, 215, 75, int(50 * prog))
        elif self._state == PROCESSING:
            rim_c = QColor(150, 130, 255, 45)
        else:
            rim_c = QColor(255, 255, 255, 30)

        p.setPen(QPen(rim_c, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rad - 0.5, rad - 0.5)

        # Top rim highlight (brighter arc at the top for glass lip)
        highlight_pen = QPen(QColor(255, 255, 255, 35), 1.0,
                             Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(highlight_pen)
        arc_rect = QRectF(cx - rad + 0.5, cy - rad + 0.5,
                          (rad - 0.5) * 2, (rad - 0.5) * 2)
        p.drawArc(arc_rect, 30 * 16, 120 * 16)  # top arc

    # ── Processing spinner ─────────────────────────────────────────────────

    def _paint_spinner(self, p: QPainter, cx, cy, rad):
        arc_r = rad * 0.82
        arc_rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)

        # Background track
        p.setPen(QPen(QColor(255, 255, 255, 14), 2.5,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), arc_r, arc_r)

        # Gradient arc
        segs  = 28
        span  = 150
        seg_d = span / segs
        angle = self._spin_angle

        for i in range(segs):
            f = i / segs
            a_start = -angle - i * seg_d
            alpha = int(210 * (1.0 - f * 0.72))
            r = int(70  + f * 185)
            g = int(130 + f * 80)
            b = 255
            p.setPen(QPen(QColor(r, g, b, alpha), 2.8,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(arc_rect, int(a_start * 16), int(-seg_d * 16))

        # Pulsing centre dot
        pulse = 0.5 + 0.5 * math.sin(self._t * 5.0)
        dr = 2.8 + pulse * 1.6
        da = int(140 + pulse * 110)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(90, 170, 255, da)))
        p.drawEllipse(QPointF(cx, cy), dr, dr)

    # ── Done glow ──────────────────────────────────────────────────────────

    def _paint_done_glow(self, p: QPainter, cx, cy, rad):
        prog = self._check_prog
        pulse = 0.5 + 0.5 * math.sin(self._t * 4.0)

        glow = QRadialGradient(QPointF(cx, cy), rad * 0.80)
        glow.setColorAt(0.0, QColor(50, 215, 75, int((65 + 35 * pulse) * prog)))
        glow.setColorAt(0.4, QColor(50, 215, 75, int(30 * prog)))
        glow.setColorAt(1.0, QColor(50, 215, 75, 0))
        p.fillRect(QRectF(0, 0, WINDOW_SIZE, WINDOW_SIZE), QBrush(glow))

    def _paint_checkmark(self, p: QPainter, cx, cy, rad):
        prog = self._check_prog
        if prog <= 0:
            return

        sz = rad * 0.34
        p1 = QPointF(cx - sz * 0.50, cy + sz * 0.05)
        p2 = QPointF(cx - sz * 0.08, cy + sz * 0.45)
        p3 = QPointF(cx + sz * 0.55, cy - sz * 0.42)

        alpha = int(255 * min(1.0, prog * 1.5))
        p.setPen(QPen(QColor(50, 215, 75, alpha), 3.6,
                      Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)

        s1 = min(1.0, prog * 2.0)
        if s1 > 0:
            e1 = QPointF(p1.x() + (p2.x() - p1.x()) * s1,
                         p1.y() + (p2.y() - p1.y()) * s1)
            p.drawLine(p1, e1)

        s2 = max(0.0, min(1.0, prog * 2.0 - 1.0))
        if s2 > 0:
            e2 = QPointF(p2.x() + (p3.x() - p2.x()) * s2,
                         p2.y() + (p3.y() - p2.y()) * s2)
            p.drawLine(p2, e2)


# ── IPC client helper ──────────────────────────────────────────────────────────
def send_command(cmd: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("127.0.0.1", IPC_PORT))
        s.sendall(cmd.encode())
        s.close()
    except Exception as e:
        print(f"HUD send failed: {e}")


# ── Demo with simulated voice ─────────────────────────────────────────────────
def _demo(hud: SiriOrbHUD):
    """
    Simulates the full cycle:
      idle (calm orb) → speaking (active blobs) → pause → speaking → process → done
    """
    print("[HUD] 🎬 Demo starting...", flush=True)

    demo_start = [time.time()]

    def voice_pulse():
        t = time.time() - demo_start[0]
        if t < 1.5:
            # Phase 1: idle — very low voice (calm orb)
            hud._voice_raw = 0.02
        elif t < 4.0:
            # Phase 2: speaking — dynamic voice
            hud._voice_raw = (0.4
                + 0.25 * math.sin(t * 3.5)
                + 0.15 * math.sin(t * 8.1)
                + 0.10 * math.sin(t * 13.3))
            hud._voice_raw = max(0.0, min(1.0, hud._voice_raw))
            hud._last_vol_t = time.time()
        elif t < 5.0:
            # Phase 3: pause between sentences
            hud._voice_raw = 0.03
            hud._last_vol_t = time.time()
        elif t < 7.5:
            # Phase 4: speaking again
            hud._voice_raw = (0.35
                + 0.30 * math.sin(t * 4.2)
                + 0.20 * math.sin(t * 9.7))
            hud._voice_raw = max(0.0, min(1.0, hud._voice_raw))
            hud._last_vol_t = time.time()
        else:
            voice_timer.stop()

    voice_timer = QTimer()
    voice_timer.setInterval(33)   # ~30 fps voice updates
    voice_timer.timeout.connect(voice_pulse)

    # Timeline
    QTimer.singleShot(300,   hud.show_listening)
    QTimer.singleShot(350,   voice_timer.start)
    QTimer.singleShot(8500,  hud.show_processing)
    QTimer.singleShot(11500, hud.show_done)
    QTimer.singleShot(14000, lambda: (print("[HUD] Demo finished"), sys.exit(0)))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    hud = SiriOrbHUD()

    if "--demo" in sys.argv:
        _demo(hud)
    else:
        print("[HUD] Waiting for IPC: listen / process / done / hide", flush=True)

    sys.exit(app.exec())
```

---

## `ear.py` — Add Volume Sending

Add these changes to your existing `ear.py`:

```python
# ─── At the top of ear.py, add this import ────────────────────────────────────
import socket as sock_module   # rename to avoid clash with existing 'socket' usage

# ─── Add this constant near the other config ─────────────────────────────────
VOL_PORT = 57235   # UDP port for sending mic volume to HUD


# ─── Add this NEW METHOD to the Ear class ─────────────────────────────────────
class Ear:
    # ... (keep all existing code) ...

    def _start_volume_sender(self):
        """
        Background thread that sends mic RMS level to the HUD via UDP.
        Runs only while recording. Lightweight, ~25 updates/sec.
        """
        udp = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_DGRAM)

        def _sender():
            while True:
                with self._lock:
                    if not self.is_recording:
                        break
                    rms = self.last_rms
                try:
                    udp.sendto(
                        f"vol:{rms:.4f}".encode(),
                        ('127.0.0.1', VOL_PORT)
                    )
                except Exception:
                    pass
                time.sleep(0.04)
            udp.close()

        threading.Thread(target=_sender, daemon=True).start()


    # ─── MODIFY on_press: add one line after self.stream.start_stream() ───
    def on_press(self, key):
        if key != HOTKEY:
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
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()

            # ★ NEW: start sending volume levels to HUD
            self._start_volume_sender()

        except Exception as e:
            print(f"\r❌ Mic error: {e}", flush=True)
            with self._lock:
                self.is_recording = False
```

---

## `start.sh` — Add Qt Fix

Add this line near the top of `start.sh` (after `export BACKEND=...`):

```bash
# Fix Qt rendering on Intel Macs
export QT_MAC_WANTS_LAYER=1
```

---

## Test It

```bash
# Step 1: Test HUD alone (see the full animation cycle)
./.venv/bin/python hud.py --demo

# Step 2: If demo works, run the full system
./start.sh
```

### What You'll See

| State | Visual |
|-------|--------|
| **Press CMD, silent** | Dark glass orb with barely-moving color blobs — calm Siri marble |
| **Start speaking** | Blobs orbit wider, colors brighten, size pulses with your voice |
| **Speak louder** | Even more dramatic swirl, outer glow intensifies |
| **Release CMD** | Blobs calm down, gradient spinner arc appears over orb |
| **Transcription done** | Green internal glow + animated checkmark → fade out |

### Visual Stack (what creates the Siri look)

```
┌─ Glass rim (thin bright arc at top)
│  ┌─ Specular highlight (white reflection dot — sells the 3D glass)
│  │  ┌─ Vignette (dark edges — creates curved surface illusion)
│  │  │  ┌─ Color blobs (blue/pink/teal/purple — the Siri colors)
│  │  │  │  ┌─ Dark base (nearly black, blue-tinted)
│  │  │  │  │
│  │  │  │  └──── fills the circle
│  │  │  └─────── 5 radial gradients, voice-reactive position/size
│  │  └────────── radial gradient: transparent center → dark edges
│  └───────────── white radial gradient in upper-left area
└──────────────── thin colored circle + bright top arc
```

The **vignette** and **specular highlight** are what your original code was missing — they're what make a flat circle look like a 3D glass sphere.