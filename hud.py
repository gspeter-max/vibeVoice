"""
hud.py — iOS-style Pill Waveform HUD (macOS visibility fixed)
===============================================================
Dark rounded-rectangle capsule with animated white vertical bars.

States:
  - hidden     : small idle pill outline (always on-screen)
  - listening  : pill EXPANDS, bars animate with real mic volume
  - processing : bars do a slow sweep/pulse
  - done       : brief flash then back to idle outline

IPC:
  TCP 57234 → "listen" | "process" | "done" | "hide"
  UDP 57235 → "vol:0.XXX"

Test:  python hud.py --demo
"""

import sys
import os
import math
import time
import socket

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QRectF, QThread, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPainterPath, QPen, QBrush,
)

# ── Pill dimensions ───────────────────────────────────────────────────────────
PILL_W_IDLE   = 60     # was 80
PILL_H_IDLE   = 20     # was 26
PILL_W_ACTIVE = 120    # was 160
PILL_H_ACTIVE = 32     # was 44

PADDING  = 20
WINDOW_W = PILL_W_ACTIVE + PADDING * 2
WINDOW_H = PILL_H_ACTIVE + PADDING * 2

# ── Bar config ────────────────────────────────────────────────────────────────
NUM_BARS  = 7
BAR_W     = 2.8
BAR_GAP   = 7.0      # centre-to-centre spacing
BAR_MAX_H = PILL_H_ACTIVE * 0.58
BAR_MIN_H = PILL_H_ACTIVE * 0.10
BAR_R     = 1.5      # rounded tip

IPC_PORT = 57234
VOL_PORT = 57235

HIDDEN     = "hidden"
LISTENING  = "listening"
PROCESSING = "processing"
DONE       = "done"


class IPCServer(QThread):
    command = Signal(str)

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", IPC_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            print(f"[HUD] TCP on :{IPC_PORT}", flush=True)
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
            print(f"[HUD] TCP failed: {e}", flush=True)
        finally:
            srv.close()


class VolumeListener(QThread):
    volume = Signal(float)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", VOL_PORT))
            sock.settimeout(0.1)
            print(f"[HUD] UDP on :{VOL_PORT}", flush=True)
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
            print(f"[HUD] UDP failed: {e}", flush=True)
        finally:
            sock.close()


# ── Main HUD widget ────────────────────────────────────────────────────────
class PillHUD(QWidget):

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

        # ★ FIX 1 — THE KEY FIX
        # macOS hides ALL Tool windows the moment your app loses focus
        # (i.e. you click on Safari, VS Code, etc.). This attribute
        # tells macOS: "keep this Tool window visible even when the
        # app is inactive."  Without this one line, the pill vanishes
        # as soon as you interact with anything else.
        try:
            self.setAttribute(
                Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True
            )
            print("[HUD] WA_MacAlwaysShowToolWindow ✓", flush=True)
        except AttributeError:
            print("[HUD] WA_MacAlwaysShowToolWindow not available — "
                  "using keepalive fallback", flush=True)

        self.setFixedSize(WINDOW_W, WINDOW_H)

        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - WINDOW_W // 2,
            screen.bottom() - WINDOW_H - 4,   # was - 20
        )

        self._state        = HIDDEN
        self._t0           = time.time()
        self._t            = 0.0
        self._fade         = 1.0
        self._fade_dir     = 0
        self._voice_raw    = 0.0
        self._voice_smooth = 0.0
        self._last_vol_t   = 0.0

        # Per-bar state
        self._bar_h     = [BAR_MIN_H] * NUM_BARS
        self._bar_phase = [i * 0.72 for i in range(NUM_BARS)]

        self._cur_w = float(PILL_W_IDLE)
        self._cur_h = float(PILL_H_IDLE)

        # 60 fps animation timer (only runs when animating)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        # IPC threads
        self._ipc = IPCServer(self)
        self._ipc.command.connect(self._on_command)
        self._ipc.start()

        self._vol = VolumeListener(self)
        self._vol.volume.connect(self._on_volume)
        self._vol.start()

        # ★ FIX 2 — KEEPALIVE TIMER (safety net)
        # Every 3 seconds, re-assert visibility.  If macOS somehow
        # hid the window (Exposé, space switch, etc.) this brings
        # it back.  Cheap — just a show()+raise_(), no repainting.
        self._keepalive = QTimer(self)
        self._keepalive.setInterval(3000)
        self._keepalive.timeout.connect(self._ensure_visible)
        self._keepalive.start()

        # Initial show
        self.setWindowOpacity(1.0)
        self.show()

        # ★ FIX 4 — NATIVE macOS WINDOW LEVEL (optional, best fix)
        # If pyobjc is installed, set the Cocoa window level to
        # NSFloatingWindowLevel so it truly floats above everything,
        # and join all Spaces so it's visible on every desktop.
        QTimer.singleShot(300, self._set_native_level)

        print("[HUD] Pill HUD ready ✓", flush=True)

    # ── Visibility helpers ─────────────────────────────────────────────────
    def _ensure_visible(self):
        """Re-assert visibility WITHOUT stealing focus."""
        if not self.isVisible():
            self.show()
        self.setWindowOpacity(1.0)

    def _set_native_level(self):
        """Use macOS native API for bulletproof always-on-top (needs pyobjc)."""
        try:
            from AppKit import NSApp, NSFloatingWindowLevel
            for w in NSApp.windows():
                w.setLevel_(NSFloatingWindowLevel + 2)
                # Show on ALL Spaces / desktops
                w.setCollectionBehavior_(
                    w.collectionBehavior()
                    | (1 << 0)   # canJoinAllSpaces
                    | (1 << 4)   # moveToActiveSpace
                )
            print("[HUD] Native macOS window level set ✓", flush=True)
        except ImportError:
            # pyobjc not installed — Qt flags + keepalive still work fine
            pass
        except Exception as e:
            print(f"[HUD] Native level failed (non-critical): {e}", flush=True)

    # ── Public state transitions ───────────────────────────────────────────
    def show_listening(self):
        self._enter(LISTENING)

    def show_processing(self):
        self._enter(PROCESSING)

    def show_done(self):
        self._enter(DONE)
        QTimer.singleShot(900, self._return_to_idle)

    def hide_hud(self):
        """Shrink back to idle outline (pill stays visible, never fully hidden)."""
        self._return_to_idle()

    def _return_to_idle(self):
        self._state    = HIDDEN
        self._fade_dir = 0
        self._bar_h    = [BAR_MIN_H] * NUM_BARS
        # Start timer so the shrink animation plays smoothly
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    # ── IPC dispatcher ────────────────────────────────────────────────────
    def _on_command(self, cmd):
        c = cmd.strip().lower()
        print(f"[HUD] ← {c}", flush=True)
        if   c == "listen":  self.show_listening()
        elif c == "process": self.show_processing()
        elif c == "done":    self.show_done()
        elif c == "hide":    self.hide_hud()

    def _on_volume(self, val):
        self._voice_raw  = min(1.0, val * 6.0)
        self._last_vol_t = time.time()

    def _enter(self, state):
        self._state    = state
        self._t0       = time.time()
        self._t        = 0.0
        self._fade_dir = 0
        self.setWindowOpacity(1.0)
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self):
        self._t = time.time() - self._t0

        # Animate pill size (smooth lerp)
        target_w = PILL_W_ACTIVE if self._state == LISTENING else PILL_W_IDLE
        target_h = PILL_H_ACTIVE if self._state == LISTENING else PILL_H_IDLE

        self._cur_w += (target_w - self._cur_w) * 0.18
        self._cur_h += (target_h - self._cur_h) * 0.18

        # Voice decay
        if time.time() - self._last_vol_t > 0.15:
            self._voice_raw *= 0.80

        target = self._voice_raw if self._state == LISTENING else 0.0
        spd    = 0.38 if target > self._voice_smooth else 0.08
        self._voice_smooth += (target - self._voice_smooth) * spd

        v   = self._voice_smooth
        t   = self._t
        mid = (NUM_BARS - 1) / 2.0

        for i in range(NUM_BARS):
            ph = self._bar_phase[i]
            centre_w = 1.0 - abs(i - mid) / mid * 0.35

            if self._state == LISTENING:
                wave = (
                    0.50 * math.sin(t * 7.0  + ph) +
                    0.30 * math.sin(t * 12.5 + ph * 1.8) +
                    0.20 * math.sin(t * 19.0 + ph * 0.8)
                )
                wave  = (wave + 1.0) / 2.0
                idle  = BAR_MIN_H + (BAR_MAX_H - BAR_MIN_H) * 0.15
                tgt   = idle + (BAR_MAX_H * wave * centre_w - idle) * min(1.0, v * 2.2)

            elif self._state == PROCESSING:
                wave = 0.5 + 0.5 * math.sin(t * 3.2 - i * 0.58)
                tgt  = BAR_MIN_H + (BAR_MAX_H * 0.45 - BAR_MIN_H) * wave

            elif self._state == DONE:
                prog = min(1.0, self._t * 5.0)
                tgt  = BAR_MAX_H * (1.0 - prog) + BAR_MIN_H * prog

            else:
                tgt = BAR_MIN_H

            cur = self._bar_h[i]
            self._bar_h[i] += (tgt - cur) * (0.42 if tgt > cur else 0.10)

        self.update()

        # Stop animation timer once idle and shrink is done (saves CPU)
        if self._state == HIDDEN and abs(self._cur_w - PILL_W_IDLE) < 0.3:
            self._timer.stop()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        pw = self._cur_w
        ph = self._cur_h
        px = (WINDOW_W - pw) / 2
        py = (WINDOW_H - ph) / 2
        cx = WINDOW_W / 2
        cy = WINDOW_H / 2
        r  = ph / 2

        # Pill shape
        pill = QPainterPath()
        pill.addRoundedRect(QRectF(px, py, pw, ph), r, r)

        # ★ FIX 5 — SUBTLE IDLE FILL
        # Old code used fill_alpha=0 for idle → completely transparent.
        # macOS can optimize away fully-transparent window regions.
        # alpha=50 gives a subtle dark tint so the pill is always
        # visible and the compositor never discards it.
        fill_alpha = 50 if self._state == HIDDEN else 240
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(16, 16, 18, fill_alpha)))
        p.drawPath(pill)

        # Border — always visible
        p.setPen(QPen(QColor(90, 90, 95, 200), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(pill)

        # Bars — only when not idle
        if self._state != HIDDEN:
            p.setClipPath(pill)

            total_w = (NUM_BARS - 1) * BAR_GAP + BAR_W
            start_x = cx - total_w / 2

            for i in range(NUM_BARS):
                bx = start_x + i * BAR_GAP
                bh = max(BAR_MIN_H, self._bar_h[i])
                by = cy - bh / 2

                mid_val = (NUM_BARS - 1) / 2.0
                cf    = 1.0 - abs(i - mid_val) / mid_val * 0.18
                alpha = int(225 * cf)

                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(255, 255, 255, alpha)))
                p.drawRoundedRect(
                    QRectF(bx - BAR_W / 2, by, BAR_W, bh), BAR_R, BAR_R
                )

            p.setClipping(False)

        p.end()


def send_command(cmd: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("127.0.0.1", IPC_PORT))
        s.sendall(cmd.encode())
        s.close()
    except Exception as e:
        print(f"HUD send failed: {e}")


def _demo(hud: PillHUD):
    print("[HUD] Demo starting...", flush=True)
    t0 = [time.time()]

    def voice_pulse():
        t = time.time() - t0[0]
        if t < 1.0:
            hud._voice_raw = 0.02
        elif t < 5.5:
            hud._voice_raw  = max(0, min(1,
                0.55 + 0.28 * math.sin(t * 4.5) +
                0.18 * math.sin(t * 9.8) +
                0.10 * math.sin(t * 17.0)))
            hud._last_vol_t = time.time()
        elif t < 6.5:
            hud._voice_raw = 0.02
        elif t < 10.0:
            hud._voice_raw  = max(0, min(1,
                0.45 + 0.32 * math.sin(t * 5.5) +
                0.20 * math.sin(t * 11.5)))
            hud._last_vol_t = time.time()
        else:
            vt.stop()

    vt = QTimer()
    vt.setInterval(30)
    vt.timeout.connect(voice_pulse)

    QTimer.singleShot(300,   hud.show_listening)
    QTimer.singleShot(350,   vt.start)
    QTimer.singleShot(6000,  hud.hide_hud)
    QTimer.singleShot(7500,  hud.show_listening)
    QTimer.singleShot(11000, hud.show_processing)
    QTimer.singleShot(12500, hud.show_done)
    QTimer.singleShot(14000, lambda: (print("[HUD] Demo done."), sys.exit(0)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # ★ FIX 6 — HIDE DOCK ICON
    # Without this, running hud.py spawns a Python icon in the Dock.
    # Setting activation policy to "Accessory" removes it.
    try:
        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        print("[HUD] Dock icon hidden ✓", flush=True)
    except ImportError:
        pass  # pyobjc not installed — dock icon will show (cosmetic only)
    except Exception:
        pass

    hud = PillHUD()

    if "--demo" in sys.argv:
        _demo(hud)
    else:
        print("[HUD] Waiting for IPC: listen / process / done / hide", flush=True)

    sys.exit(app.exec())