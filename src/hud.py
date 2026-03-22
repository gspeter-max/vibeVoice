"""
hud.py — iOS-style Pill Waveform HUD (macOS visibility fixed)
===============================================================
Dark rounded-rectangle capsule with animated white vertical bars.

States:
  - hidden     : small idle pill outline (always on-screen)
  - listening  : pill EXPANDS, bars animate with real mic volume
  - thinking   : bars turn BLUE, indicates VAD detected end-of-speech
  - processing : bars do a slow sweep/pulse (White)
  - done       : brief flash then back to idle outline

IPC:
  TCP 57234 → "listen" | "thinking" | "process" | "done" | "hide"
  UDP 57235 → "vol:0.XXX"

Test:  python hud.py --demo
"""

import sys
import os
import math
import time
import socket
import subprocess

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
THINKING   = "thinking"
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


# ── Sound effects (using provided external files) ─────────────────────────────
def _init_sounds():
    # Use repo root as base for assets/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    listen_path = os.path.join(base_dir, "assets", "ui-alert-synth-beep-epic-stock-media-1-00-00.mp3")
    done_path   = os.path.join(base_dir, "assets", "mixkit-tile-game-reveal-960.wav")
    return listen_path, done_path


def _play_sound(path):
    """Play a sound file without blocking, without stealing focus."""
    if not os.path.exists(path):
        return
    try:
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


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
            screen.bottom() - WINDOW_H - 4,
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
        self._snd_listen, self._snd_done = _init_sounds()

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

        # Keepalive timer
        self._keepalive = QTimer(self)
        self._keepalive.setInterval(3000)
        self._keepalive.timeout.connect(self._ensure_visible)
        self._keepalive.start()

        # Initial show
        self.setWindowOpacity(1.0)
        self.show()

        # Native macOS window level
        QTimer.singleShot(300, self._set_native_level)

        print("[HUD] Pill HUD ready ✓", flush=True)

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
            pass
        except Exception as e:
            print(f"[HUD] Native level failed (non-critical): {e}", flush=True)

    # ── Public state transitions ───────────────────────────────────────────
    def show_listening(self):
        _play_sound(self._snd_listen)
        self._enter(LISTENING)

    def show_thinking(self):
        self._enter(THINKING)

    def show_processing(self):
        self._enter(PROCESSING)

    def show_done(self):
        _play_sound(self._snd_done)
        self._enter(DONE)
        QTimer.singleShot(900, self._return_to_idle)

    def hide_hud(self):
        self._return_to_idle()

    def _return_to_idle(self):
        self._state    = HIDDEN
        self._fade_dir = 0
        self._bar_h    = [BAR_MIN_H] * NUM_BARS
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    # ── IPC dispatcher ────────────────────────────────────────────────────
    def _on_command(self, cmd):
        c = cmd.strip().lower()
        print(f"[HUD] ← {c}", flush=True)
        if   c == "listen":    self.show_listening()
        elif c == "thinking":  self.show_thinking()
        elif c == "process":   self.show_processing()
        elif c == "done":      self.show_done()
        elif c == "hide":      self.hide_hud()

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
        target_w = PILL_W_ACTIVE if self._state in (LISTENING, THINKING, PROCESSING) else PILL_W_IDLE
        target_h = PILL_H_ACTIVE if self._state in (LISTENING, THINKING, PROCESSING) else PILL_H_IDLE

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

            elif self._state == THINKING:
                # Oscillating blue bars
                wave = (math.sin(t * 5.0 + i * 0.8) + 1.0) / 2.0
                tgt  = BAR_MIN_H + (BAR_MAX_H * 0.5) * wave

            elif self._state == PROCESSING:
                sweep = (math.sin(t * 2.8) + 1.0) / 2.0 * (NUM_BARS - 1)
                dist  = abs(i - sweep)
                glow  = math.exp(-dist * dist * 0.7)
                breath = 0.12 + 0.06 * math.sin(t * 1.6 + i * 0.5)
                tgt = BAR_MIN_H + (BAR_MAX_H * 0.72) * max(glow, breath)

            elif self._state == DONE:
                prog = min(1.0, self._t * 5.0)
                tgt  = BAR_MAX_H * (1.0 - prog) + BAR_MIN_H * prog

            else:
                tgt = BAR_MIN_H

            cur = self._bar_h[i]
            self._bar_h[i] += (tgt - cur) * (0.42 if tgt > cur else 0.10)

        self.update()

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

        pill = QPainterPath()
        pill.addRoundedRect(QRectF(px, py, pw, ph), r, r)

        fill_alpha = 50 if self._state == HIDDEN else 240
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(16, 16, 18, fill_alpha)))
        p.drawPath(pill)

        p.setPen(QPen(QColor(90, 90, 95, 200), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(pill)

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

                if self._state == THINKING:
                    color = QColor(64, 156, 255, int(225 * cf)) # Brighter blue
                elif self._state == PROCESSING:
                    sweep = (math.sin(self._t * 2.8) + 1.0) / 2.0 * (NUM_BARS - 1)
                    glow_cf = math.exp(-abs(i - sweep) ** 2 * 0.7)
                    color = QColor(255, 255, 255, int((160 + 95 * glow_cf) * cf))
                else:
                    color = QColor(255, 255, 255, int(225 * cf))

                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color))
                p.drawRoundedRect(
                    QRectF(bx - BAR_W / 2, by, BAR_W, bh), BAR_R, BAR_R
                )

            p.setClipping(False)
        p.end()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    try:
        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    hud = PillHUD()
    sys.exit(app.exec())
