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

from src.theme_manager import ThemeManager, THEME_ORIGINAL

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
        print(f"[HUD] 🚀 IPCServer thread starting...", flush=True)
        srv = None
        try:
            print(f"[HUD] 🔧 Creating TCP socket...", flush=True)
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            print(f"[HUD] 🔧 Binding to 127.0.0.1:{IPC_PORT}...", flush=True)
            srv.bind(("127.0.0.1", IPC_PORT))

            print(f"[HUD] 🔧 Setting listen queue...", flush=True)
            srv.listen(5)

            print(f"[HUD] 🔧 Setting socket timeout...", flush=True)
            srv.settimeout(1.0)

            print(f"[HUD] 🌐 TCP server listening on :{IPC_PORT}", flush=True)

            while not self.isInterruptionRequested():
                try:
                    conn, addr = srv.accept()
                    print(f"[HUD] 🔌 Connection accepted from {addr}", flush=True)
                    data = conn.recv(256).decode().strip()
                    print(f"[HUD] 📨 Received data: '{data}'", flush=True)
                    conn.close()
                    if data:
                        print(f"[HUD] 📢 Emitting command signal: '{data}'", flush=True)
                        self.command.emit(data)
                    else:
                        print(f"[HUD] ⚠️ Empty data received", flush=True)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[HUD] ❌ Error in receive loop: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[HUD] ❌ TCP server FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            if srv:
                print(f"[HUD] 🔧 Closing TCP socket...", flush=True)
                srv.close()
            print(f"[HUD] 🛑 IPCServer thread terminated", flush=True)


class VolumeListener(QThread):
    volume = Signal(float)
    frequency_bands = Signal(dict)  # New signal for frequency data

    def run(self):
        print(f"[HUD] 🚀 VolumeListener thread starting...", flush=True)
        sock = None
        try:
            print(f"[HUD] 🔧 Creating UDP socket...", flush=True)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            print(f"[HUD] 🔧 Binding to 127.0.0.1:{VOL_PORT}...", flush=True)
            sock.bind(("127.0.0.1", VOL_PORT))

            print(f"[HUD] 🔧 Setting socket timeout...", flush=True)
            sock.settimeout(0.1)

            print(f"[HUD] 🌐 UDP server listening on :{VOL_PORT}", flush=True)

            while not self.isInterruptionRequested():
                try:
                    data, addr = sock.recvfrom(128)
                    txt = data.decode().strip()
                    print(f"[HUD] 📨 Received UDP from {addr}: '{txt}'", flush=True)
                    if txt.startswith("vol:"):
                        # Parse new format: "vol:RMS,bass:BASS,mid:MID,treble:TREBLE"
                        try:
                            parts = txt.split(',')
                            vol_val = float(parts[0][4:])  # Extract "vol:X.XXX"

                            # Parse frequency bands
                            freq_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.34}
                            for part in parts[1:]:
                                if ':' in part:
                                    key, val = part.split(':', 1)
                                    if key in freq_bands:
                                        freq_bands[key] = float(val)

                            print(f"[HUD] 🎤 Emitting volume signal: {vol_val:.4f}, freq: {freq_bands}", flush=True)
                            self.volume.emit(vol_val)
                            self.frequency_bands.emit(freq_bands)
                        except (ValueError, IndexError) as e:
                            print(f"[HUD] ⚠️ Failed to parse volume data: {e}", flush=True)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[HUD] ❌ Error in UDP receive loop: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[HUD] ❌ UDP server FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            if sock:
                print(f"[HUD] 🔧 Closing UDP socket...", flush=True)
                sock.close()
            print(f"[HUD] 🛑 VolumeListener thread terminated", flush=True)


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

        # Frequency bands for color mapping
        self._frequency_bands = {'bass': 0.33, 'mid': 0.33, 'treble': 0.34}

        # Theme manager initialization
        self._theme_manager = ThemeManager(THEME_ORIGINAL)

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
        print(f"[HUD] 🔧 Creating IPCServer thread...", flush=True)
        self._ipc = IPCServer(self)
        print(f"[HUD] 🔧 Connecting IPC command signal...", flush=True)
        self._ipc.command.connect(self._on_command)
        print(f"[HUD] 🔧 Starting IPCServer thread...", flush=True)
        self._ipc.start()

        print(f"[HUD] 🔧 Creating VolumeListener thread...", flush=True)
        self._vol = VolumeListener(self)
        print(f"[HUD] 🔧 Connecting volume signal...", flush=True)
        self._vol.volume.connect(self._on_volume)
        print(f"[HUD] 🔧 Connecting frequency bands signal...", flush=True)
        self._vol.frequency_bands.connect(self._on_frequency_bands)
        print(f"[HUD] 🔧 Starting VolumeListener thread...", flush=True)
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
        print("[HUD] 🎙️ Setting state to LISTENING", flush=True)
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
        elif c.startswith("mouse_click:"):
            # Parse mouse click count: "mouse_click:1", "mouse_click:2", etc.
            try:
                click_num = int(c.split(":")[1])
                self._on_mouse_click(click_num)
            except (ValueError, IndexError):
                print(f"[HUD] Invalid mouse_click command: {c}", flush=True)
        else:
            print(f"[HUD] Unknown command: {c}", flush=True)

    def _on_volume(self, val):
        self._voice_raw  = min(1.0, val * 6.0)
        self._last_vol_t = time.time()
        # DEBUG: Log every volume packet for diagnosis
        print(f"[HUD] 🎤 Received volume: {val:.4f} -> voice_raw={self._voice_raw:.4f}", flush=True)

    def _on_frequency_bands(self, freq_bands: dict):
        """Update frequency bands for color mapping."""
        self._frequency_bands = freq_bands
        print(f"[HUD] 🎵 Frequency bands updated: {freq_bands}", flush=True)

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

        # Update hue offset for animated themes (no longer needed, kept for compatibility)
        pass

        # Animate pill size (smooth lerp)
        target_w = PILL_W_ACTIVE if self._state in (LISTENING, THINKING, PROCESSING) else PILL_W_IDLE
        target_h = PILL_H_ACTIVE if self._state in (LISTENING, THINKING, PROCESSING) else PILL_H_IDLE

        self._cur_w += (target_w - self._cur_w) * 0.18
        self._cur_h += (target_h - self._cur_h) * 0.18

        # Voice decay
        if time.time() - self._last_vol_t > 0.15:
            self._voice_raw *= 0.80

        # Mouse click counter timeout - INSTANT reset after 1.5 seconds
        if self._show_mouse_counter:
            time_since_click = time.time() - self._last_mouse_click_time
            if time_since_click > self._mouse_click_timeout:
                # INSTANT reset - no gradual fade
                self._show_mouse_counter = False
                self._mouse_click_count = 0
                self._mouse_counter_alpha = 0.0
                print(f"[HUD] ⚡ Timeout (1.5s) - counter instantly reset", flush=True)
                self.update()  # Immediate repaint

        target = self._voice_raw if self._state == LISTENING else 0.0
        spd    = 0.38 if target > self._voice_smooth else 0.08
        self._voice_smooth += (target - self._voice_smooth) * spd

        v   = self._voice_smooth
        t   = self._t
        mid = (NUM_BARS - 1) / 2.0

        # DEBUG: Log voice data every 60 ticks (~1 second)
        if int(self._t * 60) % 60 == 0 and self._state == LISTENING:
            print(f"[HUD DEBUG] voice_raw={self._voice_raw:.3f} voice_smooth={v:.3f} state={self._state}", flush=True)

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

        # Use theme manager for background and border
        fill_alpha = 50 if self._state == HIDDEN else 240
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._theme_manager.create_background_brush(px, py, ph, fill_alpha))
        p.drawPath(pill)

        # Use theme manager for border
        p.setPen(self._theme_manager.create_border_pen(px, py, pw, ph, 0.0))
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

                # Calculate normalized bar height factor (0.0 to 1.0)
                bar_height_factor = (bh - BAR_MIN_H) / (BAR_MAX_H - BAR_MIN_H) if BAR_MAX_H > BAR_MIN_H else 0

                # Use voice intensity for dynamic coloring
                voice_intensity = self._voice_smooth

                # Get dynamic color from theme manager with frequency bands
                color = self._theme_manager.get_bar_color(
                    bar_index=i,
                    total_bars=NUM_BARS,
                    voice_intensity=voice_intensity,
                    bar_height_factor=bar_height_factor,
                    frequency_bands=self._frequency_bands
                )

                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color))
                p.drawRoundedRect(
                    QRectF(bx - BAR_W / 2, by, BAR_W, bh), BAR_R, BAR_R
                )

            p.setClipping(False)

        # Draw mouse click counter (outside the pill)
        # Dots show during clicking process, whether starting or stopping recording
        # Only hide when 4th click triggers action (handled in _on_mouse_click)
        # Strict threshold: must be BOTH visible AND have meaningful alpha
        if self._show_mouse_counter and self._mouse_counter_alpha > 0.05:
            self._draw_mouse_click_counter(p, cx, cy)

        p.end()

    def _draw_mouse_click_counter(self, p, cx, cy):
        """Draw 4 dots indicating mouse click progress."""
        num_clicks = 4
        dot_radius = 4.0
        dot_spacing = 12.0
        total_width = (num_clicks - 1) * dot_spacing
        start_x = cx - total_width / 2
        dot_y = cy + PILL_H_ACTIVE / 2 + 15  # Below the pill

        for i in range(num_clicks):
            dot_x = start_x + i * dot_spacing

            # Calculate alpha for this dot
            if i < self._mouse_click_count:
                # Active dot (clicked)
                dot_alpha = int(255 * self._mouse_counter_alpha)
                color = QColor(0, 255, 128, dot_alpha)  # Bright green
            else:
                # Inactive dot (not yet clicked)
                dot_alpha = int(80 * self._mouse_counter_alpha)
                color = QColor(255, 255, 255, dot_alpha)  # Dim white

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawEllipse(
                QRectF(dot_x - dot_radius, dot_y - dot_radius, dot_radius * 2, dot_radius * 2)
            )

        # Draw click count text
        text = f"{self._mouse_click_count}/4"
        p.setPen(QColor(255, 255, 255, int(255 * self._mouse_counter_alpha)))
        font = p.font()
        font.setPixelSize(11)
        p.setFont(font)
        p.drawText(QRectF(cx - 20, dot_y + 10, 40, 20), Qt.AlignmentFlag.AlignCenter, text)

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
