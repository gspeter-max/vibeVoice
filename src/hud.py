"""
hud.py — macOS menu bar waveform HUD
====================================
Compact native status item with premium white waveform bars.

States:
  - hidden     : small idle pill outline (always on-screen)
  - listening  : pill EXPANDS, bars animate with real mic volume
  - thinking   : bars soften into a restrained white shimmer
  - processing : bars do a slow white sweep/pulse
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
import random

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

import objc
from PySide6.QtCore import QObject, QTimer, QThread, Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

try:
    from AppKit import (
        NSApp,
        NSColor,
        NSBezierPath,
        NSMakeRect,
        NSView,
        NSStatusBar,
    )
except Exception:  # pragma: no cover - menu bar rendering is macOS-only
    NSApp = None
    NSColor = None
    NSBezierPath = None
    NSMakeRect = None
    NSView = None
    NSStatusBar = None

# ── Menu bar dimensions ───────────────────────────────────────────────────────
STATUS_ITEM_W = 64
STATUS_ITEM_H = 22
STATUS_ITEM_VERTICAL_SHIFT = 0.0

# ── Bar config ────────────────────────────────────────────────────────────────
NUM_BARS  = 9
BAR_W     = 2.0
BAR_GAP   = 2.5
BAR_MAX_H = 15.0
BAR_MIN_H = 5.0
BAR_R     = 0.9     # rounded tip
BAR_COLOR_MODE = "monochrome"
WAVE_STYLE = "chaotic-zigzag"
LISTEN_NOISE_DRIFT_LERP = 0.32
LISTEN_NOISE_DRIFT_WEIGHT = 0.34
LISTEN_NOISE_SPARK_WEIGHT = 0.18
LISTEN_ZIGZAG_WEIGHT = 0.34
LISTEN_KICK_PROB = 0.16
LISTEN_KICK_DAMP = 0.66
LISTEN_KICK_MAG = 0.42
LISTEN_SMOOTH_BASE_SPEED_A = 3.9
LISTEN_SMOOTH_BASE_SPEED_B = 8.4
LISTEN_ZIGZAG_SPEED_PRIMARY = 3.0
LISTEN_ZIGZAG_SPEED_ALT = 4.4
THINKING_SPEED_A = 2.6
THINKING_SPEED_B = 4.6
PROCESSING_SWEEP_SPEED = 1.45
PROCESSING_BREATH_SPEED = 0.95

IPC_PORT = 57234
VOL_PORT = 57235

HIDDEN     = "hidden"
LISTENING  = "listening"
THINKING   = "thinking"
PROCESSING = "processing"
DONE       = "done"


def runtime_signature() -> str:
    return (
        f"mode={BAR_COLOR_MODE} anchor=menu-bar bars={NUM_BARS} "
        f"bar_w={BAR_W:.1f} gap={BAR_GAP:.1f} item_w={STATUS_ITEM_W} wave={WAVE_STYLE}"
    )


def bar_color_for_draw(voice_intensity: float, bar_height_factor: float) -> QColor:
    """Return strict white for waveform bars with alpha-only dynamics."""
    # Keep bars visibly bright white at all times, with subtle dynamic lift.
    alpha = int(248 + voice_intensity * 5 + bar_height_factor * 3)
    alpha = max(248, min(255, alpha))
    return QColor(255, 255, 255, alpha)


def _triangle_wave(x: float) -> float:
    """Return a sharp triangle wave in range [-1, 1]."""
    frac = x - math.floor(x)
    return 1.0 - 4.0 * abs(frac - 0.5)


def compute_menu_bar_waveform_layout(
    *,
    status_width: float,
    status_height: float,
    num_bars: int,
    bar_width: float,
    bar_gap: float,
    bar_height: float,
) -> list[dict]:
    """Return centered bar rectangles for a compact menu-bar waveform."""
    total_wave_width = num_bars * bar_width + max(0, num_bars - 1) * bar_gap
    start_x = (status_width - total_wave_width) / 2.0
    start_y = (status_height - bar_height) / 2.0

    layout = []
    for idx in range(num_bars):
        x = start_x + idx * (bar_width + bar_gap)
        layout.append(
            {
                "x": x,
                "y": start_y,
                "width": bar_width,
                "height": bar_height,
            }
        )
    return layout


class MenuBarWaveformView(NSView if NSView is not None else object):
    def initWithFrame_(self, frame):
        self = objc.super(MenuBarWaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._hud = None
        return self

    def setHud_(self, hud):
        self._hud = hud

    def isFlipped(self):
        return True

    def drawRect_(self, _dirtyRect):
        if self._hud is None:
            return
        self._hud._draw_menu_bar_waveform(self.bounds())


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


# ── Main menu bar controller ────────────────────────────────────────────────
class MenuBarWaveformController(QObject):

    def __init__(self):
        super().__init__()
        self._status_item = None
        self._status_view = None

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
        print(f"[HUD] Theme mode: {runtime_signature()}", flush=True)
        preview = bar_color_for_draw(voice_intensity=1.0, bar_height_factor=1.0)
        print(
            f"[HUD] Theme preview RGB=({preview.red()},{preview.green()},{preview.blue()}) "
            f"alpha={preview.alpha()}",
            flush=True,
        )

        # Per-bar state
        self._bar_h     = [BAR_MIN_H] * NUM_BARS
        self._bar_phase = [i * 0.28 for i in range(NUM_BARS)]
        self._bar_noise = [0.0] * NUM_BARS
        self._bar_kick  = [0.0] * NUM_BARS

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

        self._ensure_menu_bar_item()
        self._request_menu_bar_view_redraw()

        print("[HUD] Menu bar HUD ready ✓", flush=True)

    def _ensure_visible(self):
        """Re-assert the menu-bar item exists and remains visible."""
        self._ensure_menu_bar_item()
        self._request_menu_bar_view_redraw()

    def _ensure_menu_bar_item(self):
        if self._status_item is not None:
            return
        if NSStatusBar is None:
            raise RuntimeError("macOS status bar APIs are unavailable")
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            STATUS_ITEM_W
        )
        if self._status_view is None and NSView is not None:
            self._status_view = MenuBarWaveformView.alloc().initWithFrame_(
                NSMakeRect(0, 0, STATUS_ITEM_W, STATUS_ITEM_H)
            )
            self._status_view.setHud_(self)
        if self._status_view is not None:
            self._status_item.setView_(self._status_view)
            self._status_view.setFrame_(NSMakeRect(0, 0, STATUS_ITEM_W, STATUS_ITEM_H))

    def _status_bar_alpha(self) -> float:
        if self._state == LISTENING:
            return min(1.0, 0.88 + self._voice_smooth * 0.12)
        if self._state == THINKING:
            return 0.96
        if self._state == PROCESSING:
            return 0.98
        if self._state == DONE:
            return 1.0
        return 0.74

    def _request_menu_bar_view_redraw(self):
        if self._status_view is not None:
            self._status_view.setNeedsDisplay_(True)

    def _draw_menu_bar_waveform(self, bounds):
        if NSColor is None or NSBezierPath is None or NSMakeRect is None:
            return

        width = float(bounds.size.width)
        height = float(bounds.size.height)
        NSColor.clearColor().set()
        NSBezierPath.bezierPathWithRect_(NSMakeRect(0, 0, width, height)).fill()

        layout = compute_menu_bar_waveform_layout(
            status_width=width,
            status_height=height,
            num_bars=NUM_BARS,
            bar_width=BAR_W,
            bar_gap=BAR_GAP,
            bar_height=BAR_MAX_H,
        )
        alpha = self._status_bar_alpha()
        for idx, rect in enumerate(layout):
            bh = max(BAR_MIN_H, self._bar_h[idx])
            scale = 0.62 + (bh - BAR_MIN_H) / max(1.0, BAR_MAX_H - BAR_MIN_H) * 0.38
            current_h = max(BAR_MIN_H, min(BAR_MAX_H, bh))
            y = (height - current_h) / 2.0 + STATUS_ITEM_VERTICAL_SHIFT
            x = rect["x"]
            w = rect["width"]

            NSColor.colorWithCalibratedWhite_alpha_(1.0, alpha * scale).set()
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, w, current_h),
                BAR_R,
                BAR_R,
            )
            path.fill()

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
        self._bar_noise = [0.0] * NUM_BARS
        self._bar_kick = [0.0] * NUM_BARS
        if not self._timer.isActive():
            self._timer.start()
        self._request_menu_bar_view_redraw()

    # ── IPC dispatcher ────────────────────────────────────────────────────
    def _on_command(self, cmd):
        c = cmd.strip()
        print(f"[HUD] ← {c}", flush=True)
        lowered = c.lower()
        if lowered.startswith("draft:") or lowered.startswith("final:"):
            return
        if lowered == "listen":
            self.show_listening()
        elif lowered == "thinking":
            self.show_thinking()
        elif lowered == "process":
            self.show_processing()
        elif lowered == "done":
            self.show_done()
        elif lowered == "hide":
            self.hide_hud()
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
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self):
        self._t = time.time() - self._t0

        # Update hue offset for animated themes (no longer needed, kept for compatibility)
        pass

        # Voice decay
        if time.time() - self._last_vol_t > 0.15:
            self._voice_raw *= 0.80

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
            centre_w = 1.0 - abs(i - mid) / mid * 0.18

            if self._state == LISTENING:
                # Chaotic zigzag drift: fast-rising random motion with sharp edges.
                noise_target = random.uniform(-1.0, 1.0)
                self._bar_noise[i] += (noise_target - self._bar_noise[i]) * LISTEN_NOISE_DRIFT_LERP
                noise_drift = self._bar_noise[i] * LISTEN_NOISE_DRIFT_WEIGHT

                # Stronger spark for non-smooth, jagged motion.
                noise_spark = random.uniform(-1.0, 1.0) * LISTEN_NOISE_SPARK_WEIGHT

                # Stochastic kick occasionally flips direction sharply.
                if random.random() < (LISTEN_KICK_PROB * (0.45 + 0.55 * min(1.0, v))):
                    self._bar_kick[i] = random.uniform(-LISTEN_KICK_MAG, LISTEN_KICK_MAG)
                self._bar_kick[i] *= LISTEN_KICK_DAMP
                kick = self._bar_kick[i]

                smooth_base = (
                    0.34 * math.sin(t * LISTEN_SMOOTH_BASE_SPEED_A + ph * 0.90) +
                    0.16 * math.sin(t * LISTEN_SMOOTH_BASE_SPEED_B + ph * 1.20)
                )
                zig_primary = _triangle_wave(t * LISTEN_ZIGZAG_SPEED_PRIMARY + i * 0.11 + ph * 0.30) * LISTEN_ZIGZAG_WEIGHT
                zig_alt = (-1.0 if i % 2 else 1.0) * _triangle_wave(t * LISTEN_ZIGZAG_SPEED_ALT + ph * 0.21) * 0.18
                raw_wave = smooth_base + zig_primary + zig_alt + noise_drift + noise_spark + kick
                wave = max(0.0, min(1.0, (raw_wave + 1.0) / 2.0))
                idle  = BAR_MIN_H + (BAR_MAX_H - BAR_MIN_H) * 0.16
                activity = min(1.0, 0.44 + v * 1.55)
                tgt   = idle + (BAR_MAX_H * wave * centre_w - idle) * activity

            elif self._state == THINKING:
                wave = (
                    0.72 * math.sin(t * THINKING_SPEED_A + i * 0.22) +
                    0.28 * math.sin(t * THINKING_SPEED_B + i * 0.12)
                )
                wave = (wave + 1.0) / 2.0
                tgt = BAR_MIN_H + (BAR_MAX_H * 0.36) * wave

            elif self._state == PROCESSING:
                sweep = (math.sin(t * PROCESSING_SWEEP_SPEED) + 1.0) / 2.0 * (NUM_BARS - 1)
                dist  = abs(i - sweep)
                glow  = math.exp(-dist * dist * 0.22)
                breath = 0.14 + 0.04 * math.sin(t * PROCESSING_BREATH_SPEED + i * 0.16)
                tgt = BAR_MIN_H + (BAR_MAX_H * 0.55) * max(glow, breath)

            elif self._state == DONE:
                prog = min(1.0, self._t * 5.0)
                tgt  = BAR_MAX_H * (1.0 - prog) + BAR_MIN_H * prog

            else:
                tgt = BAR_MIN_H

            cur = self._bar_h[i]
            self._bar_h[i] += (tgt - cur) * (0.42 if tgt > cur else 0.10)

        self._request_menu_bar_view_redraw()

        if self._state == HIDDEN:
            self._timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    try:
        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    hud = MenuBarWaveformController()
    sys.exit(app.exec())
