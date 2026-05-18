"""Platform-specific helpers for Ear runtime behavior.

This module owns small operating-system integrations that should not live
inside the main Ear runtime class.
"""

from __future__ import annotations

from pathlib import Path
import platform

from src import log


_start_sound = None


def load_start_sound() -> None:
    """Load the Ear start sound once so playback is fast during recording start."""

    global _start_sound
    try:
        sound_path = str(Path(__file__).parent.parent.parent.parent / "sound_effect" / "start.mp3")

        if platform.system() == "Darwin":
            try:
                from AppKit import NSSound

                _start_sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                    sound_path,
                    True,
                )
            except ImportError:
                pass

        if _start_sound is None:
            try:
                from PySide6.QtCore import QUrl
                from PySide6.QtMultimedia import QSoundEffect

                _start_sound = QSoundEffect()
                _start_sound.setSource(QUrl.fromLocalFile(sound_path))
                _start_sound.setVolume(1.0)
            except ImportError:
                log.info("[Ear] PySide6.QtMultimedia not available for sound effects")
    except (OSError, RuntimeError) as error:
        log.info("[Ear] Failed to load start sound: %s", error)


def play_start_sound() -> None:
    """Play the already-loaded Ear start sound without changing its behavior."""

    if not _start_sound:
        return

    try:
        if platform.system() == "Darwin" and hasattr(_start_sound, "isPlaying"):
            if _start_sound.isPlaying():
                _start_sound.stop()
            _start_sound.play()
        else:
            _start_sound.play()
    except (RuntimeError, AttributeError) as error:
        log.info("[Ear] Failed to play start sound: %s", error)


def enable_macos_voice_isolation() -> None:
    """Turn on macOS voice processing when the host system exposes that API."""

    try:
        import AVFoundation
    except ImportError:
        log.info("[Ear] AVFoundation not available")
        return

    try:
        engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = engine.inputNode()
        if hasattr(input_node, "setVoiceProcessingEnabled_error_"):
            success, error = input_node.setVoiceProcessingEnabled_error_(True, None)
            if not success:
                log.info(f"[Ear] Voice processing not enabled: {error}")
        else:
            log.info("[Ear] inputNode does not support setVoiceProcessingEnabled_error_")
    except RuntimeError as error:
        log.info("[Ear] Voice isolation init failed: %s", error)
