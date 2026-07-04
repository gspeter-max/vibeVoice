import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from structlog import get_logger

log = get_logger(__file__)


@dataclass(frozen=True)
class SoundPath:
    FINISH = Path(__file__).parent / "sound_effect" / "finished.mp3"
    STARTING = Path(__file__).parent / "sound_effect" / "start.mp3"


@dataclass()
class PlaySound:
    sound_type: Literal["FINISH", "STARTING"]

    def __post_init__(self):
        self.command = {
            "Darwin": ["afplay", str(getattr(SoundPath, self.sound_type))],
            "Linux": ["aplay", str(getattr(SoundPath, self.sound_type))],
        }

    def __call__(self):
        system = platform.system()

        if cmd := self.command.get(system):
            subprocess.Popen(cmd)
        elif system == "Windows":
            log.warning("its not implemented")
        else:
            log.warning(f"Unknown OS: {system}")
