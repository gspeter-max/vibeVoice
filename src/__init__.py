"""
Common Initialization Module
============================
This module sets up the global logging configuration for the entire application using Python's built-in logging.
It defines how logs are formatted, including timestamps, log levels, and callsite information
(file name, function name, and line number) to make debugging easier.
The resulting 'log' object is used by all other modules to ensure consistent logging output.
"""

import logging
import os
import sys

# Determine log level from environment, defaulting to INFO
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
is_brain_process: bool = False
if sys.argv and sys.argv[0]:
    is_brain_process = os.path.basename(sys.argv[0]) == "brain.py"


class MilestoneFormatter(logging.Formatter):
    def __init__(self, is_debug: bool = False):
        self.is_debug = is_debug
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        if self.is_debug or record.levelno > logging.INFO:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            formatter = logging.Formatter(fmt="%(message)s")
        return formatter.format(record)


if is_brain_process:
    os.makedirs("logs", exist_ok=True)
    handler = logging.FileHandler("logs/brain.log", encoding="utf-8")
else:
    handler = logging.StreamHandler(sys.stdout)

# Configure the built-in logging module
handler.setFormatter(MilestoneFormatter(is_debug=(log_level == logging.DEBUG)))
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.addHandler(handler)

# Export a single global logger for the project.
log = logging.getLogger("vibeVoice")
