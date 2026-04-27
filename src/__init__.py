"""
Common Initialization Module
============================
This module sets up the global logging configuration for the entire application using structlog.
It defines how logs are formatted, including timestamps, log levels, and callsite information
(file name, function name, and line number) to make debugging easier.
The resulting 'log' object is used by all other modules to ensure consistent logging output.
"""

import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

# Configure structlog with a clean, readable console output format.
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        CallsiteParameterAdder({
            CallsiteParameter.FILENAME,
            CallsiteParameter.FUNC_NAME,
            CallsiteParameter.LINENO,
        }),
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(
            colors=True,
            force_colors=True,
            pad_event_to=40,
        )
    ],
)

# Export a single global logger for the project.
log = structlog.get_logger()
