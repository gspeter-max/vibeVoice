import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

structlog.configure(
    processors=[
        # 1. Add log level (info, debug, etc.)
        structlog.stdlib.add_log_level,
        
        # 2. Add the callsite parameters (The "Advanced" Part)
        CallsiteParameterAdder(
            {
                CallsiteParameter.FILENAME,   # File name
                CallsiteParameter.FUNC_NAME,  # Function name
                CallsiteParameter.LINENO,     # Line number
            }
        ),
        
        # 3. Format timestamps
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f", utc=False),
        
        # 4. Render as colored console output (best for debugging)
        structlog.dev.ConsoleRenderer()
    ],
)

log = structlog.get_logger()
