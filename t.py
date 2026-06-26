import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder
from structlog.types import Processor

structlog.configure(
    processors=[
        CallsiteParameterAdder(
            {CallsiteParameter.FILENAME, CallsiteParameter.FUNC_NAME, CallsiteParameter.PATHNAME}
        ),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
breakpoint()
log = structlog.get_logger(__file__)
log.info("something something")
