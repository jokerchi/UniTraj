"""
Colored Logging System with File Output.

Provides a ``Logger`` class with colorized console output and optional file logging.
Inspired by https://github.com/SebiSebi/friendlylog.

Usage:
    logger = Logger(__name__, log_path="output.log", colorize=True)
    logger.info("Training started")
"""

import logging
import sys
from copy import copy
from typing import Optional, TextIO, Union

from colored import attr, fg

# ============================================================================
# Log Level Constants
# ============================================================================

DEBUG: str = "debug"
INFO: str = "info"
WARNING: str = "warning"
ERROR: str = "error"
CRITICAL: str = "critical"

LOG_LEVELS: dict = {
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR,
    CRITICAL: logging.CRITICAL,
}


# ============================================================================
# Internal Formatter
# ============================================================================

class _Formatter(logging.Formatter):
    """Custom log formatter with optional ANSI colorization."""

    def __init__(self, colorize: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colorize = colorize

    @staticmethod
    def _process(msg: str, loglevel: str, colorize: bool) -> str:
        """Apply color codes to the message based on log level."""
        loglevel = str(loglevel).lower()
        if loglevel not in LOG_LEVELS:
            raise RuntimeError(f"{loglevel} should be one of {LOG_LEVELS}.")

        msg = f"{loglevel.upper()}: {msg}"

        if not colorize:
            return msg

        color_map = {
            DEBUG: fg(5),
            INFO: fg(4),
            WARNING: f"{fg(214)}{attr(1)}",
            ERROR: f"{fg(202)}{attr(1)}",
            CRITICAL: f"{fg(196)}{attr(1)}",
        }

        if loglevel in (WARNING, ERROR, CRITICAL):
            return f"{color_map[loglevel]}{msg}{attr(21)}{attr(0)}"
        return f"{color_map[loglevel]}{msg}{attr(0)}"

    def format(self, record: logging.LogRecord) -> str:
        record = copy(record)
        record.msg = self._process(record.msg, record.levelname, self.colorize)
        return super().format(record)


# ============================================================================
# Logger
# ============================================================================

class Logger:
    """Colorized logger with console and file output.

    Args:
        name: Logger name (used internally).
        colorize: Enable ANSI color output on console.
        log_path: Optional path to a log file.
        stream: Output stream for console logging (default: stdout).
        level: Minimum log level (default: "info").
    """

    def __init__(
        self,
        name: str = "default",
        colorize: bool = False,
        log_path: Optional[str] = None,
        stream: TextIO = sys.stdout,
        level: str = INFO,
    ):
        self.name = name

        # Internal logger (name-mangled for encapsulation)
        self.__logger = logging.getLogger(f"_logger-{name}")
        self.__logger.propagate = False
        self.setLevel(level.lower())

        # Custom formatter with timestamp and process ID
        self.__formatter = _Formatter(
            colorize=colorize,
            fmt="[%(process)d][%(asctime)s.%(msecs)03d @ %(funcName)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )

        # Handler registry
        self.__stream_to_handler: dict = {}
        self.clear_handlers()
        self.__main_handler = self.add_handler(stream)

        # File handler (if path provided)
        if log_path is not None:
            fh = logging.FileHandler(log_path, "w")
            self.__logger.addHandler(fh)

        # Expose logging methods directly
        self.debug = self.__logger.debug
        self.info = self.__logger.info
        self.warning = self.__logger.warning
        self.error = self.__logger.error
        self.critical = self.__logger.critical

    # --- Decorator ---

    def log_function(self):
        """Decorator that logs function entry (with args) and exit."""

        def wrapper(func):
            def func_wrapper(*args, **kwargs):
                self.__logger.info(
                    f"calling <{func.__name__}>\n\t  args: {args}\n\tkwargs: {kwargs}"
                )
                out = func(*args, **kwargs)
                self.__logger.info(f"exiting <{func.__name__}>")
                return out

            return func_wrapper

        return wrapper

    # --- Level management ---

    def setLevel(self, level: Union[str, int]) -> None:
        """Set the minimum log level (string name or logging constant)."""
        if isinstance(level, int):
            self.__logger.setLevel(level)
        else:
            if level.lower() not in LOG_LEVELS:
                raise ValueError(f"level should be one of {LOG_LEVELS}")
            self.__logger.setLevel(LOG_LEVELS[level.lower()])

    # --- Handler management ---

    def add_handler(self, stream: TextIO) -> logging.StreamHandler:
        """Add a stream handler and return it."""
        handler = logging.StreamHandler(stream)
        handler.setFormatter(self.__formatter)
        self.__logger.addHandler(handler)
        self.__stream_to_handler[stream] = handler
        return handler

    def remove_handler(self, stream: TextIO) -> bool:
        """Remove the handler for the given stream. Returns True if successful."""
        if stream in self.__stream_to_handler:
            self.__logger.removeHandler(self.__stream_to_handler[stream])
            self.__stream_to_handler.pop(stream)
            return True
        return False

    def clear_handlers(self) -> None:
        """Remove all handlers."""
        self.__logger.handlers = []
        self.__stream_to_handler = {}

    def get_handlers(self) -> list:
        """Return the list of registered handlers."""
        return self.__logger.handlers

    # --- Internal accessors (use with care) ---

    @property
    def inner_logger(self) -> logging.Logger:
        """Direct access to the underlying ``logging.Logger``."""
        return self.__logger

    @property
    def inner_stream_handler(self) -> logging.StreamHandler:
        """Access to the main console stream handler."""
        return self.__main_handler

    @property
    def inner_formatter(self) -> _Formatter:
        """Access to the custom formatter."""
        return self.__formatter


# ============================================================================
# Utility
# ============================================================================

def log_info(args, logger: Logger) -> None:
    """Log key training configuration settings.

    Args:
        args: Configuration namespace with ``data`` and ``training`` attributes.
        logger: Logger instance to write to.
    """
    logger.info("***********************************")
    logger.info(f"Dataset: {args.data.dataset}")
    logger.info(f"Trajectory Length: {args.data.traj_length}")
    logger.info(f"Epochs: {args.training.n_epochs}")
    logger.info(f"Batch Size: {args.training.batch_size}")
    logger.info("***********************************")
