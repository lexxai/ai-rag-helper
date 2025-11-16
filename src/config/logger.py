import logging
import sys


def configure_logger(
    name: str | None = None,
    level: str = "INFO",
    log_format: str | None = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with standardized settings.

    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string (uses default if None)
        include_timestamp: Whether to include timestamp in log format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Define log format
    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - " "%(filename)s:%(lineno)d - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - " "%(filename)s:%(lineno)d - %(message)s"

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def setup_root_logger(level: str = "INFO", log_format: str | None = None) -> None:
    """
    Setup the root logger for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Optional custom log format string
    """
    if log_format is None:
        # log_format = "%(asctime)s - %(name)s - %(levelname)s - " "%(filename)s:%(lineno)d - %(message)s"
        log_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Get a logger instance with optional custom level.

    Args:
        name: Logger name (typically __name__)
        level: Optional logging level override

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    return logger
