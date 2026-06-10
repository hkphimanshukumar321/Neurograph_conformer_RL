"""
Logging setup — structured logging with Rich handler and file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "neurograph",
    level: str = "INFO",
    log_file: str | Path | None = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Configure and return a logger with console and optional file handlers.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level: "DEBUG", "INFO", "WARNING", "ERROR".
    log_file : str or Path, optional
        If provided, also log to this file.
    use_rich : bool
        If True, use Rich for formatted console output.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                level=level,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
