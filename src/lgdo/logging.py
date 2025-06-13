"""This module implements some helpers for setting up logging."""

# ruff: noqa: UP007
from __future__ import annotations

import logging
from typing import Optional

import colorlog

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
FATAL = logging.FATAL
CRITICAL = logging.CRITICAL


def setup(level: int = logging.INFO, logger: Optional[logging.Logger] = None) -> None:
    """Setup a colorful logging output.

    If `logger` is None, sets up only the ``lgdo`` logger.

    Parameters
    ----------
    level
        logging level (see :mod:`logging` module).
    logger
        if not `None`, setup this logger.

    Examples
    --------
    >>> from lgdo import logging
    >>> logging.setup(level=logging.DEBUG)
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )

    if logger is None:
        logger = colorlog.getLogger("lgdo")

    logger.setLevel(level)
    logger.addHandler(handler)
