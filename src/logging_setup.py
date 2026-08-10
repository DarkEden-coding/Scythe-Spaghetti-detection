"""Logging configuration.

Replaces the scattered ``print()`` calls. Everything goes to stdout (which
systemd captures into the journal) and to a rotating file, so the planned
``/get_log_file`` command has something to serve.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from src import paths

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 3

#: Third-party loggers that are chatty at INFO.
NOISY_LOGGERS = ("discord", "discord.client", "discord.gateway", "urllib3", "PIL")


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    console: bool = True,
) -> Path | None:
    """Install handlers on the root logger. Safe to call more than once."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(resolved)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    target = log_file if log_file is not None else paths.LOG_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            target, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)
    except OSError as exc:
        root.warning("Could not open log file %s: %s", target, exc)
        target = None

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(resolved, logging.WARNING))

    return target


__all__ = ["configure_logging"]
