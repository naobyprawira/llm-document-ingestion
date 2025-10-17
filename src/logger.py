"""Logging helpers and global configuration.

This module centralises the project’s logging setup and exposes a thin
``get_logger`` helper that injects ``job``, ``file`` and ``phase`` keys into
log records.  It also defines a handful of configuration constants read
from the environment.  Other modules import these constants rather than
parsing environment variables repeatedly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Read basic configuration from the environment.  These values are used
# throughout the pipeline.  You can override them via environment variables
# (see README for details).
MAX_CONCURRENT_VLM = int(os.getenv("MAX_CONCURRENT_VLM", "5"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1024"))

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_base_logger = logging.getLogger("ingestion")
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | job=%(job)s file=%(file)s phase=%(phase)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
if not _base_logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_formatter)
    _base_logger.addHandler(stream_handler)

_log_file_path = os.getenv("LOG_FILE", "logs/ingestion.log")
if _log_file_path:
    try:
        log_path = Path(_log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid adding duplicate file handlers when module reloads.
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path) for h in _base_logger.handlers):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(_formatter)
            _base_logger.addHandler(file_handler)
    except Exception:
        # Fallback silently to console-only logging if file handler setup fails.
        pass

_base_logger.setLevel(_LOG_LEVEL)
_base_logger.propagate = False


class _Adapter(logging.LoggerAdapter):
    """Inject ``job``, ``file`` and ``phase`` fields into log records.

    The FastAPI app logs each request using a unique ``job`` key.  The
    ``phase`` value can be used to differentiate between parsing, chunking,
    embedding and other stages of the pipeline.  See the README for more
    information on logging conventions.
    """

    def __init__(self, logger: logging.Logger, job: str = "-", file: str = "-", phase: str = "-") -> None:
        super().__init__(logger, {"job": job, "file": file, "phase": phase})

    def with_phase(self, phase: str) -> "_Adapter":
        """Return a new adapter with an updated phase but the same job/file."""
        return _Adapter(self.logger, self.extra.get("job", "-"), self.extra.get("file", "-"), phase)


def get_logger(job: str = "-", file: str = "-", phase: str = "-") -> _Adapter:
    """Return a logger adapter with the given context values.

    :param job: Unique identifier for the current job/request.
    :param file: The filename being processed.
    :param phase: The current stage of the pipeline (parse, chunk, embed, etc.).
    :return: A logger adapter that injects context values into log records.
    """
    return _Adapter(_base_logger, job, file, phase)


__all__ = [
    "MAX_CONCURRENT_VLM",
    "EMBED_BATCH_SIZE",
    "CHUNK_MAX_CHARS",
    "get_logger",
]
