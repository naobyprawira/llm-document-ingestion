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
from typing import Any

# Read basic configuration from the environment.  These values are used
# throughout the pipeline.  You can override them via environment variables
# (see README for details).
MAX_CONCURRENT_VLM = int(os.getenv("MAX_CONCURRENT_VLM", "5"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1024"))

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_base_logger = logging.getLogger("ingestion")
if not _base_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s job=%(job)s file=%(file)s phase=%(phase)s msg=%(message)s"
    )
    _handler.setFormatter(_formatter)
    _base_logger.addHandler(_handler)
_base_logger.setLevel(_LOG_LEVEL)


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