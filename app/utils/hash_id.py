"""
Utility functions for generating deterministic identifiers.

The ingestion pipeline uses point identifiers to ensure that re‑ingesting the
same document yields the same point IDs.  This module exposes a helper
that constructs a deterministic integer ID from a document ID, page
number and chunk identifier.  The hash is truncated to 15 hex digits
and converted to an integer to fit within Qdrant’s 64‑bit space.
"""
from __future__ import annotations

import hashlib


def point_id(doc_id: str, page: int, chunk_id: str) -> int:
    """Compute a deterministic integer ID for a vector point.

    Parameters
    ----------
    doc_id: str
        The unique document identifier used during ingestion.
    page: int
        The 1‑based page number within the document.
    chunk_id: str
        A chunk identifier (e.g. "1:1" for first chunk on page one).

    Returns
    -------
    int
        A positive integer derived from the SHA‑1 hash of the inputs.
    """
    key = f"{doc_id}:{page}:{chunk_id}"
    # Compute SHA‑1 and take the first 15 hex chars (60 bits)
    digest = hashlib.sha1(key.encode()).hexdigest()[:15]
    return int(digest, 16)
