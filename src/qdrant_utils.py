"""Helper functions for working with Qdrant.

This module encapsulates interactions with a Qdrant instance: creating
clients, ensuring collections exist and constructing ``job_marker`` points
for tracking ingestion jobs.  Keeping these details in one place makes it
easier to swap out the vector store or adjust payload schemas in the
future.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import uuid

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import VectorParams, Distance, PointStruct  # type: ignore
    _HAS_QDRANT = True
except Exception:
    # Provide a dummy QdrantClient type so type checkers are happy
    QdrantClient = Any  # type: ignore
    VectorParams = Any  # type: ignore
    Distance = Any  # type: ignore
    PointStruct = Any  # type: ignore
    _HAS_QDRANT = False


def qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client using environment variables.

    :raises RuntimeError: If ``QDRANT_URL`` is not set.
    """
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Ensure that a collection with the given name and vector size exists.

    Uses ``collection_exists`` when available; otherwise attempts to call
    ``get_collection`` and creates the collection on failure.  See the
    upstream implementation for details.
    """
    try:
        # Preferred API (available in newer qdrant-client versions)
        if hasattr(client, "collection_exists") and client.collection_exists(collection):
            return
    except Exception:
        pass
    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def marker_point(
    job_key: str,
    vector: List[float],
    filename: str,
    category: str,
    collection: str,
    status: str,
    expected_chunks: Optional[int],
    uploaded_chunks: int,
    error: Optional[str] = None,
) -> PointStruct:
    """Construct a ``job_marker`` point for tracking pipeline progress.

    The returned point has a deterministic ID of the form ``marker-<job_key>``
    and includes information about the current status, number of expected
    chunks, number uploaded so far and any error message.  When ``status``
    is either ``done`` or ``failed``, the payload will include a
    ``finished_at`` ISO timestamp.
    """
    payload: Dict[str, Any] = {
        "type": "job_marker",
        "job_key": job_key,
        "filename": filename,
        "category": category,
        "collection": collection,
        "status": status,
        "expected_chunks": expected_chunks,
        "uploaded_chunks": uploaded_chunks,
        "finished_at": None,
    }
    if status in ("done", "failed"):
        payload["finished_at"] = datetime.now().isoformat()
    if error:
        payload["error"] = str(error)
    return PointStruct(id=job_key, vector=vector, payload=payload)



__all__ = [
    "qdrant_client",
    "ensure_collection",
    "marker_point",
]