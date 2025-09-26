"""Helper functions for working with Qdrant.

This module encapsulates interactions with a Qdrant instance: creating
clients, ensuring collections exist and constructing ``job_marker`` points
for tracking ingestion jobs. Keeping these details in one place makes it
easier to swap out the vector store or adjust payload schemas in the future.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

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


# ---------------------------
# Client & collection helpers
# ---------------------------

def qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client using environment variables.

    :raises RuntimeError: If ``QDRANT_URL`` is not set.
    """
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Ensure that a collection with the given name and vector size exists."""
    try:
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


# ---------------------------
# Safety guard for IDs (optional but useful during debugging)
# ---------------------------

UUID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|"
    r"urn:uuid:[0-9a-fA-F-]{36})$"
)

def _is_valid_point_id(pid: Any) -> bool:
    if isinstance(pid, int):
        return pid >= 0
    if isinstance(pid, str):
        return bool(UUID_RE.match(pid))
    return False

def safe_upsert(client: QdrantClient, *, collection_name: str, points: Iterable, wait: bool = True):
    """Validate point IDs before upsert to catch bad IDs early."""
    pts_list = list(points)
    for p in pts_list:
        pid = getattr(p, "id", None)
        if pid is None and isinstance(p, dict):
            pid = p.get("id")
        if not _is_valid_point_id(pid):
            print(f"[FATAL] Invalid point id about to be upserted: {pid!r}")
            raise ValueError(f"Invalid Qdrant point id: {pid!r}")
    return client.upsert(collection_name=collection_name, points=pts_list, wait=wait)


# ---------------------------
# Marker helpers
# ---------------------------

def marker_id_for_filename(collection: str, filename_no_ext: str) -> str:
    """Stable UUIDv5 for a given collection + filename base (valid Qdrant id)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{filename_no_ext.lower()}"))

def marker_point(
    job_key: str,
    vector: List[float],
    filename: str,          # may include extension
    category: str,
    collection: str,
    status: str,
    expected_chunks: Optional[int],
    uploaded_chunks: int,
    error: Optional[str] = None,
) -> PointStruct:
    """Construct a ``job_marker`` point for tracking pipeline progress.

    The point *ID* is a UUIDv5 derived from ``collection + filename_without_ext``,
    which satisfies Qdrant's ID constraints (uint64 or UUID) and remains stable.
    The human-friendly filename (sans extension) is stored in payload["filename"].
    """
    base = os.path.splitext(filename)[0]  # ensure no extension
    payload: Dict[str, Any] = {
        "type": "job_marker",
        "job_key": job_key,
        "filename": base,
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

    point_id = marker_id_for_filename(collection, base)  # <-- VALID UUID
    return PointStruct(id=point_id, vector=vector, payload=payload)


__all__ = [
    "qdrant_client",
    "ensure_collection",
    "marker_id_for_filename",
    "marker_point",
    "safe_upsert",
]
