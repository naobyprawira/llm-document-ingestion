"""Helper functions for working with Qdrant.

Encapsulates: client creation, collection ensure, safe upsert, and
schema-consistent point builders (content + metadata.*).
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

    :raises RuntimeError: If QDRANT_URL is not set.
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
# Safety guard for IDs
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
# Schema helpers
# ---------------------------

def build_payload(*, content: Optional[str] = None, **metadata: Any) -> Dict[str, Any]:
    """
    Normalized payload:
      - Text under top-level "content" (if provided)
      - All other fields nested under "metadata"
    """
    payload: Dict[str, Any] = {"metadata": metadata}
    if content is not None:
        payload["content"] = content
    return payload


# ---------------------------
# Marker helpers
# ---------------------------

def marker_id_for_filename(collection: str, filename_no_ext: str) -> str:
    """Stable UUIDv5 for collection + filename base (valid Qdrant id)."""
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
    """
    Construct a 'job_marker' point for tracking pipeline progress.

    SCHEMA: no 'content' for markers; everything under payload['metadata'].
    """
    base = os.path.splitext(filename)[0]  # strip extension

    meta: Dict[str, Any] = {
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
        meta["finished_at"] = datetime.now().isoformat()
    if error:
        meta["error"] = str(error)

    point_id = marker_id_for_filename(collection, base)
    return PointStruct(id=point_id, vector=vector, payload={"metadata": meta})


__all__ = [
    "qdrant_client",
    "ensure_collection",
    "marker_id_for_filename",
    "marker_point",
    "safe_upsert",
    "build_payload",
]
