"""Helper functions for working with Qdrant.

Encapsulates: client creation, collection ensure, safe upsert, and
schema-consistent point builders (text + metadata.*).
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

# -------- Robust import of client & models --------
try:
    from qdrant_client import QdrantClient, models  # type: ignore
    _HAS_QDRANT = True
except Exception:
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models   # type: ignore
        _HAS_QDRANT = True
    except Exception:
        QdrantClient = Any  # type: ignore
        models = None       # type: ignore
        _HAS_QDRANT = False

if _HAS_QDRANT:
    VectorParams = models.VectorParams
    Distance = models.Distance
    PointStruct = models.PointStruct
    SparseVectorParams = getattr(models, "SparseVectorParams", None)
    Modifier = getattr(models, "Modifier", None)
    SparseIndexParams = getattr(models, "SparseIndexParams", None)
else:
    VectorParams = Any  # type: ignore
    Distance = Any      # type: ignore
    PointStruct = Any   # type: ignore
    SparseVectorParams = Any  # type: ignore
    Modifier = None
    SparseIndexParams = Any  # type: ignore

# ---------------------------
# Client & collection helpers
# ---------------------------

def qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client using environment variables."""
    if not _HAS_QDRANT:
        raise ImportError("qdrant-client is required")
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))

def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Ensure that a collection with the given name and vector size exists.

    Creates a dense vector space (COSINE) and configures BM25/IDF sparse vectors
    using Python client's `sparse_vectors_config`.
    """
    # 1) Exists shortcut
    try:
        if hasattr(client, "collection_exists") and client.collection_exists(collection_name=collection):
            return
    except Exception:
        pass

    # 2) Try get
    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass

    # 3) Create with dense + sparse (BM25/IDF)
    sparse_cfg = None
    if SparseVectorParams is not None:
        # IDF modifier enum if available, else use string fallback
        modifier_val = Modifier.IDF if Modifier and hasattr(Modifier, "IDF") else "idf"
        sparse_cfg = {"bm25": SparseVectorParams(modifier=modifier_val)}
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        **({"sparse_vectors_config": sparse_cfg} if sparse_cfg else {}),
    )

UUID_RE = re.compile(
    r"^(?:[0-9]+|"
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
        pid = p.get("id")
        if not _is_valid_point_id(pid):
            raise ValueError(f"Invalid point id: {pid!r}")
    return client.upsert(collection_name=collection_name, points=pts_list, wait=wait)

# ---------------------------
# Payload builder
# ---------------------------

def build_payload(*, text: Optional[str] = None, content: Optional[str] = None, **metadata: Any) -> Dict[str, Any]:
    """
    Normalized payload for chunks:
      - top-level: 'text' (preferred; falls back to 'content'), plus any of
        'chunk_index', 'dim', 'type' if provided
      - nested: only 'filename' and 'category' under 'metadata'
    """
    payload: Dict[str, Any] = {}

    # Prefer 'content'; fallback to 'text'
    if content is not None:
        payload["content"] = content
    elif text is not None:
        payload["content"] = text

    # Lift selected fields to top-level if present
    for k in ("chunk_index", "dim", "type"):
        if k in metadata:
            payload[k] = metadata.pop(k)

    # Keep only filename & category under metadata
    meta_out: Dict[str, Any] = {}
    if "filename" in metadata:
        meta_out["filename"] = metadata.pop("filename")
    if "category" in metadata:
        meta_out["category"] = metadata.pop("category")

    payload["metadata"] = meta_out
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
    Construct a 'job_marker' point payload using the normalized schema.
    Stored entirely under payload['metadata'].
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
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
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
