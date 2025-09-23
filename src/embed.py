"""Embedding utilities for text chunks.

Public API:
- get_genai_embedding(text, model_name=...) -> List[float]
- embed_chunks(...) -> int  # embed ALL chunks and upsert in ONE call

Notes:
- Vector search uses the dedicated `vector` field in Qdrant (not payload).
- We store the chunk `text` + rich metadata in payload for retrieval & filters.
- Payload indexes are created for: filename, job_key, file_type, category.
- We now strip file extensions from the stored `filename` (basename only).
"""

from __future__ import annotations

from typing import List
import os
import uuid
import time

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import PointStruct  # type: ignore
    try:
        from qdrant_client.http import models as rest  # type: ignore
        KEYWORD_SCHEMA = getattr(rest.PayloadSchemaType, "KEYWORD", "keyword")
    except Exception:
        KEYWORD_SCHEMA = "keyword"
    _HAS_QDRANT = True
except Exception:
    QdrantClient = object  # type: ignore
    PointStruct = object  # type: ignore
    KEYWORD_SCHEMA = "keyword"
    _HAS_QDRANT = False

from .qdrant_utils import ensure_collection, marker_point
from .logger import get_logger


# --- Google GenAI embedding (mirrors upstream call style) ---
def get_genai_embedding(text: str, model_name: str | None = None) -> List[float]:
    """
    Return a single embedding vector for `text` using google-genai.

    Uses client.models.embed_content(). Default model can be set via
    GENAI_EMBED_MODEL; fallback 'gemini-embedding-001'.
    """
    from google import genai  # type: ignore

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GENAI_API_KEY) is not set")

    client = genai.Client(api_key=api_key)
    model = model_name or os.getenv("GENAI_EMBED_MODEL", "gemini-embedding-001")

    response = client.models.embed_content(model=model, contents=[text])
    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise RuntimeError("Gen AI embed_content returned no embeddings.")
    vec = getattr(embeddings[0], "values", None)
    if vec is None:
        raise RuntimeError("Embedding response missing 'values'.")
    return vec


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """
    Create payload indexes for fields we filter on.
    Safe to call repeatedly; server-side creation is idempotent.
    """
    # Add "type" so filters like {"key":"type","match":{"value":"job_marker"}} work
    fields = ("filename", "job_key", "file_type", "category", "type")

    for field in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=KEYWORD_SCHEMA,  # "keyword" schema for exact-match filters
                wait=True,
            )
        except Exception:
            # Ignore "already exists" or similar benign errors
            pass


def embed_chunks(
    *,
    chunks: List[str],
    filename: str,
    category: str,
    collection: str,
    job_key: str,                  # UUID string
    client: QdrantClient,
    file_type: str | None = None,  # e.g., "pdf", "png"
    wait: bool = True,
) -> int:
    """
    Embed ALL chunks and upload them to Qdrant in a single upsert call.

    - Stores the embedding in point.vector (for similarity search).
    - Stores `text` and rich metadata in payload for retrieval & filtering.
    - Optionally duplicates the embedding into payload if EMBED_IN_PAYLOAD=1.
    - Stores `filename` **without extension** (basename).
    """
    log = get_logger(job=job_key, file=filename, phase="embed")

    if not chunks:
        log.warning("No chunks to embed; skipping upsert.")
        return 0

    t0 = time.perf_counter()

    # Normalize filenames: remove extension for stored name
    filename_base = os.path.splitext(filename)[0]  # root without extension

    # 1) Prepare collection + indexes (+ marker)
    marker_vec = get_genai_embedding("job_marker")
    ensure_collection(client, collection, len(marker_vec))
    _ensure_payload_indexes(client, collection)

    # Initial progress marker (running, 0 uploaded) — use basename
    client.upsert(
        collection_name=collection,
        points=[
            marker_point(
                job_key=job_key,
                vector=marker_vec,
                filename=filename_base,   # << no extension stored
                category=category,
                collection=collection,
                status="running",
                expected_chunks=len(chunks),
                uploaded_chunks=0,
            )
        ],
        wait=True,
    )

    # 2) Embed all chunks synchronously
    embed_model = os.getenv("GENAI_EMBED_MODEL", "gemini-embedding-001")
    vectors = [get_genai_embedding(text, model_name=embed_model) for text in chunks]

    # 3) Build points; include all indexed fields + text + metadata
    ft = (file_type or os.path.splitext(filename)[1].lstrip(".") or "").lower()
    include_vec_in_payload = (os.getenv("EMBED_IN_PAYLOAD", "0") == "1")
    now_ts = int(time.time())

    points: List[PointStruct] = []
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            "text": text,
            "category": category,
            "text_len": len(text),
            "filename": filename_base,
            "file_type": ft,# << no extension stored
            "job_key": job_key,
            "chunk_index": idx,
            "embedding_model": embed_model,
            "embedding_dim": len(vec),
            "embedding_ts": now_ts,
        }
        if include_vec_in_payload:
            payload["embedding"] = vec  # optional duplication; off by default

        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,          # used for vector search
                payload=payload,     # text + metadata
            )
        )

    # 4) Single upsert with all points
    client.upsert(collection_name=collection, points=points, wait=wait)

    # 5) Final progress marker (done)
    client.upsert(
        collection_name=collection,
        points=[
            marker_point(
                job_key=job_key,
                vector=marker_vec,
                filename=filename_base,   # << no extension stored
                category=category,
                collection=collection,
                status="done",
                expected_chunks=len(chunks),
                uploaded_chunks=len(points),
            )
        ],
        wait=True,
    )

    log.info(
        f"embed_chunks done uploaded={len(points)} took={time.perf_counter()-t0:.3f}s"
    )
    return len(points)


__all__ = [
    "get_genai_embedding",
    "embed_chunks",
]
