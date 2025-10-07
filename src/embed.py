"""
Embedding utilities for text chunks.

This module exposes helpers for obtaining embeddings from Google's GenAI
and upserting them into a Qdrant collection.  All functions are synchronous.
"""

from __future__ import annotations

import os
import uuid
import json
from typing import List, Dict, Any, Optional

from tqdm import tqdm  # type: ignore

from .qdrant_utils import ensure_collection, build_payload

# ---------------------------
# GenAI embedding
# ---------------------------

def _get_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY/GENAI_API_KEY")
    return api_key


def get_genai_embedding(text: str, model_name: str = "gemini-embedding-001") -> List[float]:
    """Request an embedding from Google GenAI for a single text."""
    try:
        from google import genai  # type: ignore
    except Exception as exc:
        raise ImportError("google-genai client is required") from exc

    api_key = _get_api_key()
    client = genai.Client(api_key=api_key)
    resp = client.models.embed_content(model=model_name, contents=[text])
    embeddings = getattr(resp, "embeddings", None)
    if not embeddings:
        raise RuntimeError("GenAI embed_content returned no embeddings.")
    vec = getattr(embeddings[0], "values", None)
    if vec is None:
        raise RuntimeError("Embedding response missing 'values'.")
    return vec

# ---------------------------
# Qdrant upsert helpers
# ---------------------------

def upsert_vectors(
    chunks: List[str],
    vectors: List[List[float]],
    collection: str,
    *,
    id_prefix: Optional[str] = None,
    use_uuid: bool = True,
    filename: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Upsert a list of vectors into Qdrant with normalized payloads."""
    try:
        from qdrant_client import QdrantClient, models  # type: ignore
    except ImportError as exc:
        raise ImportError("qdrant-client is not installed. pip install qdrant-client") from exc

    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))

    if not vectors:
        return

    vector_size = len(vectors[0])
    # Ensure collection exists (dense + sparse BM25/IDF)
    try:
        ensure_collection(client, collection, vector_size)
    except Exception:
        # Fallback to simple dense collection if ensure fails
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    # Ensure keyword indexes for nested metadata (idempotent)
    try:
        client.create_payload_index(collection_name=collection, field_name="metadata.category", field_schema="keyword", wait=True)
        client.create_payload_index(collection_name=collection, field_name="metadata.filename", field_schema="keyword", wait=True)
    except Exception:
        pass

    points: List[Dict[str, Any]] = []
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        point_id = uuid.uuid4().hex if use_uuid else f"{id_prefix or 'doc'}-{idx}"
        payload = build_payload(
            content=text,
            filename=(filename or id_prefix or "unknown"),
            category=(category or os.getenv("DOC_CATEGORY", "Uncategorized")),
            chunk_index=idx,
            dim=len(vec),
            type="chunk",
        )
        points.append({"id": point_id, "vector": vec, "payload": payload})

    client.upsert(collection_name=collection, points=points, wait=True)

def embed_and_upload_json(
    chunks: List[str],
    collection: str = os.getenv("QDRANT_COLLECTION", "documents"),
    model_name: str = "gemini-embedding-001",
    *,
    id_prefix: Optional[str] = None,
    use_uuid: bool = True,
    filename: Optional[str] = None,
    category: Optional[str] = None,
) -> int:
    """Embed a list of text chunks and upload to Qdrant."""
    if not chunks:
        return 0
    vectors: List[List[float]] = []
    for ch in tqdm(chunks, desc="Embedding chunks"):
        vectors.append(get_genai_embedding(ch, model_name=model_name))
    upsert_vectors(
        chunks,
        vectors,
        collection=collection,
        id_prefix=id_prefix,
        use_uuid=use_uuid,
        filename=filename,
        category=category,
    )
    return len(chunks)

def embed_and_upload_file(
    input_path: str,
    collection: str,
    model_name: str = "gemini-embedding-001",
    *,
    use_uuid: bool = True,
) -> int:
    """Load chunks from a JSON file, embed them and upload to Qdrant."""
    chunks = load_chunks(input_path)
    id_prefix = None
    if not use_uuid:
        id_prefix = os.path.splitext(os.path.basename(input_path))[0]
    filename = os.path.splitext(os.path.basename(input_path))[0]
    category = os.getenv("DOC_CATEGORY", "Uncategorized")
    return embed_and_upload_json(
        chunks,
        collection=collection,
        model_name=model_name,
        id_prefix=id_prefix,
        use_uuid=use_uuid,
        filename=filename,
        category=category,
    )

def load_chunks(path: str) -> List[str]:
    """Load a list of text chunks from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of strings.")
    return [str(x) for x in data]

__all__ = [
    "get_genai_embedding",
    "upsert_vectors",
    "embed_and_upload_json",
    "embed_and_upload_file",
]
