"""
Embedding utilities for text chunks.

This module exposes helpers for obtaining embeddings from Google's Gen
AI models and upserting them into a Qdrant collection.  The primary
entry point used by the synchronous pipeline is :func:`get_genai_embedding`.
The rest of the functions mirror the upstream implementation and may
be used directly when embedding from the command line.  All functions
here are synchronous.

Environment variables:

``GOOGLE_API_KEY``
    API key for Google GenAI (mandatory).
``GEMINI_API_KEY``
    Alias for ``GOOGLE_API_KEY``; retained for backward compatibility.
``QDRANT_URL`` and ``QDRANT_API_KEY``
    Connection settings for Qdrant.  If ``QDRANT_URL`` is not set the
    upsert helpers will raise a ``RuntimeError``.
``QDRANT_COLLECTION``
    Default collection name when none is supplied.
"""

from __future__ import annotations

import os
import uuid
import json
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    # Minimal fallback when tqdm isn't installed
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def get_genai_embedding(text: str, model_name: str = "gemini-embedding-001") -> List[float]:
    """Return an embedding vector for ``text`` using Google’s GenAI.

    This helper wraps the GenAI SDK and automatically injects the API key
    from the environment.  The default model is ``gemini-embedding-001``,
    which provides high-quality embeddings and is widely supported.  The
    model can be overridden via the ``model_name`` argument.

    :param text: The text to embed.
    :param model_name: Name of the embedding model (default: ``gemini-embedding-001``).
    :return: A list of floats representing the embedding vector.
    :raises ImportError: If ``google-genai`` is not installed.
    :raises RuntimeError: If the API response does not contain embeddings.
    """
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. Install with: pip install google-genai"
        ) from exc
    # Retrieve API key from either GOOGLE_API_KEY or GENAI_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GENAI_API_KEY) environment variable is not set")
    # Initialise the client using the API key
    client = genai.Client(api_key=api_key)
    # Perform the embedding request
    resp = client.models.embed_content(model=model_name, contents=[text])
    embeddings = getattr(resp, "embeddings", None)
    if not embeddings:
        raise RuntimeError("GenAI embed_content returned no embeddings.")
    vec = getattr(embeddings[0], "values", None)
    if vec is None:
        raise RuntimeError("Embedding response missing 'values'.")
    return vec


def upsert_vectors(
    chunks: List[str],
    vectors: List[List[float]],
    collection: str,
    *,
    id_prefix: Optional[str] = None,
    use_uuid: bool = True,
) -> None:
    """Upsert a list of vectors into Qdrant with basic payloads.

    :param chunks: The text chunks corresponding to the vectors.
    :param vectors: The embedding vectors (must be the same length as ``chunks``).
    :param collection: Name of the Qdrant collection to write to.
    :param id_prefix: Optional prefix used when ``use_uuid`` is ``False``.
    :param use_uuid: Generate random UUIDs for point IDs when ``True``, else
        deterministic IDs based on ``id_prefix`` and the chunk index.
    :raises ImportError: If ``qdrant-client`` is not installed.
    :raises RuntimeError: If ``QDRANT_URL`` is not set.
    """
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is not installed. Install with: pip install qdrant-client"
        ) from exc
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    if not vectors:
        return
    vector_size = len(vectors[0])
    # Create collection if it doesn't already exist
    try:
        client.get_collection(collection_name=collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    points: List[Dict[str, Any]] = []
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        if use_uuid:
            point_id = uuid.uuid4().hex
        else:
            point_id = f"{id_prefix or 'doc'}-{idx}"
        points.append(
            {
                "id": point_id,
                "vector": vec,
                "payload": {
                    "text": text,
                    "source": id_prefix,
                    "chunk_index": idx,
                    "dim": len(vec),
                },
            }
        )
    client.upsert(collection_name=collection, points=points, wait=True)


def embed_and_upload_json(
    chunks: List[str],
    collection: str = os.getenv("QDRANT_COLLECTION", "documents"),
    model_name: str = "gemini-embedding-001",
    *,
    id_prefix: Optional[str] = None,
    use_uuid: bool = True,
) -> int:
    """Embed a list of text chunks and upload to Qdrant.

    :param chunks: The text chunks to embed.
    :param collection: Qdrant collection name.  Defaults to
        ``QDRANT_COLLECTION`` or ``documents``.
    :param model_name: Embedding model name.
    :param id_prefix: Prefix used when ``use_uuid`` is ``False``.
    :param use_uuid: Use random UUIDs for IDs when ``True``.
    :return: The number of uploaded embeddings.
    """
    if not chunks:
        return 0
    vectors: List[List[float]] = []
    for ch in tqdm(chunks, desc="Embedding chunks"):
        vectors.append(get_genai_embedding(ch, model_name=model_name))
    upsert_vectors(chunks, vectors, collection=collection, id_prefix=id_prefix, use_uuid=use_uuid)
    return len(chunks)


def embed_and_upload_file(
    input_path: str,
    collection: str,
    model_name: str = "gemini-embedding-001",
    *,
    use_uuid: bool = True,
) -> int:
    """Load chunks from a JSON file, embed them and upload to Qdrant.

    :param input_path: Path to a JSON file containing a list of strings.
    :param collection: Qdrant collection name.
    :param model_name: Name of the embedding model.
    :param use_uuid: Use random UUID IDs when ``True``; else use deterministic IDs.
    :return: Number of uploaded embeddings.
    """
    chunks = load_chunks(input_path)
    id_prefix = None
    if not use_uuid:
        id_prefix = os.path.splitext(os.path.basename(input_path))[0]
    return embed_and_upload_json(
        chunks,
        collection=collection,
        model_name=model_name,
        id_prefix=id_prefix,
        use_uuid=use_uuid,
    )


def load_chunks(path: str) -> List[str]:
    """Load a list of text chunks from a JSON file.

    :param path: Path to a JSON file containing a list of strings.
    :return: A list of strings.
    :raises ValueError: If the file does not contain a list.
    """
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