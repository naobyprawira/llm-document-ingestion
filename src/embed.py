#!/usr/bin/env python3
"""
Embed pre‑chunked text and load embeddings into Qdrant.

This module exposes high‑level helpers for taking a list of text
``chunks`` in memory, obtaining embeddings via Google’s Gen AI SDK
(``GeminiVLM``) and upserting the resulting vectors into a Qdrant
collection. It can also load chunks from a JSON file on disk. These
helpers are used by the API layer to persist processed documents.

Environment variables:

``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
    The key used by the Google Gen AI SDK. At least one of these must
    be set or calls to ``get_genai_embedding`` will raise an error.

``QDRANT_URL`` (required) and ``QDRANT_API_KEY`` (optional)
    Connection information for the Qdrant instance. If ``QDRANT_URL``
    is not set, ``upsert_vectors`` will raise a ``RuntimeError``.

``QDRANT_COLLECTION`` (optional)
    Default collection name used when not explicitly supplied.

The core logic here mirrors the upstream implementation; it has not
been modified beyond formatting and comments for clarity.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    # Define a no-op load_dotenv if python-dotenv is not installed
    def load_dotenv(*args, **kwargs):  # type: ignore
        return None

# Ensure environment variables are loaded before using google/qdrant clients
load_dotenv()

# Lightweight progress bar fallback – tqdm is optional
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def load_chunks(path: str) -> List[str]:
    """Load a list of text chunks from a JSON file.

    :param path: Path to a JSON file containing a list of strings.
    :return: List of strings extracted from the JSON list.
    :raises ValueError: If the file does not contain a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of strings.")
    return [str(x) for x in data]


def get_genai_embedding(text: str, model_name: str = "gemini-embedding-001") -> List[float]:
    """Obtain an embedding vector for the given text using Google's Gen AI SDK.

    :param text: The text to embed.
    :param model_name: Embedding model (e.g. ``gemini-embedding-001`` or
        ``text-embedding-004``).
    :return: A list of floats representing the embedding vector.
    :raises ImportError: If google‑genai is not installed.
    :raises RuntimeError: If the API response does not contain embeddings.
    """
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. Install with: pip install google-genai"
        ) from exc

    client = genai.Client()  # uses GEMINI_API_KEY or GOOGLE_API_KEY from env
    resp = client.models.embed_content(model=model_name, contents=[text])
    embeddings = getattr(resp, "embeddings", None)
    if not embeddings:
        raise RuntimeError("Gen AI embed_content returned no embeddings.")
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
    """Upsert vectors into a Qdrant collection with payloads.

    :param chunks: The text chunks that were embedded.
    :param vectors: The corresponding embedding vectors.
    :param collection: Name of the Qdrant collection.
    :param id_prefix: Optional prefix used when ``use_uuid`` is False to
        generate stable IDs based on the document name.
    :param use_uuid: If True, generate random UUIDs for each vector ID; if
        False, use deterministic ``-`` IDs.
    """
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models   # type: ignore
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

    # Build points with unique IDs
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
                    "content": text,
                    "source": id_prefix,
                    "chunk_index": idx,
                    "dim": len(vec),
                },
            }
        )

    # Upsert points; wait=True ensures write completes before returning
    client.upsert(collection_name=collection, points=points, wait=True)


def embed_and_upload_json(
    chunks: List[str],
    collection: str = os.getenv("QDRANT_COLLECTION", "documents"),
    model_name: str = "gemini-embedding-001",
    *,
    id_prefix: Optional[str] = None,
    use_uuid: bool = True,
) -> int:
    """Embed an in‑memory list of text chunks and upload to Qdrant.

    :param chunks: List[str] of text to embed.
    :param collection: Qdrant collection name. Defaults to the
        ``QDRANT_COLLECTION`` env var or ``documents``.
    :param model_name: Name of the embedding model.
    :param id_prefix: If ``use_uuid`` is False, prefix used to build stable
        IDs based on the document name.
    :param use_uuid: Use randomly generated UUID IDs when True; else use
        deterministic IDs.
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
    """Load chunks from a JSON file, embed them, and upload to Qdrant.

    :param input_path: Path to a JSON file containing a list of strings.
    :param collection: Qdrant collection name.
    :param model_name: Embedding model name.
    :param use_uuid: Use randomly generated UUID IDs when True; else use
        deterministic ``-`` IDs.
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


def main() -> None:
    """CLI entrypoint for embedding chunks from a file and uploading them."""
    parser = argparse.ArgumentParser(description="Embed chunks and upload to Qdrant.")
    parser.add_argument("--input", required=True, help="Path to JSON file with text chunks")
    parser.add_argument(
        "--collection",
        required=False,
        default=os.environ.get("QDRANT_COLLECTION"),
        help="Qdrant collection name (or set QDRANT_COLLECTION env var)",
    )
    parser.add_argument(
        "--model",
        default="gemini-embedding-001",
        help="Embedding model name (e.g., gemini-embedding-001, text-embedding-004)",
    )
    parser.add_argument(
        "--id-scheme",
        choices=["uuid", "prefix"],
        default="uuid",
        help="Point ID scheme: 'uuid' (default) or 'prefix' (=<file>-<idx>).",
    )
    args = parser.parse_args()

    if not args.collection:
        raise SystemExit("ERROR: --collection is required (or set QDRANT_COLLECTION).")

    use_uuid = args.id_scheme == "uuid"
    count = embed_and_upload_file(
        input_path=args.input,
        collection=args.collection,
        model_name=args.model,
        use_uuid=use_uuid,
    )
    print(f"Uploaded {count} embeddings to Qdrant collection '{args.collection}'.")


if __name__ == "__main__":  # pragma: no cover
    main()