"""Synchronous document ingestion pipeline.

This module orchestrates the end‑to‑end ingestion flow: parsing PDFs or
images, cleaning and chunking the resulting markdown, embedding chunks
and writing them into Qdrant.  All operations are synchronous; there
are no ``async def`` functions or semaphores.  Concurrency is used only
within the parser to describe figures via Gemini (see
``parser_utils._describe_figures``).
"""

from __future__ import annotations

import os
import time
import uuid
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .parser_utils import parse_document
from .chunk import chunk_markdown
from .embed import get_genai_embedding
from .qdrant_utils import qdrant_client, ensure_collection, marker_point
from .logger import get_logger, EMBED_BATCH_SIZE, CHUNK_MAX_CHARS


def _batched(seq: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive ``size``-length lists from ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def process_file(
    *,
    data: bytes,
    filename: str,
    ext: str,
    category: str,
    collection: str,
    job_key: str,
) -> Dict[str, Any]:
    """Run the synchronous ingestion pipeline over a document.

    :param data: Raw bytes of the uploaded file (PDF or image).
    :param filename: The original filename for logging.
    :param ext: File extension (lowercase, including dot, e.g. `.pdf`).
    :param category: Arbitrary category tag provided by the client.
    :param collection: Name of the Qdrant collection to upsert into.
    :param job_key: Unique identifier for this ingestion job.
    :return: A dictionary containing metadata about the processed document and the enriched markdown.
    """
    log = get_logger(job=job_key, file=filename, phase="process")
    t_overall = time.perf_counter()

    # Determine the document identifier (basename without extension).  This
    # value is used as the marker ID so that duplicate files can be
    # detected prior to ingestion.  We keep the original filename with
    # extension for logging and for returning to clients.
    doc_id = os.path.splitext(filename)[0]

    # -------- PARSE --------
    if ext == ".pdf":
        # Write to a temporary file because Docling requires a path
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            markdown, metadata = parse_document(tmp_path, filename, ext, job_key=job_key)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    else:
        # Image inputs provide the raw bytes directly
        markdown, metadata = parse_document("", filename, ext, image_bytes=data, job_key=job_key)
    log.with_phase("parse").info(
        f"Completed parse stage in {time.perf_counter() - t_overall:.3f}s"
    )

    # -------- CHUNK --------
    t_chunk = time.perf_counter()
    chunks: List[str] = chunk_markdown(markdown, max_chars=CHUNK_MAX_CHARS)
    expected = len(chunks)
    log.with_phase("chunk").info(
        f"Completed chunk stage count={expected} in {time.perf_counter() - t_chunk:.3f}s"
    )

    uploaded = 0
    # -------- EMBED & UPSERT --------
    # Attempt to obtain a Qdrant client.  If this fails (e.g. Qdrant is
    # misconfigured), embedding and upserts will be skipped but the
    # remainder of the pipeline still runs.  This ensures that
    # ingestion cannot be completely blocked by connectivity issues.
    has_qdrant = False
    try:
        client = qdrant_client()
        has_qdrant = True
    except Exception as exc:
        client = None  # type: ignore
        log.with_phase("embed").warning(f"Skipping Qdrant upsert: {exc}")

    marker_vec: Optional[List[float]] = None
    if has_qdrant and chunks:
        # Prepare marker vector and ensure collection exists.  The
        # marker vector is a dummy embedding used only to satisfy the
        # Qdrant schema.  It is stored alongside the job marker to
        # record ingestion progress.
        marker_vec = get_genai_embedding("job_marker")
        ensure_collection(client, collection, len(marker_vec))
        # Create payload indexes for filename and category so that
        # downstream queries can efficiently filter points by these
        # fields.  These calls are idempotent: if an index already
        # exists Qdrant will ignore the request.
        for field_name in ("filename", "category"):
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema="keyword",
                    wait=True,
                )
            except Exception:
                pass
        # Upsert an initial job marker using the document ID as the
        # point ID.  This allows the API to skip reprocessing of
        # documents whose filename (without extension) already
        # exists in the collection.
        client.upsert(
            collection_name=collection,
            points=[
                marker_point(
                    job_key,
                    marker_vec,
                    doc_id,
                    category,
                    collection,
                    "running",
                    expected,
                    uploaded,
                )
            ],
            wait=True,
        )
        log.with_phase("embed").info(
            f"Started embedding: total_chunks={expected} batch_size={EMBED_BATCH_SIZE}"
        )

        chunk_index_offset = 0
        for batch in _batched(chunks, EMBED_BATCH_SIZE):
            t_batch = time.perf_counter()
            # Compute embeddings serially.  Note that these calls
            # respect the configured embedding model via the default
            # ``model_name`` parameter in ``get_genai_embedding``.
            vectors = [get_genai_embedding(text) for text in batch]
            # Build points for this batch.  Place 'text' and 'category'
            # first in the payload dictionary to make it easier to
            # visually inspect the stored objects in Qdrant’s UI.
            points: List[Dict[str, Any]] = []
            for i, (text_chunk, vec) in enumerate(zip(batch, vectors)):
                idx = chunk_index_offset + i
                payload = {
                    "text": text_chunk,
                    "category": category,
                    "filename": doc_id,
                    "chunk_index": idx,
                    "dim": len(vec),
                }
                points.append(
                    {
                        "id": uuid.uuid4().hex,
                        "vector": vec,
                        "payload": payload,
                    }
                )
            # Upsert this batch of points.  ``wait=True`` ensures that
            # Qdrant finishes writing the points before returning.
            client.upsert(collection_name=collection, points=points, wait=True)
            uploaded += len(points)
            # Update progress marker with the number of uploaded chunks.
            client.upsert(
                collection_name=collection,
                points=[
                    marker_point(
                        job_key,
                        marker_vec,
                        doc_id,
                        category,
                        collection,
                        "running",
                        expected,
                        uploaded,
                    )
                ],
                wait=True,
            )
            log.with_phase("embed").info(
                f"Batch uploaded={len(points)} total_uploaded={uploaded} took={time.perf_counter() - t_batch:.3f}s"
            )
            chunk_index_offset += len(batch)
        # Upsert final marker marking completion of the job.
        client.upsert(
            collection_name=collection,
            points=[
                marker_point(
                    job_key,
                    marker_vec,
                    doc_id,
                    category,
                    collection,
                    "done",
                    expected,
                    uploaded,
                )
            ],
            wait=True,
        )
        log.with_phase("embed").info(
            f"Completed embedding uploaded={uploaded} overall={time.perf_counter() - t_overall:.3f}s"
        )
    else:
        log.with_phase("embed").warning(
            "No embeddings were uploaded (no Qdrant client or no chunks)"
        )

    total_seconds = time.perf_counter() - t_overall
    # Compose result
    result: Dict[str, Any] = {
        "file_name": filename,
        "file_type": metadata.get("file_type"),
        "page_count": metadata.get("page_count"),
        "images_count": metadata.get("images_count"),
        "uploaded_chunks": uploaded,
        "collection": collection,
        "total_processing_seconds": total_seconds,
        "category": category,
        "markdown": markdown,
    }
    return result


__all__ = [
    "process_file",
]