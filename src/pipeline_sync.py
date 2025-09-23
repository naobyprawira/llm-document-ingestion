"""Synchronous document ingestion pipeline.

This module orchestrates the end-to-end ingestion flow: parsing PDFs or
images, cleaning and chunking the resulting markdown, embedding chunks
and writing them into Qdrant. All operations are synchronous; there
are no ``async def`` functions or semaphores. Concurrency is used only
within the parser to describe figures via Gemini (see
``parser_utils._describe_figures``).
"""

from __future__ import annotations

import os
import time
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from .parser_utils import parse_document
from .chunk import chunk_markdown
from .embed import embed_chunks
from .qdrant_utils import qdrant_client
from .logger import get_logger, CHUNK_MAX_CHARS


def process_file(
    *,
    data: bytes,
    filename: str,
    ext: str,
    category: str,
    collection: str,
    job_key: str,
) -> Dict[str, Any]:
    """Run the synchronous ingestion pipeline over a document."""
    log = get_logger(job=job_key, file=filename, phase="process")
    t_overall = time.perf_counter()

    # -------- PARSE --------
    if ext == ".pdf":
        # Docling wants a path
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
        # Image bytes directly
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

    # -------- EMBED & UPSERT (BATCH, ONE CALL) --------
    try:
        client = qdrant_client()
        if chunks:
            # Prefer file_type from parser metadata; fall back to extension value (without dot)
            file_type = (metadata.get("file_type") or ext.lstrip(".") or "").lower()

            uploaded = embed_chunks(
                chunks=chunks,
                filename=filename,      # embed.py will strip extension for storage
                category=category,
                collection=collection,
                job_key=job_key,        # UUID
                client=client,
                file_type=file_type,    # used in payload + index
                wait=True,
            )
        else:
            log.with_phase("embed").warning("No chunks to embed; skipping Qdrant upsert.")
    except Exception as exc:
        log.with_phase("embed").warning(f"Skipping Qdrant upsert: {exc}")

    total_seconds = time.perf_counter() - t_overall

    # Compose result (also return basename without extension)
    filename_base = os.path.splitext(filename)[0]
    result: Dict[str, Any] = {
        "file_name": filename_base,             # << no extension in API response
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
