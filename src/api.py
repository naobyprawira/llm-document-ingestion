"""FastAPI application exposing a single `/process` endpoint.

Flow:
1) Pre-check in Qdrant to avoid duplicate work.
   - Primary: retrieve by deterministic marker UUID (collection + filename sans ext).
   - Fallback: scroll by payload (metadata.filename [+ metadata.type='job_marker']) after
     ensuring keyword payload indexes (idempotent).

2) If already exists:
   - conflict|ok|skip behavior

3) If not exists:
   - Default async background ingestion; optional sync if async_mode=0.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional, Iterable

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from .pipeline_sync import process_file
from .qdrant_utils import qdrant_client
from .logger import get_logger
from .bm25_endpoint import attach_bm25_endpoints
from .retrieve import attach_retrieval_endpoints


app = FastAPI(title="Document Ingestion API (default async)")
attach_bm25_endpoints(app)
attach_retrieval_endpoints(app)

# ---------------------------
# Helpers
# ---------------------------

def _marker_id_for_filename(collection: str, filename_no_ext: str) -> str:
    """Deterministic UUIDv5 marker id based on collection + filename (sans extension)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{filename_no_ext.lower()}"))


def _idempotent_create_indexes(client, collection: str, fields: Iterable[str]) -> None:
    """
    Create keyword payload indexes for given fields (safe to call repeatedly).

    Matches your payload_schema:
      - content
      - metadata.category
      - metadata.filename
      - metadata.chunk_index
      - metadata.dim

    Plus 'metadata.type' (tiny extra) to keep job_marker filter fast/stable.
    """
    for field in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema="keyword",
                wait=True,
            )
        except Exception:
            # ignore "already exists" or transient errors
            pass


def _run_ingestion_bg(
    *,
    data: bytes,
    filename: str,
    ext: str,
    category: str,
    collection: str,
    job_key: str,
) -> None:
    """Background ingestion runner; logs failures instead of raising."""
    log = get_logger(job=job_key, file=filename, phase="bg")
    try:
        process_file(
            data=data,
            filename=filename,
            ext=ext,
            category=category,
            collection=collection,
            job_key=job_key,
        )
        log.info("background ingestion completed")
    except Exception as e:
        log.error(f"background ingestion failed: {e}")


# ---------------------------
# FastAPI routes
# ---------------------------

@app.post("/process")
async def process_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    collection: str = Form(...),
    job_key: Optional[str] = Form(None),
    # How to signal "already exists": conflict|ok|skip (default = conflict)
    exists_mode: str = Query("conflict", pattern="^(ok|conflict|skip)$"),
    # Single optional async toggle (default async=1)
    async_mode: int = Query(1, ge=0, le=1),
):
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()
    filename_no_ext = os.path.splitext(filename)[0]

    client = qdrant_client()

    # Deterministic, valid Qdrant point ID (UUIDv5) for the marker
    marker_id = _marker_id_for_filename(collection, filename_no_ext)

    # ---------- Pre-check (1): retrieve by UUID marker id (O(1)) ----------
    try:
        pts = client.retrieve(
            collection_name=collection,
            ids=[marker_id],
            with_payload=False,
            with_vectors=False,
        )
    except Exception:
        pts = []

    # ---------- Pre-check (2): fallback scroll by payload filter ----------
    if not pts:
        try:
            # Ensure indexes match your payload_schema (plus metadata.type for markers)
            _idempotent_create_indexes(
                client,
                collection,
                fields=(
                    "metadata.category",
                    "metadata.filename",
                    "type",  # for job_marker filter
                ),
            )
            records, _ = client.scroll(
                collection_name=collection,
                scroll_filter={
                    "must": [
                        {"key": "metadata.filename", "match": {"value": filename_no_ext}},
                        {"key": "metadata.type", "match": {"value": "job_marker"}},
                    ]
                },
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            records = []
    else:
        records = pts

    if records:  # already ingested -> signal based on exists_mode
        payload = {
            "status": "exists",
            "action": "conflict" if exists_mode == "conflict" else "skip",
            "file_name": filename_no_ext,
            "collection": collection,
            "reason": "document_with_same_filename_already_indexed",
            "marker_id": marker_id,
            "category": category,
            "job_key": job_key,
        }
        if exists_mode == "conflict":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=payload)
        elif exists_mode == "skip":
            return JSONResponse({"status": "exists", "action": "skip", "file_name": filename_no_ext}, status_code=200)
        else:  # ok
            return JSONResponse(payload, status_code=200)

    # ---------- Not found: proceed ----------
    is_async = (async_mode == 1)

    data = await file.read()
    job_key_final = job_key or uuid.uuid4().hex

    if is_async:
        background_tasks.add_task(
            _run_ingestion_bg,
            data=data,
            filename=filename,
            ext=ext,
            category=category,
            collection=collection,
            job_key=job_key_final,
        )
        return JSONResponse(
            {
                "status": "accepted",
                "async_mode": 1,
                "file_name": filename_no_ext,
                "collection": collection,
                "category": category,
                "job_key": job_key_final,
            },
            status_code=status.HTTP_202_ACCEPTED,
        )

    # Sync path
    result = process_file(
        data=data,
        filename=filename,
        ext=ext,
        category=category,
        collection=collection,
        job_key=job_key_final,
    )
    return JSONResponse(result, status_code=200)


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Return basic health information and configuration presence."""
    info = {
        "python": os.sys.version.split()[0],
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "embed_model": os.getenv("GENAI_EMBED_MODEL", "gemini-embedding-001"),
        "api_key_present": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")),
        "max_concurrent_vlm": os.getenv("MAX_CONCURRENT_VLM", "5"),
        "default_async": True,
        "exists_mode_default": "conflict",
    }
    return JSONResponse(content=info)


__all__ = ["app"]
