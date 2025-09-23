"""FastAPI application exposing a single `/process` endpoint.

The endpoint accepts a PDF or image file and a category.  When called
without ``async_mode`` it blocks until the document has been fully
ingested (parse → describe → chunk → embed) and returns a JSON
structure describing the ingestion.  When ``async_mode=1`` is passed
as a form field the call returns immediately with status ``202`` and
continues the ingestion in a background thread.  Progress for
asynchronous jobs is written to Qdrant as a ``job_marker`` point.

The ingestion pipeline itself lives in :mod:`pipeline_sync` and is
completely synchronous; only the visual language model calls are
parallelised internally.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .pipeline_sync import process_file
from .logger import get_logger


app = FastAPI(title="Document Ingestion API (synchronous)")


@app.post("/process")
def process_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    async_mode: str | None = Form(default=None),
) -> JSONResponse:
    """Handle a file upload and run the ingestion pipeline.

    The ``category`` field is an arbitrary string that is stored in Qdrant
    alongside the vectors.  If ``async_mode`` is ``1`` or one of
    ``{"true", "yes", "on"}`` then the pipeline is executed in a
    background thread and a ``202`` response is returned immediately.
    Otherwise the call blocks until ingestion completes and returns a
    ``200`` with basic metadata about the processed document.  Errors
    during processing are surfaced as ``HTTPException`` instances.
    """
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()
    # Read the file up front – FastAPI deletes temporary files once the response is sent
    try:
        data = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    job_key = uuid.uuid4().hex
    is_async = False
    if async_mode:
        val = str(async_mode).strip().lower()
        if val in {"1", "true", "yes", "on"}:
            is_async = True

    if is_async:
        # Asynchronous mode: schedule the job and return immediately
        log = get_logger(job=job_key, file=filename, phase="accept")
        log.info("Accepted async job")
        # Use a keyword-only call to make intent explicit
        background_tasks.add_task(
            process_file,
            data=data,
            filename=filename,
            ext=ext,
            category=category,
            collection=os.getenv("QDRANT_COLLECTION", "documents"),
            job_key=job_key,
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "job_key": job_key,
                "filename": filename,
                "category": category,
                "collection": os.getenv("QDRANT_COLLECTION", "documents"),
            },
        )

    # Synchronous mode: run the pipeline and block until done
    try:
        res = process_file(
            data=data,
            filename=filename,
            ext=ext,
            category=category,
            collection=os.getenv("QDRANT_COLLECTION", "documents"),
            job_key=job_key,
        )
    except Exception as exc:
        get_logger(job=job_key, file=filename, phase="error").exception(f"Pipeline failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Remove the markdown from the JSON response – clients can fetch it separately if needed
    response_data = {k: v for k, v in res.items() if k != "markdown"}
    return JSONResponse(content=response_data)


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Return basic health information and configuration presence."""
    info = {
        "python": os.sys.version.split()[0],
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
        "max_concurrent_vlm": os.getenv("MAX_CONCURRENT_VLM", "5"),
    }
    return JSONResponse(content=info)


__all__ = ["app"]