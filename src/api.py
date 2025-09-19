# src/api.py
"""
FastAPI application exposing a single /process endpoint that supports:
- Synchronous mode (default): parse -> chunk -> embed, then respond with 200.
- Asynchronous mode (async=1): immediately 202 Accepted with job_key; heavy work continues
  in background using BackgroundTasks. Progress & completion are recorded as a "job_marker"
  point in Qdrant (payload-filterable), so clients can poll Qdrant without new endpoints.

Env (examples):
- MAX_CONCURRENT_PARSE: int, limit concurrent parse stages (default: 3)
- MAX_CONCURRENT_VLM:   int, limit concurrent VLM describe() calls (default: 5)
- EMBED_BATCH_SIZE:     int, number of chunks per Qdrant upsert batch (default: 64)
- QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION: Qdrant connection/config
- LOG_LEVEL:            DEBUG|INFO|WARNING|ERROR (default: INFO)
"""

from __future__ import annotations

import json
import os
import time
import uuid
import logging
import warnings
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any, Optional, Iterable

import anyio
import asyncio
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from pydantic import BaseModel  # type: ignore

from .doc_parser import DocParser, Figure
from .vlm import GeminiVLM, VLMConfig
from .prompts import default_prompt
from .markdown import FigureDesc, compose_markdown
from .chunk import chunk_markdown
from .embed import get_genai_embedding  # low-level embed for batching



# Silence torch CPU-only pin_memory warning (harmless when no accelerator)
warnings.filterwarnings(
    "ignore",
    message=r".*'pin_memory' argument is set as true but no accelerator is found.*",
)

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_base_logger = logging.getLogger("ingestion")
if not _base_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s job=%(job)s file=%(file)s phase=%(phase)s msg=%(message)s"
    )
    _handler.setFormatter(_formatter)
    _base_logger.addHandler(_handler)
_base_logger.setLevel(_LOG_LEVEL)


class _Adapter(logging.LoggerAdapter):
    """Injects job/file/phase keys so our formatter never KeyErrors."""
    def __init__(self, logger, job="-", file="-", phase="-"):
        super().__init__(logger, {"job": job, "file": file, "phase": phase})

    def with_phase(self, phase: str) -> "_Adapter":
        return _Adapter(self.logger, self.extra.get("job", "-"), self.extra.get("file", "-"), phase)


def _logger(job: str = "-", file: str = "-", phase: str = "-") -> _Adapter:
    return _Adapter(_base_logger, job, file, phase)
# -------------------------------------------------------------------------


# ====================== Concurrency controls ======================
MAX_CONCURRENT_PARSE = int(os.getenv("MAX_CONCURRENT_PARSE", "3"))
MAX_CONCURRENT_VLM   = int(os.getenv("MAX_CONCURRENT_VLM", "5"))
EMBED_BATCH_SIZE     = int(os.getenv("EMBED_BATCH_SIZE", "64"))

_PARSE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_PARSE)
_VLM_SEMAPHORE   = asyncio.Semaphore(MAX_CONCURRENT_VLM)
# ==================================================================


def _ensure_primitive(value: Any) -> Any:
    """Coerce to JSON-serializable primitive."""
    if callable(value):
        try:
            value = value()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {k: _ensure_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_primitive(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _extract_metadata_from_doc(doc: Any) -> Dict[str, Any]:
    """Extract common metadata from Docling document if present."""
    author: Optional[str] = None
    page_count: Optional[int] = None
    try:
        if hasattr(doc, "info"):
            info = getattr(doc, "info")
            if hasattr(info, "authors"):
                try:
                    authors_val = getattr(info, "authors")
                    authors = _ensure_primitive(authors_val)
                    if authors:
                        if isinstance(authors, (list, tuple, set)):
                            author = ", ".join([str(a) for a in authors])
                        else:
                            author = str(authors)
                except Exception:
                    pass
            if author is None and hasattr(info, "author"):
                try:
                    val = getattr(info, "author")
                    primitive = _ensure_primitive(val)
                    if primitive:
                        author = str(primitive)
                except Exception:
                    pass
        for attr in ["page_count", "n_pages", "num_pages"]:
            if page_count is not None:
                break
            if hasattr(doc, attr):
                try:
                    val = getattr(doc, attr)
                    primitive = _ensure_primitive(val)
                    if primitive is not None:
                        try:
                            page_count = int(primitive)  # type: ignore[assignment]
                        except Exception:
                            page_count = None
                except Exception:
                    pass
    except Exception:
        pass
    return {"author": author, "page_count": page_count}


async def _parse_document(
    tmp_path: str,
    filename: str,
    ext: str,
    image_bytes: bytes | None = None,
    *,
    job_key: str,
) -> tuple[str, Dict[str, Any]]:
    """Parse a document or image and return (markdown, metadata)."""
    log = _logger(job=job_key, file=filename, phase="parse")
    t0 = time.perf_counter()

    metadata: Dict[str, Any] = {"filename": filename, "file_type": ext.lstrip(".")}
    markdown: str = ""

    # Try to hint VLM to disable AFC if supported by your wrapper's config
    cfg = VLMConfig()
    if hasattr(cfg, "disable_afc"):
        try:
            setattr(cfg, "disable_afc", True)
        except Exception:
            pass

    if ext == ".pdf":
        log.info("start: docling.parse")
        parser = DocParser()
        # anyio.to_thread.run_sync is the recommended pattern for sync work in async apps
        doc, base_md, figures = await anyio.to_thread.run_sync(parser.parse, tmp_path)
        meta_extra = _extract_metadata_from_doc(doc)
        metadata.update(meta_extra)

        valid_figures = [f for f in figures if len(f.jpeg_bytes) >= 5000]
        metadata["images_count"] = len(valid_figures)

        vlm = GeminiVLM(cfg)
        prompt = default_prompt()

        async def describe_one(f: Figure) -> FigureDesc:
            async with _VLM_SEMAPHORE:
                # ONE call only – no local retries based on caption length.
                text = await anyio.to_thread.run_sync(lambda: vlm.describe(f.jpeg_bytes, prompt))
                log.info(f"vlm.describe: figure {f.index} completed, text_length={len(text or '')}")
                return FigureDesc(index=f.index, page=f.page, caption=f.caption, description=text or "")

        if valid_figures:
            log.info(f"vlm.describe: start, figures={len(valid_figures)} (throttled={MAX_CONCURRENT_VLM})")
            descs: List[FigureDesc] = await asyncio.gather(*(describe_one(f) for f in valid_figures))
        else:
            descs = []
        markdown = compose_markdown(base_md, descs)
        log.info(f"done: docling+vlm in {time.perf_counter()-t0:.3f}s pages={metadata.get('page_count')} images={metadata.get('images_count')}")
    else:
        log.info("start: image.describe")
        metadata.update({"page_count": 1, "images_count": 1, "author": None})
        data = image_bytes if image_bytes is not None else open(tmp_path, "rb").read()
        vlm = GeminiVLM(cfg)
        prompt = default_prompt()
        async with _VLM_SEMAPHORE:
            text = await anyio.to_thread.run_sync(lambda: vlm.describe(data, prompt))
        markdown = f"![{filename}]({filename})\n\n{(text or '').strip() or '[No description]'}"
        log.info(f"done: image.describe in {time.perf_counter()-t0:.3f}s")

    return markdown, metadata


def _batched(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---- Qdrant client (per current quickstart examples) ----
try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import VectorParams, Distance, PointStruct  # docs show http.models in quickstart
    _HAS_QDRANT = True
except Exception:
    _HAS_QDRANT = False


def _qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))


def _ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    # Prefer collection_exists() per client docs
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _marker_point(job_key: str,
                  vector: List[float],
                  filename: str,
                  category: str,
                  collection: str,
                  status: str,
                  expected_chunks: Optional[int],
                  uploaded_chunks: int,
                  error: Optional[str] = None) -> PointStruct:
    payload = {
        "type": "job_marker",
        "job_key": job_key,
        "filename": filename,
        "category": category,
        "collection": collection,
        "status": status,  # "running" | "done" | "failed"
        "expected_chunks": expected_chunks,
        "uploaded_chunks": uploaded_chunks,
        "finished_at": None,
    }
    if status in ("done", "failed"):
        payload["finished_at"] = datetime.now().isoformat()
    if error:
        payload["error"] = str(error)
    return PointStruct(id=f"marker-{job_key}", vector=vector, payload=payload)


def _embed_texts_get_vectors(texts: List[str]) -> List[List[float]]:
    """Synchronous helper to embed a batch of texts."""
    return [get_genai_embedding(t) for t in texts]


async def _process_job_async(
    data: bytes,
    filename: str,
    ext: str,
    category: str,
    collection_name: str,
    job_key: str,
) -> None:
    """
    Background pipeline:
    parse (gated) -> chunk -> embed (batched) -> job_marker updates.
    Runs after 202 Accepted has been sent to the client.
    """
    log = _logger(job=job_key, file=filename, phase="background")
    client: Optional[QdrantClient] = None
    marker_vec: Optional[List[float]] = None
    uploaded = 0
    expected: Optional[int] = None
    t_overall = time.perf_counter()

    try:
        # -------- PARSE (gated) --------
        t0 = time.perf_counter()
        async with _PARSE_SEMAPHORE:
            if ext == ".pdf":
                with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    markdown, metadata = await _parse_document(tmp_path, filename, ext, job_key=job_key)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                markdown, metadata = await _parse_document("", filename, ext, image_bytes=data, job_key=job_key)
        log.with_phase("parse").info(f"completed in {time.perf_counter() - t0:.3f}s")

        # -------- CHUNK (offloaded) --------
        t1 = time.perf_counter()
        chunks: List[str] = await anyio.to_thread.run_sync(chunk_markdown, markdown)
        expected = len(chunks)
        log.with_phase("chunk").info(f"completed count={expected} in {time.perf_counter()-t1:.3f}s")

        if not _HAS_QDRANT or not chunks:
            log.with_phase("embed").warning("skipped (no Qdrant or no chunks)")
            return

        # Prepare Qdrant client and marker vector (ensure collection)
        client = _qdrant_client()
        marker_vec = get_genai_embedding("job_marker")
        _ensure_collection(client, collection_name, len(marker_vec))

        # Upsert initial marker as "running"
        client.upsert(
            collection_name=collection_name,
            points=[_marker_point(job_key, marker_vec, filename, category, collection_name, "running", expected, uploaded)],
            wait=True,
        )
        log.with_phase("embed").info(f"start: total_chunks={expected} batch_size={EMBED_BATCH_SIZE}")

        # -------- EMBED & UPSERT IN BATCHES --------
        chunk_index_offset = 0
        for batch in _batched(chunks, EMBED_BATCH_SIZE):
            tb = time.perf_counter()
            vectors = await anyio.to_thread.run_sync(_embed_texts_get_vectors, batch)

            # Build Qdrant points for this batch
            points: List[PointStruct] = []
            for i, (text, vec) in enumerate(zip(batch, vectors)):
                idx = chunk_index_offset + i
                payload = {"filename": filename, "category": category, "chunk_index": idx, "dim": len(vec)}
                points.append(PointStruct(id=uuid.uuid4().hex, vector=vec, payload=payload))

            client.upsert(collection_name=collection_name, points=points, wait=True)
            uploaded += len(points)

            # Update progress marker
            client.upsert(
                collection_name=collection_name,
                points=[_marker_point(job_key, marker_vec, filename, category, collection_name, "running", expected, uploaded)],
                wait=True,
            )
            log.with_phase("embed").info(f"batch uploaded={len(points)} total_uploaded={uploaded} took={time.perf_counter()-tb:.3f}s")
            chunk_index_offset += len(batch)

        # Mark as done
        client.upsert(
            collection_name=collection_name,
            points=[_marker_point(job_key, marker_vec, filename, category, collection_name, "done", expected, uploaded)],
            wait=True,
        )
        log.with_phase("embed").info(f"completed total_uploaded={uploaded} overall={time.perf_counter()-t_overall:.3f}s")

    except Exception as e:
        if client and marker_vec:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=[_marker_point(job_key, marker_vec, filename, category, collection_name, "failed", expected, uploaded, error=str(e))],
                    wait=True,
                )
            except Exception:
                pass
        _logger(job=job_key, file=filename, phase="error").exception(f"pipeline failed: {e}")


# ----------------------------- FastAPI app -----------------------------
app = FastAPI(title="Document Ingestion API")


class _ProcessResponse(BaseModel):
    # For 202 response
    status: str
    job_key: str
    filename: str
    category: str
    collection: str


@app.post("/process")
async def process_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    async_mode: Optional[str] = Form(default=None),
):
    """
    Mode A (default): Full pipeline and respond when completed (HTTP 200).
    Mode B (async=1): Respond immediately (HTTP 202) & run pipeline in background.
    """
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")

    # Read bytes NOW (UploadFile temp may be deleted after response)
    data = await file.read()

    is_async = str(async_mode).strip().lower() in {"1", "true", "yes"}

    if is_async:
        # ---------- Mode B: fire-and-poll ----------
        job_key = uuid.uuid4().hex
        _logger(job=job_key, file=filename, phase="accept").info("accepted async job")
        # FastAPI BackgroundTasks: runs after the response has been sent
        background_tasks.add_task(
            _process_job_async,
            data,
            filename,
            ext,
            category,
            collection_name,
            job_key,
        )
        return JSONResponse(
            status_code=202,
            content=_ProcessResponse(
                status="accepted",
                job_key=job_key,
                filename=filename,
                category=category,
                collection=collection_name,
            ).model_dump(),  # Pydantic v2
        )

    # ---------- Mode A: synchronous pipeline ----------
    job_key = uuid.uuid4().hex
    log = _logger(job=job_key, file=filename, phase="sync")
    t_overall = time.perf_counter()
    try:
        # PARSE (gated by semaphore)
        t0 = time.perf_counter()
        async with _PARSE_SEMAPHORE:
            if ext == ".pdf":
                with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    markdown, metadata = await _parse_document(tmp_path, filename, ext, job_key=job_key)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                markdown, metadata = await _parse_document("", filename, ext, image_bytes=data, job_key=job_key)
        log.with_phase("parse").info(f"completed in {time.perf_counter()-t0:.3f}s")

        # CHUNK (offload)
        t1 = time.perf_counter()
        chunks: List[str] = await anyio.to_thread.run_sync(chunk_markdown, markdown)
        log.with_phase("chunk").info(f"completed count={len(chunks)} in {time.perf_counter()-t1:.3f}s")

        # EMBED batched (offload embedding loop)
        uploaded = 0
        if _HAS_QDRANT and chunks:
            client = _qdrant_client()
            marker_vec = get_genai_embedding("marker")
            _ensure_collection(client, collection_name, len(marker_vec))

            chunk_index_offset = 0
            for batch in _batched(chunks, EMBED_BATCH_SIZE):
                tb = time.perf_counter()
                vectors = await anyio.to_thread.run_sync(_embed_texts_get_vectors, batch)
                points: List[PointStruct] = []
                for i, (text, vec) in enumerate(zip(batch, vectors)):
                    idx = chunk_index_offset + i
                    payload = {"filename": filename, "category": category, "chunk_index": idx, "dim": len(vec)}
                    points.append(PointStruct(id=uuid.uuid4().hex, vector=vec, payload=payload))
                client.upsert(collection_name=collection_name, points=points, wait=True)
                uploaded += len(points)
                chunk_index_offset += len(batch)
                log.with_phase("embed").info(f"batch uploaded={len(points)} total_uploaded={uploaded} took={time.perf_counter()-tb:.3f}s")

        log.with_phase("embed").info(f"completed total_uploaded={uploaded} overall={time.perf_counter()-t_overall:.3f}s")
        return JSONResponse({
            "file_name": filename,
            "file_type": metadata.get("file_type"),
            "page_count": metadata.get("page_count"),
            "images_count": metadata.get("images_count"),
            "uploaded_chunks": uploaded,
            "collection": collection_name,
            "total_processing_seconds": time.perf_counter() - t_overall,
            "category": category,
        })
    except Exception as e:
        log.with_phase("error").exception(f"pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
