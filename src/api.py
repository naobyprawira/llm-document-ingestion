# src/api.py

"""
FastAPI application exposing granular document ingestion endpoints.

This API builds upon the upstream pipeline to offer fine-grained
operations for clients:

* ``POST /parse``: Parse an uploaded PDF or image. If the file is a
  PDF, Docling extracts text and images; images are described via
  ``GeminiVLM`` using the default prompt. The response includes the
  enriched Markdown and metadata such as author, filename, file type,
  page count and image count. For images, the VLM generates a single
  description and returns a simple Markdown snippet.

* ``POST /chunk``: Given a Markdown string, split it into chunks
  suitable for embedding using the same logic as the upstream
  ``chunk_markdown`` function. The response returns the list of
  chunks along with metadata.

* ``POST /embed``: Accepts a JSON list of chunks and uploads them to
  Qdrant using ``embed_and_upload_json``. Clients may override the
  target collection. The response returns the number of embedded
  chunks and timing information.

* ``POST /process``: Accepts an uploaded PDF or image and a category
  string. Runs ``/parse``, ``/chunk`` and ``/embed`` in sequence. The
  resulting Markdown, chunks and metadata are written to a temporary
  folder, zipped and returned as a file response. The metadata
  includes all fields from ``/parse`` plus the category and
  processing times for each step.

If any external dependency (Google Gen AI, Qdrant, Docling) raises
errors, the API returns an HTTP 500 with the exception message. This
behaviour matches the upstream design and avoids silently ignoring
failures.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Dict, Any, Optional

import anyio
import asyncio
from dotenv import load_dotenv

load_dotenv()  # harus sebelum import yang membaca env di import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks  # type: ignore
from fastapi.responses import JSONResponse, FileResponse  # type: ignore
from pydantic import BaseModel  # type: ignore

from .doc_parser import DocParser, Figure
from .vlm import GeminiVLM, VLMConfig
from .prompts import default_prompt
from .markdown import FigureDesc, compose_markdown
from .chunk import chunk_markdown
from .embed import embed_and_upload_json, get_genai_embedding

try:
    # qdrant_client may not always be installed at runtime
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models   # type: ignore
    _HAS_QDRANT = True
except Exception:
    _HAS_QDRANT = False

import os as _os

# In environments where python-multipart or other optional deps are missing,
# it may be desirable to import this module without registering FastAPI routes.
# Set the NO_FASTAPI environment variable to skip route registration. This
# allows internal helpers to be imported for testing without requiring
# optional dependencies.
_SKIP_FASTAPI = bool(_os.getenv("NO_FASTAPI"))

# Initialise the FastAPI application unless explicitly skipped
app = FastAPI(title="Document Ingestion API") if not _SKIP_FASTAPI else None  # type: ignore

# ============================================================================
# HARD-CODED CONCURRENCY LIMITS
# Limit how many documents can be in the *parsing stage* at the same time.
# This specifically gates the parsing work inside the /process pipeline.
# Adjust this constant to tune concurrency.
MAX_CONCURRENT_PARSE = 3
_PARSE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_PARSE)
# ============================================================================


def _ensure_primitive(value: Any) -> Any:
    """Return a JSON-serialisable primitive for the given value."""
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
    """Attempt to extract common metadata from a Docling document."""
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
    image_bytes: bytes | None = None
) -> tuple[str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {"filename": filename, "file_type": ext.lstrip(".")}
    markdown: str = ""

    if ext == ".pdf":
        parser = DocParser()
        doc, base_md, figures = await anyio.to_thread.run_sync(parser.parse, tmp_path)
        meta_extra = _extract_metadata_from_doc(doc)
        metadata.update(meta_extra)

        valid_figures = [f for f in figures if len(f.jpeg_bytes) >= 5000]
        metadata["images_count"] = len(valid_figures)

        vlm = GeminiVLM(VLMConfig())
        prompt = default_prompt()

        async def describe_one(f: Figure) -> FigureDesc:
            def _do():
                text = vlm.describe(f.jpeg_bytes, prompt)
                retry = 0
                while (not text or len(text.strip()) < 10) and retry < 3:
                    retry += 1
                    text = vlm.describe(f.jpeg_bytes, prompt)
                return text
            text = await anyio.to_thread.run_sync(_do)
            return FigureDesc(index=f.index, page=f.page, caption=f.caption, description=text)

        descs: List[FigureDesc] = []
        if valid_figures:
            descs = await asyncio.gather(*(describe_one(f) for f in valid_figures))

        markdown = compose_markdown(base_md, descs)

    else:
        metadata.update({"page_count": 1, "images_count": 1, "author": None})
        data = image_bytes if image_bytes is not None else open(tmp_path, "rb").read()
        vlm = GeminiVLM(VLMConfig())
        prompt = default_prompt()
        text = await anyio.to_thread.run_sync(lambda: vlm.describe(data, prompt))
        if not text or len(text.strip()) < 1:
            text = "[No description]"
        markdown = f"![{filename}]({filename})\n\n{text}"

    return markdown, metadata


def _embed_chunks_with_metadata(
    chunks: List[str], metadata: Dict[str, Any], collection: str, *, use_uuid: bool = True
) -> int:
    if not _HAS_QDRANT:
        return 0
    if not chunks:
        return 0
    vectors: List[List[float]] = []
    for ch in chunks:
        vec = get_genai_embedding(ch)
        vectors.append(vec)
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    vector_size = len(vectors[0])
    try:
        client.get_collection(collection_name=collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    points: List[Dict[str, Any]] = []
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        point_id = uuid.uuid4().hex if use_uuid else f"{str(metadata.get('filename','doc')).replace('.','-')}-{idx}"
        payload_base = {k: _ensure_primitive(v) for k, v in metadata.items()}
        payload = {"text": text, **payload_base, "chunk_index": idx, "dim": len(vec)}
        points.append({"id": point_id, "vector": vec, "payload": payload})
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


class ChunkRequest(BaseModel):
    markdown: str


class EmbedRequest(BaseModel):
    chunks: List[str]
    collection: Optional[str] = None


async def parse_endpoint(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()

    data = await file.read()
    tmp_path = None
    try:
        if ext == ".pdf":
            with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            markdown, metadata = await _parse_document(tmp_path, filename, ext)
        else:
            markdown, metadata = await _parse_document("", filename, ext, image_bytes=data)

        duration = time.perf_counter() - start_time
        metadata["processing_time_seconds"] = duration
        metadata["processing_finished_at"] = datetime.now().isoformat()
        metadata = {k: _ensure_primitive(v) for k, v in metadata.items()}
        return JSONResponse({"markdown": markdown, "metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


async def chunk_endpoint(body: ChunkRequest):
    start_time = time.perf_counter()
    try:
        chunks: List[str] = chunk_markdown(body.markdown)
        duration = time.perf_counter() - start_time
        metadata = {
            "chunks_count": len(chunks),
            "processing_time_seconds": duration,
            "processing_finished_at": datetime.now().isoformat(),
        }
        return JSONResponse({"chunks": chunks, "metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def embed_endpoint(body: EmbedRequest):
    start_time = time.perf_counter()
    try:
        collection = body.collection or os.getenv("QDRANT_COLLECTION", "documents")
        count = embed_and_upload_json(body.chunks, collection=collection)
        duration = time.perf_counter() - start_time
        metadata = {
            "uploaded_chunks": count,
            "collection": collection,
            "processing_time_seconds": duration,
            "processing_finished_at": datetime.now().isoformat(),
        }
        return JSONResponse({"metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_endpoint(
    file: UploadFile = File(...),
    category: str = Form(...),
):
    overall_start = time.perf_counter()
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()

    data = await file.read()
    tmp_path = None
    try:
        # ----------------- PARSE (gated by semaphore) -----------------
        async with _PARSE_SEMAPHORE:
            if ext == ".pdf":
                with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                markdown, metadata = await _parse_document(tmp_path, filename, ext)
            else:
                markdown, metadata = await _parse_document("", filename, ext, image_bytes=data)
        # --------------------------------------------------------------

        metadata["category"] = category

        # CHUNK (blocking → run in thread)
        chunks: List[str] = await anyio.to_thread.run_sync(chunk_markdown, markdown)

        # EMBED (blocking + network → run in thread)
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        embed_count = await anyio.to_thread.run_sync(
            _embed_chunks_with_metadata, chunks, metadata, collection_name
        )

        total_seconds = time.perf_counter() - overall_start
        return JSONResponse({
            "file_name": filename,
            "file_type": metadata.get("file_type"),
            "page_count": metadata.get("page_count"),
            "images_count": metadata.get("images_count"),
            "uploaded_chunks": embed_count,
            "collection": collection_name,
            "total_processing_seconds": total_seconds,
            "category": category,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if app is not None:
    app.post("/parse")(parse_endpoint)    # type: ignore[operator]
    app.post("/chunk")(chunk_endpoint)    # type: ignore[operator]
    app.post("/embed")(embed_endpoint)    # type: ignore[operator]
    app.post("/process")(process_endpoint)  # type: ignore[operator]
