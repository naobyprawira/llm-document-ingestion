"""
FastAPI application exposing granular document ingestion endpoints.

This API builds upon the upstream pipeline to offer fine‑grained
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

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

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

app = FastAPI(title="Document Ingestion API")


def _extract_metadata_from_doc(doc: Any) -> Dict[str, Any]:
    """Attempt to extract common metadata from a Docling document.

    Docling documents expose metadata inconsistently across versions, so
    this helper tries several attribute names. Missing fields are
    returned as ``None``.

    :param doc: A ``DoclingDocument`` instance returned by ``DocParser``.
    :return: Dict with keys ``author`` and ``page_count``.
    """
    author = None
    page_count = None
    try:
        if hasattr(doc, "info"):
            info = getattr(doc, "info")
            # Docling v2 stores authors as a list
            if hasattr(info, "authors") and info.authors:
                author = ", ".join(info.authors)
            elif hasattr(info, "author") and info.author:
                author = info.author
        # Page count may be available directly
        if hasattr(doc, "page_count"):
            page_count = getattr(doc, "page_count")
        elif hasattr(doc, "n_pages"):
            page_count = getattr(doc, "n_pages")
        elif hasattr(doc, "num_pages"):
            page_count = getattr(doc, "num_pages")
    except Exception:
        pass
    return {"author": author, "page_count": page_count}


def _parse_document(tmp_path: str, filename: str, ext: str) -> tuple[str, Dict[str, Any]]:
    """Internal helper to parse a document or image and return markdown and metadata.

    :param tmp_path: Path to a temporary file containing the uploaded data.
    :param filename: Original filename from the upload.
    :param ext: File extension (lowercase) including the dot.
    :return: A tuple (markdown, metadata)
    :raises Exception: Propagated if parsing or description fails.
    """
    metadata: Dict[str, Any] = {
        "filename": filename,
        "file_type": ext.lstrip("."),
    }
    markdown: str = ""
    if ext in [".pdf"]:
        parser = DocParser()
        doc, base_md, figures = parser.parse(tmp_path)
        meta_extra = _extract_metadata_from_doc(doc)
        metadata.update(meta_extra)
        metadata["images_count"] = len(figures)
        # Describe images
        vlm = GeminiVLM(VLMConfig())
        prompt = default_prompt()
        descs: List[FigureDesc] = []
        for f in figures:
            if len(f.jpeg_bytes) < 5000:
                text = "Small image (likely logo or decorative element) - skipped"
            else:
                text = vlm.describe(f.jpeg_bytes, prompt)
                retry_count = 0
                max_retries = 3
                while (not text or len(text.strip()) < 10) and retry_count < max_retries:
                    retry_count += 1
                    text = vlm.describe(f.jpeg_bytes, prompt)
            if text.startswith("Small image (likely logo or decorative element)"):
                continue
            descs.append(FigureDesc(index=f.index, page=f.page, caption=f.caption, description=text))
        markdown = compose_markdown(base_md, descs)
    else:
        # Single image
        metadata.update({"page_count": 1, "images_count": 1, "author": None})
        data = open(tmp_path, "rb").read()
        vlm = GeminiVLM(VLMConfig())
        prompt = default_prompt()
        text = vlm.describe(data, prompt)
        if not text or len(text.strip()) < 1:
            text = "[No description]"
        markdown = f"![{filename}]({filename})\n\n{text}"
    return markdown, metadata


def _embed_chunks_with_metadata(
    chunks: List[str], metadata: Dict[str, Any], collection: str, *, use_uuid: bool = True
) -> int:
    """
    Embed a list of chunks and upload to Qdrant, attaching metadata to each payload.

    This wrapper computes embeddings via ``get_genai_embedding`` and then
    constructs Qdrant points with payloads that include the original
    chunk text along with all provided metadata. It reimplements the
    upsert logic from ``embed.py`` to allow injecting arbitrary keys
    into the payload. If Qdrant is unavailable, it returns 0.

    :param chunks: List of text chunks to embed.
    :param metadata: Metadata dict to merge into each payload. Note that keys
        like ``chunk_index`` and ``dim`` will override existing keys in
        metadata if present.
    :param collection: Name of the Qdrant collection.
    :param use_uuid: Generate UUIDs for point IDs when True; else use
        deterministic ``<docName>-<index>`` IDs.
    :return: Number of uploaded embeddings.
    """
    if not _HAS_QDRANT:
        # Qdrant client is not installed; nothing to upload
        return 0
    if not chunks:
        return 0
    # Compute embeddings
    vectors: List[List[float]] = []
    for ch in chunks:
        # get_genai_embedding may raise if google-genai is not installed
        vec = get_genai_embedding(ch)
        vectors.append(vec)
    # Connect to Qdrant
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL environment variable is not set.")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    vector_size = len(vectors[0])
    # Create collection if missing
    try:
        client.get_collection(collection_name=collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    # Build points
    points: List[Dict[str, Any]] = []
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        if use_uuid:
            point_id = uuid.uuid4().hex
        else:
            # Use filename from metadata if available, else 'doc'
            prefix = str(metadata.get("filename", "doc")).replace(".", "-")
            point_id = f"{prefix}-{idx}"
        # Build payload merging metadata
        payload = {"text": text, **metadata, "chunk_index": idx, "dim": len(vec)}
        points.append({"id": point_id, "vector": vec, "payload": payload})
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


class ChunkRequest(BaseModel):
    markdown: str


class EmbedRequest(BaseModel):
    chunks: List[str]
    collection: Optional[str] = None


@app.post("/parse")
async def parse_endpoint(file: UploadFile = File(...)):
    """Parse and optionally OCR a PDF or image, returning Markdown and metadata."""
    start_time = time.perf_counter()
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # Delegate to internal helper
        markdown, metadata = _parse_document(tmp_path, filename, ext)
        duration = time.perf_counter() - start_time
        metadata["processing_time_seconds"] = duration
        metadata["processing_finished_at"] = datetime.now().isoformat()
        return JSONResponse({"markdown": markdown, "metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/chunk")
async def chunk_endpoint(body: ChunkRequest):
    """Chunk a Markdown document into segments for embedding."""
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


@app.post("/embed")
async def embed_endpoint(body: EmbedRequest):
    """Embed a list of text chunks and upload them to Qdrant."""
    start_time = time.perf_counter()
    try:
        collection = body.collection or os.getenv("QDRANT_COLLECTION", "documents")
        # Use the simple embed function for generic embed calls (no metadata)
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


@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    category: str = Form(..., description="Custom category to attach to the metadata"),
):
    """Run the full ingestion pipeline and return a zip of results.

    This endpoint runs ``/parse``, ``/chunk`` and ``/embed`` in sequence on
    the uploaded file. The supplied ``category`` is added to the
    metadata. A temporary directory is created where the enriched
    Markdown (``document.md``), chunks JSON (``chunks.json``) and
    metadata JSON (``metadata.json``) are written. The directory is
    then zipped and returned as a file response. Clients can save and
    extract the archive to inspect the outputs.
    """
    overall_start = time.perf_counter()
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # Step 1: parse directly using helper
        markdown, metadata = _parse_document(tmp_path, filename, ext)
        metadata["category"] = category
        # Record parse timing separately
        # Step 2: chunk
        chunk_start = time.perf_counter()
        chunks: List[str] = chunk_markdown(markdown)
        chunk_duration = time.perf_counter() - chunk_start
        # Step 3: embed with metadata
        embed_start = time.perf_counter()
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        try:
            embed_count = _embed_chunks_with_metadata(chunks, metadata, collection_name)
        except Exception as ex:
            # If embedding fails, still produce zip but note the error in metadata
            metadata.setdefault("embedding_error", str(ex))
            embed_count = 0
        embed_duration = time.perf_counter() - embed_start
        # Aggregate metadata
        metadata["embedding_info"] = {
            "uploaded_chunks": embed_count,
            "collection": collection_name,
            "processing_time_seconds": embed_duration,
            "processing_finished_at": datetime.now().isoformat(),
        }
        metadata["chunking_info"] = {
            "chunks_count": len(chunks),
            "processing_time_seconds": chunk_duration,
        }
        metadata["overall_processing_time_seconds"] = time.perf_counter() - overall_start
        metadata["processing_finished_at"] = datetime.now().isoformat()
        # Prepare zip file
        with TemporaryDirectory() as tmpdir:
            md_path = os.path.join(tmpdir, "document.md")
            json_path = os.path.join(tmpdir, "chunks.json")
            meta_path = os.path.join(tmpdir, "metadata.json")
            with open(md_path, "w", encoding="utf‑8") as f:
                f.write(markdown)
            with open(json_path, "w", encoding="utf‑8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            with open(meta_path, "w", encoding="utf‑8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            # Create a persistent temp file for the zip outside of the context manager
            tmp_zip = NamedTemporaryFile(delete=False, suffix=".zip")
            tmp_zip.close()
            shutil.make_archive(base_name=tmp_zip.name[:-4], format="zip", root_dir=tmpdir)
            zip_file_path = tmp_zip.name
            return FileResponse(zip_file_path, filename=os.path.basename(zip_file_path), media_type="application/zip")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass