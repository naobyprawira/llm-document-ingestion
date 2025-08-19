"""
Document ingestion pipeline.

This module orchestrates the ingestion of PDF documents into a
vector database.  It performs the following operations:

1. Opens the PDF and iterates over pages.
2. Uses heavy vector analysis to detect whether a page contains a
   flowchart.  If so, renders the page to an image and calls a
   vision-language model (VLM) to obtain a linearised description of
   the flowchart.
3. For pages without flowcharts, extracts the text and chunks it
   using structured heading-based parsing.  If no headings are present,
   falls back to a smart chunking algorithm.  Optionally, if no text
   can be extracted at all the page is passed through OCR.
4. Embeds the extracted text snippets using the configured embedding
   model.
5. Upserts the resulting vectors and payloads into Qdrant.  Payloads
   include rich metadata to facilitate downstream retrieval and
   filtering.

Usage
-----
The main entry point is the :func:`ingest_document` function.  See
``cli/ingest_pdf.py`` for a command line wrapper.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # type: ignore
from qdrant_client.http import models as qm

from ..config import AppConfig, load_config
from ..embeddings.google import EmbeddingModel
from ..chunking.chunking_structure import chunk_by_heading, smart_chunk
from ..vision.flowchart_extractor import detect_flowchart, extract_flowchart_steps
from ..parsers.pdf_text import open_pdf, extract_page_text, render_page_to_png
from ..parsers.ocr import ocr_page
from ..storage.qdrant_store import (
    get_client,
    ensure_collection,
    upsert_points,
)
from ..utils.hash_id import point_id

logger = logging.getLogger(__name__)


def ingest_document(
    filepath: str,
    doc_id: str,
    scope: str,
    company: str,
    country: str,
    language: Optional[str] = None,
    version: str = "v1",
    effective_from: str = "1970-01-01",
    effective_to: Optional[str] = None,
    security_class: str = "internal",
    *,
    start_page: int = 1,
    max_pages: Optional[int] = None,
    dry_run: bool = False,
    config: Optional[AppConfig] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Ingest a PDF document into Qdrant.

    Parameters
    ----------
    filepath: str
        Path to the PDF file on disk.
    doc_id: str
        Unique identifier for the document.  Used as a filter for deletion and retrieval.
    scope: str
        Top-level scope of the document (e.g. GovReg, AdidasCompliance, Internal).
    company: str
        Company identifier for internal documents (e.g. PRB, PBB, GROUP).  Ignored for other scopes.
    country: str
        Country code (e.g. ID or GLOBAL).
    language: Optional[str]
        Language code of the document content.  Defaults to configuration default.
    version: str
        Version identifier of the document.
    effective_from: str
        Effective date (ISO 8601) from which the document is valid.
    effective_to: Optional[str]
        Effective date (ISO 8601) until which the document is valid. ``None`` means indefinite.
    security_class: str
        Security classification (e.g. internal, confidential).  Stored as metadata.
    config: Optional[AppConfig]
        Application configuration.  If omitted, loaded via :func:`load_config`.

    Returns
    -------
    Dict[str, List[Dict[str, object]]]
        A summary of ingested content.  Keys "flowcharts" and "text_chunks"
        map to lists of dictionaries describing the items ingested.
    """
    conf = config or load_config()
    lang = language or conf.default_language
    # Prepare embedding model and Qdrant client
    embedder = EmbeddingModel(conf)
    # Determine embedding dimension by embedding a dummy string
    try:
        dummy_vector = embedder.embed_text("dummy")
    except Exception as e:  # pragma: no cover - fallback improbable
        logger.error("Failed to obtain embedding dimension: %s", e)
        return {"flowcharts": [], "text_chunks": []}
    vector_dim = len(dummy_vector)
    client = get_client(conf)
    ensure_collection(client, conf, vector_dim)

    doc_path = Path(filepath)
    if not doc_path.is_file():
        logger.error("File not found: %s", filepath)
        raise FileNotFoundError(filepath)

    doc = open_pdf(str(doc_path))
    flowcharts_info: List[Dict[str, object]] = []
    text_chunks_info: List[Dict[str, object]] = []
    points: List[qm.PointStruct] = []

    # Limit processing based on start_page and max_pages
    for page_index in range(doc.page_count):
        current_page = page_index + 1
        if current_page < start_page:
            continue
        if max_pages is not None and (current_page - start_page + 1) > max_pages:
            break
        page = doc.load_page(page_index)
        # record start time for observability
        page_start = time.perf_counter()
        # Detect flowchart presence using heavy detection
        has_chart = False
        detect_start = time.perf_counter()
        try:
            has_chart = detect_flowchart(page)
        except Exception as e:
            logger.error("Error during flowchart detection on page %d: %s", page_index + 1, e)
        detect_end = time.perf_counter()

        if has_chart:
            # Extract steps via VLM
            steps_data = extract_flowchart_steps(doc, page_index, conf)
            if steps_data:
                steps_text = steps_data["steps_text"]
                title = steps_data.get("title", "") or f"Page {page_index + 1} flowchart"
                confidence = steps_data.get("confidence", 0.0)
                # Embed
                vector = embedder.embed_text(steps_text)
                embed_end = time.perf_counter()
                pid = point_id(doc_id, page_index + 1, "flowchart")
                payload = {
                    "type": "flow_steps",
                    "doc_id": doc_id,
                    "scope": scope,
                    "company": company,
                    "country": country,
                    "language": lang,
                    "version": version,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "security_class": security_class,
                    "page": page_index + 1,
                    "title": title,
                    "confidence": confidence,
                    "content": steps_text,
                    # chunk_id omitted for flowchart
                }
                point = qm.PointStruct(id=pid, vector=vector, payload=payload)
                points.append(point)
                flowcharts_info.append(
                    {
                        "page": page_index + 1,
                        "title": title,
                        "confidence": confidence,
                    }
                )
                continue  # Skip further processing for this page

        # Otherwise treat as text; embedding and chunking times measured below

        # If no flowchart or extraction fails, treat as regular text
        try:
            page_text = extract_page_text(doc, page_index)
        except Exception as e:
            logger.error("Error extracting text from page %d: %s", page_index + 1, e)
            page_text = ""
        page_text = page_text.strip()
        # Fallback to OCR if no text extracted
        if not page_text:
            # Render page and run OCR
            try:
                image_bytes = render_page_to_png(doc, page_index, conf.image_dpi)
                ocr_result = ocr_page(image_bytes)
                page_text = (ocr_result or "").strip()
            except Exception as e:
                logger.error("Error performing OCR on page %d: %s", page_index + 1, e)
                page_text = ""
        if not page_text:
            continue
        # Structured chunking based on numbered headings
        chunks = chunk_by_heading(page_text)
        chunking_end = time.perf_counter()
        # If only one chunk with empty heading and its body equals the page text,
        # fall back to smart chunking
        if len(chunks) == 1 and (not chunks[0][0]):
            smart_chunks = smart_chunk(page_text, max_words=conf.max_tokens_per_chunk)
            for idx, chunk_text in enumerate(smart_chunks):
                embed_start = time.perf_counter()
                vector = embedder.embed_text(chunk_text)
                embed_end = time.perf_counter()
                chunk_id = f"{page_index + 1}:{idx + 1}"
                pid = point_id(doc_id, page_index + 1, chunk_id)
                payload = {
                    "type": "text_chunk",
                    "doc_id": doc_id,
                    "scope": scope,
                    "company": company,
                    "country": country,
                    "language": lang,
                    "version": version,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "security_class": security_class,
                    "page": page_index + 1,
                    "chunk_id": chunk_id,
                    "heading": "",
                    "content": chunk_text,
                }
                point = qm.PointStruct(id=pid, vector=vector, payload=payload)
                points.append(point)
                text_chunks_info.append(
                    {
                        "page": page_index + 1,
                        "chunk_id": chunk_id,
                        "heading": "",
                        "length": len(chunk_text.split()),
                    }
                )
        else:
            for idx, (heading, body) in enumerate(chunks):
                if not body:
                    continue
                embed_start = time.perf_counter()
                vector = embedder.embed_text(body)
                embed_end = time.perf_counter()
                chunk_id = f"{page_index + 1}:{idx + 1}"
                pid = point_id(doc_id, page_index + 1, chunk_id)
                payload = {
                    "type": "text_chunk",
                    "doc_id": doc_id,
                    "scope": scope,
                    "company": company,
                    "country": country,
                    "language": lang,
                    "version": version,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "security_class": security_class,
                    "page": page_index + 1,
                    "chunk_id": chunk_id,
                    "heading": heading,
                    "content": body,
                }
                point = qm.PointStruct(id=pid, vector=vector, payload=payload)
                points.append(point)
                text_chunks_info.append(
                    {
                        "page": page_index + 1,
                        "chunk_id": chunk_id,
                        "heading": heading,
                        "length": len(body.split()),
                    }
                )

        # End-of-page observability logging
        page_end = time.perf_counter()
        logger.debug(
            "Page %d processed: detect %.3fs, chunk+embed %.3fs, total %.3fs", 
            page_index + 1,
            detect_end - detect_start,
            page_end - chunking_end,
            page_end - page_start,
        )

    # Batch upsert all points unless dry_run
    if not dry_run:
        if points:
            upsert_points(client, conf, points)
        else:
            logger.info("No content ingested from document %s", doc_id)
    else:
        logger.info("Dry run enabled; no points were upserted")
    return {"flowcharts": flowcharts_info, "text_chunks": text_chunks_info}
