"""
Document ingestion pipeline.

This module orchestrates the ingestion of PDF documents into a
vector database.  It performs the following operations:

1. Opens the PDF and iterates over pages.
2. Detects flowcharts; if detected (or forced), renders the page to an image
   and calls a VLM to obtain a linearised description of the flowchart.
3. Extracts text for each page and chunks it using heading-aware + paragraph-safe
   rules. Falls back to OCR when no text can be extracted.
4. Embeds all extracted snippets with Gemini embeddings.
5. Upserts vectors + rich payloads into Qdrant with deterministic point IDs.

The main entry point is :func:`ingest_document`. See ``cli/ingest_pdf.py``.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from ..config import AppConfig, load_config
from ..embeddings.google import EmbeddingModel
from ..chunking.chunking_structure import chunk_document
from ..vision.flowchart_extractor import extract_flowchart_steps
from ..parsers.pdf_text import open_pdf, extract_page_text
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
    max_pages: int = 0,           # 0 = process all from start_page
    dry_run: bool = False,
    force_vlm: bool = False,
    config: Optional[AppConfig] = None,
) -> Dict[str, int]:
    """
    Ingest a PDF into Qdrant.

    Returns:
      {"n_flowcharts": int, "n_text_chunks": int}
    """
    conf = config or load_config()
    lang = language or conf.default_language

    # Open PDF
    doc = open_pdf(filepath)
    total_pages = doc.page_count

    # Page window [start_idx, end_idx)
    start_idx = max(0, start_page - 1)
    end_idx = total_pages if max_pages == 0 else min(total_pages, start_idx + max_pages)

    # Prepare embedding + Qdrant
    embedder = EmbeddingModel(conf)
    client = None
    if not dry_run:
        # Determine embedding vector dimension once
        probe_vec = embedder.embed_text("dim probe")
        vector_dim = len(probe_vec)

        client = get_client(conf)
        ensure_collection(client, conf, vector_dim)

    n_flowcharts = 0
    n_text_chunks = 0

    for page_index in range(start_idx, end_idx):
        t0 = time.perf_counter()

        # 1) Flowchart extraction (detector is inside extractor unless forced)
        fc = extract_flowchart_steps(doc, page_index, conf, force=force_vlm)
        if fc and fc.get("has_flowchart"):
            steps_text = (fc.get("steps_text") or "").strip()
            if steps_text:
                n_flowcharts += 1
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
                    "chunk_id": "flowchart",
                    "heading": (fc.get("title") or "").strip(),
                    "content": steps_text,
                    "source_file": os.path.basename(filepath),
                }
                if not dry_run:
                    vec = embedder.embed_text(steps_text)
                    pid = point_id(doc_id, payload["page"], payload["chunk_id"])
                    upsert_points(client, conf, [payload], [vec], point_ids=[pid])

        # 2) Text extraction
        text = extract_page_text(doc, page_index) or ""
        if not text.strip():
            # OCR fallback only if we didn't already get flow steps
            if not (fc and fc.get("has_flowchart")):
                text = ocr_page(doc, page_index) or ""

        # 3) Chunking + embeddings
        if text.strip():
            # One policy for all: heading split + paragraph-safe splitting
            chunks = chunk_document(text, max_words=conf.max_tokens_per_chunk, min_words=200)

            if not dry_run:
                payloads: List[dict] = []
                vectors: List[List[float]] = []
                ids: List[int] = []

                for i, (heading, content) in enumerate(chunks):
                    n_text_chunks += 1
                    chunk_id = f"{i:04d}"
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
                        "content": content,
                        "source_file": os.path.basename(filepath),
                    }
                    payloads.append(payload)
                    vectors.append(embedder.embed_text(content))
                    ids.append(point_id(doc_id, payload["page"], chunk_id))

                if payloads:
                    upsert_points(client, conf, payloads, vectors, point_ids=ids)
            else:
                # Dry-run path just counts without writing
                n_text_chunks += len(chunks)

        t1 = time.perf_counter()
        logger.debug(
            "Page %d processed in %.2fs (flowchart=%s)",
            page_index + 1,
            t1 - t0,
            bool(fc and fc.get("has_flowchart")),
        )

    if dry_run:
        logger.info("Dry run enabled; no points were upserted.")

    return {"n_flowcharts": n_flowcharts, "n_text_chunks": n_text_chunks}
