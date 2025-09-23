"""Parsing and figure description utilities.

This module wraps the heavy docling parsing functionality in a synchronous API
and adds concurrency around calls to the Gemini visual language model
(VLM).  It is responsible for turning raw documents (PDFs or images)
into enriched markdown and extracting useful metadata.

The primary entry point is :func:`parse_document`, which handles both
PDF and image inputs.  Internally it calls either :func:`_parse_pdf`
or :func:`_parse_image`.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from .doc_parser import DocParser, Figure  # parses PDFs and extracts figures
from .markdown import FigureDesc, compose_markdown
from .vlm import GeminiVLM, VLMConfig
from .prompts import default_prompt
from .logger import get_logger, MAX_CONCURRENT_VLM


def _ensure_primitive(value: Any) -> Any:
    """Coerce arbitrary objects into JSON‑serialisable primitives.

    If the value is callable it will be invoked (best effort).  Containers are
    traversed recursively.  Unserialisable objects are converted to strings.
    """
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
        import json  # local import to avoid circular deps
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def extract_metadata_from_doc(doc: Any) -> Dict[str, Any]:
    """Extract common metadata from a Docling document if present.

    Attempts to read author(s) and number of pages.  When information is
    unavailable it returns ``None`` for those fields.
    """
    author: Optional[str] = None
    page_count: Optional[int] = None
    try:
        if hasattr(doc, "info"):
            info = getattr(doc, "info")
            # First try a plural authors field
            if hasattr(info, "authors"):
                try:
                    authors_val = getattr(info, "authors")
                    authors = _ensure_primitive(authors_val)
                    if authors:
                        if isinstance(authors, (list, tuple, set)):
                            author = ", ".join(str(a) for a in authors)
                        else:
                            author = str(authors)
                except Exception:
                    pass
            # Fallback to singular author field
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


def clean_markdown(md: str) -> str:
    """Normalise markdown by collapsing excessive blank lines and stripping whitespace.

    This helper removes carriage returns, collapses three or more consecutive
    newlines into two and strips trailing spaces at the end of lines.  It
    improves the quality of the downstream chunks and embeddings.
    """
    # Normalise line endings
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse more than two blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Strip trailing spaces
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md.strip()


def _describe_one(vlm: GeminiVLM, f: Figure, prompt: str, job_key: str, filename: str) -> FigureDesc:
    """Describe a single figure with retries.

    Small images (less than ~5 kB) are assumed to be logos or decorative
    elements and skipped.  Descriptions shorter than ten characters are
    retried up to three times.  Any exceptions are caught and logged.
    """
    log = get_logger(job=job_key, file=filename, phase="vlm")
    if len(f.jpeg_bytes) < 5000:
        log.info(f"Skipping small figure {f.index} (page {f.page})")
        return FigureDesc(index=f.index, page=f.page, caption=f.caption, description="")
    text = ""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            text = vlm.describe(f.jpeg_bytes, prompt)
            if text and len(text.strip()) >= 10:
                break
        except Exception as exc:
            log.warning(f"VLM describe failed for figure {f.index}: {exc}")
        time.sleep(0.5)
    if not text or len(text.strip()) < 10:
        log.warning(
            f"Failed to get a satisfactory description for figure {f.index} after {max_retries} attempts"
        )
        description = ""
    else:
        description = text.strip()
    return FigureDesc(index=f.index, page=f.page, caption=f.caption, description=description)


def _describe_figures(figures: List[Figure], prompt: str, job_key: str, filename: str) -> List[FigureDesc]:
    """Describe all figures concurrently using a thread pool.

    Respects the ``MAX_CONCURRENT_VLM`` limit.  Returns only the
    descriptions that are non-empty.  Entries with empty descriptions
    indicate that the figure was skipped.
    """
    if not figures:
        return []
    vlm = GeminiVLM(VLMConfig())
    descs: List[FigureDesc] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        future_map = {
            executor.submit(_describe_one, vlm, f, prompt, job_key, filename): f for f in figures
        }
        for fut in as_completed(future_map):
            try:
                desc = fut.result()
                if desc.description:
                    descs.append(desc)
            except Exception as exc:
                # Log unexpected errors and skip
                get_logger(job=job_key, file=filename, phase="vlm").warning(
                    f"Unexpected error describing figure: {exc}"
                )
    # Sort descriptions by index so they appear in document order
    descs.sort(key=lambda d: d.index)
    return descs


def _parse_pdf(tmp_path: str, filename: str, job_key: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a PDF at ``tmp_path`` and return enriched markdown and metadata.

    The PDF is converted to markdown and figures using Docling.  Figures are
    described concurrently via Gemini.  The returned markdown includes a
    “Deskripsi Gambar” section appended to the end.
    """
    log = get_logger(job=job_key, file=filename, phase="parse")
    parser = DocParser()
    doc, base_md, figures = parser.parse(tmp_path)
    metadata: Dict[str, Any] = {"filename": filename, "file_type": "pdf"}
    meta_extra = extract_metadata_from_doc(doc)
    metadata.update(meta_extra)
    # Filter out tiny figures before description
    valid_figs = [f for f in figures if len(f.jpeg_bytes) >= 5000]
    metadata["images_count"] = len(valid_figs)
    prompt = default_prompt()
    descs = _describe_figures(valid_figs, prompt, job_key, filename)
    markdown = compose_markdown(base_md, descs)
    markdown = clean_markdown(markdown)
    log.info(
        f"Parsed PDF: pages={metadata.get('page_count')} images={metadata.get('images_count')}"
    )
    return markdown, metadata


def _parse_image(data: bytes, filename: str, ext: str, job_key: str) -> Tuple[str, Dict[str, Any]]:
    """Describe a single image and return markdown and metadata.

    Image inputs bypass Docling.  A ``GeminiVLM`` is invoked directly to
    produce a description, which is inserted into a simple markdown stub.
    """
    log = get_logger(job=job_key, file=filename, phase="parse")
    metadata = {"filename": filename, "file_type": ext.lstrip("."), "page_count": 1, "images_count": 1, "author": None}
    vlm = GeminiVLM(VLMConfig())
    prompt = default_prompt()
    description = ""
    try:
        description = vlm.describe(data, prompt)
    except Exception as exc:
        log.warning(f"VLM describe failed for image: {exc}")
    if not description or len(description.strip()) < 10:
        description = "[No description]"
    markdown = f"![{filename}]({filename})\n\n{description.strip()}"
    return clean_markdown(markdown), metadata


def parse_document(
    tmp_path: str,
    filename: str,
    ext: str,
    *,
    image_bytes: Optional[bytes] = None,
    job_key: str,
) -> Tuple[str, Dict[str, Any]]:
    """Parse a document (PDF or image) and return markdown and metadata.

    :param tmp_path: Path on disk for PDFs.  Ignored for image inputs.
    :param filename: Original file name supplied by the client.
    :param ext: File extension (lowercase, including dot).
    :param image_bytes: Raw bytes of the image when ``ext`` is not `.pdf`.
    :param job_key: Unique identifier for the job, used for logging.
    :return: A tuple ``(markdown, metadata)``.
    """
    if ext == ".pdf":
        return _parse_pdf(tmp_path, filename, job_key)
    else:
        if image_bytes is None:
            # In synchronous code we assume the caller has already read the file
            data = open(tmp_path, "rb").read()
        else:
            data = image_bytes
        return _parse_image(data, filename, ext, job_key)


__all__ = [
    "parse_document",
    "extract_metadata_from_doc",
    "clean_markdown",
]