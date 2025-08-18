"""
PDF utilities.

This module contains helper functions for working with PDF documents.
It uses PyMuPDF (``fitz``) for PDF parsing and rendering. Functions
include extracting text and converting pages to images for downstream
processing by vision-language models.
"""
from __future__ import annotations

import io
from typing import List, Tuple

import fitz  # type: ignore


def open_pdf(filepath: str) -> fitz.Document:
    """Open a PDF file and return the document object."""
    return fitz.open(filepath)


def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    """
    Extract text from the given page of a PDF document.

    Parameters
    ----------
    doc: fitz.Document
        The PDF document.
    page_index: int
        Zero-based page index.

    Returns
    -------
    str
        The text content of the page.
    """
    page = doc.load_page(page_index)
    return page.get_text("text")


def render_page_to_png(doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    """
    Render a single PDF page to a PNG image and return the image bytes.

    The page is rendered at the requested DPI. A higher DPI produces
    smoother images at the cost of more processing time and larger
    memory usage. Values between 300–350 are typically sufficient for
    diagram recognition.

    Parameters
    ----------
    doc: fitz.Document
        The PDF document.
    page_index: int
        Zero-based page index.
    dpi: int
        Dots per inch resolution for rendering.

    Returns
    -------
    bytes
        The PNG image data.
    """
    page = doc.load_page(page_index)
    # Convert DPI into a zoom factor. PDF default is 72 DPI.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    return png_bytes
