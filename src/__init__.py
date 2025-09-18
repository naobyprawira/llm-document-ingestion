"""
Package for document ingestion.

This package provides a simple pipeline to convert PDFs to enriched
Markdown by extracting text with Docling, generating descriptions
for images via a vision language model (VLM), chunking the result
and persisting embeddings to a Qdrant vector store. A FastAPI
application is exposed in ``api.py`` for web usage.
"""

__all__ = [
    "main",
    "pipeline",
    "doc_parser",
    "util_img",
    "vlm",
    "prompts",
    "markdown",
    "chunk",
    "embed",
    "api",
]