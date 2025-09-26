"""Package root for the document ingestion API.

This package exposes a synchronous document ingestion pipeline and a FastAPI
application under :mod:`llm_document_ingestion.src.api`.  See the
``README.md`` at the repository root for usage details.
"""

# Re-export the FastAPI app for convenience
from .api import app  # noqa: F401