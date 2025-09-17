"""
Exposes key classes and functions from the document ingestion package.

This file makes the ``src`` directory a Python package and re-exports
commonly used helpers. Downstream modules can import directly from
``llm_document_ingestion`` without referring to internal modules.

Note: We deliberately do not perform any heavy initialisation here.
Objects like ``DocParser`` or ``GeminiVLM`` will only be constructed
when needed by callers. See the individual modules for more details.
"""

from .doc_parser import DocParser, Figure  # noqa: F401
from .markdown import FigureDesc, compose_markdown  # noqa: F401
from .chunk import chunk_markdown  # noqa: F401
from .embed import embed_and_upload_json  # noqa: F401
from .vlm import GeminiVLM, VLMConfig  # noqa: F401
from .prompts import default_prompt  # noqa: F401
