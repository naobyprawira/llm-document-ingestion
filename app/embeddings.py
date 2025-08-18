"""
Embedding utilities.

Always uses Google's Generative AI embedding API with
model 'models/gemini-embedding-001'.
Requires GOOGLE_API_KEY (or GEMINI_API_KEY / GENAI_API_KEY) in env.
"""
from __future__ import annotations

import logging
import os
from typing import List

try:
    import google.generativeai as genai  # pip install google-generativeai
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing dependency: pip install google-generativeai") from e

from .config import AppConfig

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")


class EmbeddingModel:
    """Wrapper around Gemini embeddings."""

    def __init__(self, config: AppConfig) -> None:
        # Resolve API key from config or env.
        api_key = (
            getattr(config, "google_api_key", None)
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set (or GEMINI_API_KEY/GENAI_API_KEY).")

        # Configure SDK once.
        genai.configure(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        """Return embedding vector for a single text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            resp = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_query",
            )
        except Exception as e:
            logger.error("Gemini embed_content failed: %s", e)
            raise

        # Support dict or object-like responses
        vec = None
        if isinstance(resp, dict):
            vec = resp.get("embedding") or (resp.get("data") or {}).get("embedding")
        else:
            vec = getattr(resp, "embedding", None)

        if vec is None:
            raise RuntimeError("Embedding response missing 'embedding' field.")

        return [float(x) for x in vec]
