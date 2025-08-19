"""
Embedding utilities.

Always uses Google's Generative AI embedding API with
model 'models/gemini-embedding-001'.
Requires GOOGLE_API_KEY (or GEMINI_API_KEY / GENAI_API_KEY) in env.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Sequence

try:
    import google.generativeai as genai  # pip install google-generativeai
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing dependency: pip install google-generativeai") from e

from ..config import AppConfig

logger = logging.getLogger(__name__)

# Allow override via env while defaulting to Gemini embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# Very small heuristic for retryable failures
_RETRY_HINTS = ("429", "rate", "timeout", "temporar", "unavailable", "5")


def _extract_vector(resp) -> List[float]:
    """
    Normalize SDK return shapes.
    """
    if isinstance(resp, dict):
        vec = resp.get("embedding") or (resp.get("data") or {}).get("embedding")
    else:
        vec = getattr(resp, "embedding", None)
    if vec is None:
        raise RuntimeError("Embedding response missing 'embedding'.")
    return [float(x) for x in vec]


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

    def embed_text(
        self,
        text: str,
        *,
        task_type: str = "retrieval_query",
        max_retries: int = 3,
        base_delay_s: float = 1.0,
    ) -> List[float]:
        """Return embedding vector for a single text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        delay = base_delay_s
        for attempt in range(max_retries):
            try:
                resp = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,            # IMPORTANT: use content=, not prompt=
                    task_type=task_type,
                )
                return _extract_vector(resp)
            except Exception as e:
                # Retry on likely transient errors
                msg = str(e).lower()
                retryable = any(h in msg for h in _RETRY_HINTS)
                if attempt < max_retries - 1 and retryable:
                    logger.warning("embed_text retry %d due to: %s", attempt + 1, e)
                    time.sleep(delay)
                    delay *= 2
                    continue
                logger.error("Gemini embed_content failed: %s", e)
                raise

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        task_type: str = "retrieval_document",
        max_retries: int = 3,
    ) -> List[List[float]]:
        """
        Simple and robust batch helper. Loops per item to avoid SDK batch
        shape changes across versions.
        """
        out: List[List[float]] = []
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                continue
            out.append(self.embed_text(t, task_type=task_type, max_retries=max_retries))
        return out
