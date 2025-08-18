"""
Application configuration and constants.

This module reads environment variables and exposes configuration values
used throughout the ingestion pipeline. Defaults are chosen to make
development simple while allowing override via environment.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from dataclasses import dataclass


@dataclass
class QdrantSettings:
    """Settings for connecting to Qdrant."""

    url: str
    api_key: str | None
    collection: str


@dataclass
class AppConfig:
    """Top-level application configuration."""

    # Qdrant configuration
    qdrant: QdrantSettings
    # Gemini API key (used for VLM and embeddings)
    google_api_key: str | None
    # Default language for document metadata
    default_language: str
    # Maximum tokens per chunk when performing smart chunking
    max_tokens_per_chunk: int
    # DPI used when rendering PDF pages to images for VLM
    image_dpi: int


def load_config() -> AppConfig:
    """
    Load configuration from environment variables.

    The following environment variables are recognised:

    - QDRANT_URL: URL of the Qdrant instance. Defaults to "http://localhost:6333".
    - QDRANT_API_KEY: API key for Qdrant if using cloud instance. Optional.
    - QDRANT_COLLECTION: Name of the collection to use. Defaults to "hr_docs".
    - GOOGLE_API_KEY: API key for Google Generative AI. Required for VLM and embeddings.
    - DEFAULT_LANGUAGE: Language code used when none specified. Defaults to "id".
    - MAX_TOKENS_PER_CHUNK: Token limit for smart text chunking. Defaults to 900.
    - IMAGE_DPI: Resolution in dots per inch used when converting PDF pages to PNG. Defaults to 350.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "hr_docs")
    google_api_key = os.getenv("GOOGLE_API_KEY") or None
    default_language = os.getenv("DEFAULT_LANGUAGE", "id")
    max_tokens = int(os.getenv("MAX_TOKENS_PER_CHUNK", "900"))
    image_dpi = int(os.getenv("IMAGE_DPI", "350"))

    return AppConfig(
        qdrant=QdrantSettings(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection=qdrant_collection,
        ),
        google_api_key=google_api_key,
        default_language=default_language,
        max_tokens_per_chunk=max_tokens,
        image_dpi=image_dpi,
    )
