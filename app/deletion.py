"""
Deletion utilities for ingested documents.

This module provides helpers to permanently remove points from
Qdrant. Hard deletions are performed based on document identifiers or
document page identifiers. Soft deletes (modifying the effective
period) are not implemented in this version since the requirement
explicitly favours hard deletion.
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import AppConfig
from .qdrant_client import (
    delete_by_filter,
    build_doc_filter,
    build_doc_page_filter,
)

logger = logging.getLogger(__name__)


def delete_document(client: QdrantClient, config: AppConfig, doc_id: str) -> None:
    """
    Permanently delete all points belonging to a document.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration.
    doc_id: str
        Document identifier used during ingestion.
    """
    logger.info("Deleting all points for document '%s'", doc_id)
    flt = build_doc_filter(doc_id)
    delete_by_filter(client, config, flt)


def delete_document_page(client: QdrantClient, config: AppConfig, doc_id: str, page: int) -> None:
    """
    Permanently delete all points associated with a specific page of a document.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration.
    doc_id: str
        Document identifier used during ingestion.
    page: int
        Page number (1-based) to delete.
    """
    logger.info("Deleting page %d of document '%s'", page, doc_id)
    flt = build_doc_page_filter(doc_id, page)
    delete_by_filter(client, config, flt)
