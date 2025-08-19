"""
Qdrant client utilities.

This module wraps common Qdrant operations such as ensuring the collection
exists, upserting points with payloads, and deleting points via filters.

The functions defined here rely on the configuration defined in
``app.config``.  They abstract away the low‑level details of connecting
to Qdrant and managing payload indices.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import AppConfig

logger = logging.getLogger(__name__)


def get_client(config: AppConfig) -> QdrantClient:
    """Create a Qdrant client based on the provided configuration."""
    return QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)


def ensure_collection(client: QdrantClient, config: AppConfig, vector_dim: int) -> None:
    """
    Ensure that the configured collection exists and has appropriate payload
    indexes.  If the collection does not exist, it is created with the
    specified vector dimension.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration, provides the collection name.
    vector_dim: int
        The dimension of the vectors that will be stored in the collection.
    """
    collection = config.qdrant.collection
    existing = client.get_collections().collections
    if not any(c.name == collection for c in existing):
        logger.info("Creating Qdrant collection '%s' with vector size %d", collection, vector_dim)
        client.recreate_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
            optimizers_config=qm.OptimizersConfigDiff(memmap_threshold=20000),
        )
        # Create payload indexes to accelerate filtering by metadata fields.
        index_fields = {
            "type": qm.PayloadSchemaType.KEYWORD,
            "doc_id": qm.PayloadSchemaType.KEYWORD,
            "scope": qm.PayloadSchemaType.KEYWORD,
            "company": qm.PayloadSchemaType.KEYWORD,
            "country": qm.PayloadSchemaType.KEYWORD,
            "language": qm.PayloadSchemaType.KEYWORD,
            "version": qm.PayloadSchemaType.KEYWORD,
            "security_class": qm.PayloadSchemaType.KEYWORD,
            "effective_from": qm.PayloadSchemaType.KEYWORD,
            "effective_to": qm.PayloadSchemaType.KEYWORD,
            "page": qm.PayloadSchemaType.INTEGER,
            "chunk_id": qm.PayloadSchemaType.KEYWORD,
        }
        for field_name, schema_type in index_fields.items():
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception as e:  # pragma: no cover - index creation should succeed
                logger.warning("Failed to create index for field '%s': %s", field_name, e)


def upsert_points(
    client: QdrantClient,
    config: AppConfig,
    points: Sequence[qm.PointStruct],
) -> None:
    """
    Upsert a batch of points into the configured collection.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration providing the collection name.
    points: Sequence[qm.PointStruct]
        A sequence of points to upsert.
    """
    if not points:
        return
    client.upsert(collection_name=config.qdrant.collection, points=points)


def delete_by_filter(client: QdrantClient, config: AppConfig, flt: qm.Filter) -> None:
    """
    Permanently delete all points matching the given filter in the configured
    collection.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration providing the collection name.
    flt: qm.Filter
        The filter specifying which points to delete.
    """
    client.delete(collection_name=config.qdrant.collection, points_selector=qm.FilterSelector(filter=flt))


def soft_delete_by_filter(
    client: QdrantClient,
    config: AppConfig,
    flt: qm.Filter,
    effective_to: Optional[str] = None,
) -> None:
    """
    Soft delete all points matching the given filter by updating their
    ``effective_to`` payload field.  This preserves the points but marks
    them as inactive for retrieval.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration providing the collection name.
    flt: qm.Filter
        The filter specifying which points to update.
    effective_to: Optional[str]
        ISO timestamp to set for the ``effective_to`` field.  If omitted,
        the current UTC timestamp will be used.
    """
    from datetime import datetime, timezone

    # Default to current time if not provided
    eff_to = effective_to or datetime.now(timezone.utc).isoformat()
    try:
        client.update_payload(
            collection_name=config.qdrant.collection,
            points_selector=qm.FilterSelector(filter=flt),
            payload={"effective_to": eff_to},
        )
    except Exception as e:  # pragma: no cover - should succeed
        logger.error("Soft delete failed: %s", e)


def delete_document(client: QdrantClient, config: AppConfig, doc_id: str, *, soft: bool = False) -> None:
    """
    Delete or soft delete all points belonging to a document.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration providing the collection name.
    doc_id: str
        Identifier of the document to delete.
    soft: bool, optional
        When True, perform a soft deletion by setting the ``effective_to``
        payload; otherwise, remove the points permanently.
    """
    flt = build_doc_filter(doc_id)
    if soft:
        soft_delete_by_filter(client, config, flt)
    else:
        delete_by_filter(client, config, flt)


def delete_document_page(
    client: QdrantClient, config: AppConfig, doc_id: str, page: int, *, soft: bool = False
) -> None:
    """
    Delete or soft delete all points belonging to a specific page of a document.

    Parameters
    ----------
    client: QdrantClient
        The Qdrant client.
    config: AppConfig
        Application configuration providing the collection name.
    doc_id: str
        Identifier of the document.
    page: int
        Page number (1‑indexed) of the document to delete.
    soft: bool, optional
        When True, perform a soft deletion; otherwise, remove permanently.
    """
    flt = build_doc_page_filter(doc_id, page)
    if soft:
        soft_delete_by_filter(client, config, flt)
    else:
        delete_by_filter(client, config, flt)


def build_doc_filter(doc_id: str) -> qm.Filter:
    """Build a Qdrant filter matching all points belonging to a document."""
    return qm.Filter(
        must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        ]
    )


def build_doc_page_filter(doc_id: str, page: int) -> qm.Filter:
    """Build a Qdrant filter matching all points for a specific document page."""
    return qm.Filter(
        must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key="page", match=qm.MatchValue(value=page)),
        ]
    )
