"""
Qdrant client utilities.

This module wraps common Qdrant operations such as ensuring the collection
exists, upserting points with payloads, and deleting points via filters.

The functions defined here rely on the configuration defined in
``app.config``. They abstract away the low-level details of connecting
to Qdrant and managing payload indices.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

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
    indexes. If the collection does not exist, it is created with the
    specified vector dimension.
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
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to create index for field '%s': %s", field_name, e)


def upsert_points(
    client: QdrantClient,
    config: AppConfig,
    payloads_or_points: Union[Sequence[qm.PointStruct], Sequence[dict]],
    vectors: Optional[Sequence[Sequence[float]]] = None,
    point_ids: Optional[Sequence[int]] = None,
) -> None:
    """
    Upsert vectors into Qdrant.

    Two call modes are supported:
      1) upsert_points(client, config, [PointStruct, ...])
      2) upsert_points(client, config, [payload, ...], [vector, ...], point_ids=[...])

    When explicit IDs are provided, they are attached to the corresponding points.
    """
    if not payloads_or_points:
        return

    # Mode 1: user already built PointStructs
    if vectors is None and isinstance(payloads_or_points[0], qm.PointStruct):  # type: ignore[index]
        points = list(payloads_or_points)  # type: ignore[assignment]
    else:
        # Mode 2: build PointStructs from payloads + vectors (+ optional ids)
        payloads = list(payloads_or_points)  # type: ignore[assignment]
        if vectors is None:
            raise ValueError("vectors must be provided when passing payload dictionaries")
        if len(payloads) != len(vectors):
            raise ValueError("payloads and vectors must have the same length")
        if point_ids is not None and len(point_ids) != len(payloads):
            raise ValueError("point_ids length must match payloads length when provided")

        points: List[qm.PointStruct] = []
        for i, (pl, vec) in enumerate(zip(payloads, vectors)):
            pid = None if point_ids is None else int(point_ids[i])
            points.append(qm.PointStruct(id=pid, vector=[float(x) for x in vec], payload=pl))

    client.upsert(collection_name=config.qdrant.collection, points=points, wait=True)


def delete_by_filter(client: QdrantClient, config: AppConfig, flt: qm.Filter) -> None:
    """Permanently delete all points matching the given filter."""
    client.delete(collection_name=config.qdrant.collection, points_selector=qm.FilterSelector(filter=flt))


def soft_delete_by_filter(
    client: QdrantClient,
    config: AppConfig,
    flt: qm.Filter,
    effective_to: Optional[str] = None,
) -> None:
    """
    Soft delete all points matching the filter by updating their ``effective_to`` payload.
    """
    from datetime import datetime, timezone

    eff_to = effective_to or datetime.now(timezone.utc).isoformat()
    try:
        client.update_payload(
            collection_name=config.qdrant.collection,
            points_selector=qm.FilterSelector(filter=flt),
            payload={"effective_to": eff_to},
        )
    except Exception as e:  # pragma: no cover
        logger.error("Soft delete failed: %s", e)


def delete_document(client: QdrantClient, config: AppConfig, doc_id: str, *, soft: bool = False) -> None:
    """Delete or soft delete all points for a document."""
    flt = build_doc_filter(doc_id)
    if soft:
        soft_delete_by_filter(client, config, flt)
    else:
        delete_by_filter(client, config, flt)


def delete_document_page(
    client: QdrantClient, config: AppConfig, doc_id: str, page: int, *, soft: bool = False
) -> None:
    """Delete or soft delete all points for a specific page of a document."""
    flt = build_doc_page_filter(doc_id, page)
    if soft:
        soft_delete_by_filter(client, config, flt)
    else:
        delete_by_filter(client, config, flt)


def build_doc_filter(doc_id: str) -> qm.Filter:
    """Filter matching all points belonging to a document."""
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])


def build_doc_page_filter(doc_id: str, page: int) -> qm.Filter:
    """Filter matching all points for a specific document page."""
    return qm.Filter(
        must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key="page", match=qm.MatchValue(value=page)),
        ]
    )
