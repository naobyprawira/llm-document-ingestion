"""Helpers for comparing OneDrive inventories against Qdrant filenames."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.http import models  # type: ignore


@dataclass
class DriveItem:
    """Normalized representation of a OneDrive file."""

    raw_name: str
    base_name: str
    base_key: str
    category: str
    download_url: str


@dataclass
class CategoryDiff:
    """Comparison summary for a single category."""

    category: str
    items: List[DriveItem]
    item_lookup: Dict[str, DriveItem]
    qdrant_records: Dict[str, models.Record]
    drive_keys: set[str]
    qdrant_keys: set[str]
    stale_keys: List[str]
    missing_keys: List[str]


def _normalize_name(name: str) -> Tuple[str, str]:
    base = os.path.splitext(name)[0].strip()
    return base, base.lower()


def _infer_category(item: Dict[str, object]) -> Optional[str]:
    if isinstance(item.get("category"), str) and item["category"]:
        return item["category"]  # type: ignore[return-value]
    parent = item.get("parentReference")
    if isinstance(parent, dict):
        if isinstance(parent.get("name"), str) and parent["name"].strip():
            return parent["name"]  # type: ignore[return-value]
        path = parent.get("path")
        if isinstance(path, str) and path:
            tail = path.split("/")
            if tail:
                return tail[-1]
    return None


def drive_items_from_payload(
    payload: object,
    *,
    default_category: Optional[str],
    require_download: bool = True,
) -> List[DriveItem]:
    if isinstance(payload, dict) and "value" in payload:
        items = payload["value"]  # type: ignore[assignment]
    else:
        items = payload

    if not isinstance(items, list):
        raise ValueError("Inventory payload must be a list or a dict with 'value'.")

    results: List[DriveItem] = []
    seen: set[Tuple[str, str]] = set()

    for raw in items:
        if not isinstance(raw, dict):
            continue
        file_info = raw.get("file")
        if require_download and not isinstance(file_info, dict):
            continue  # skip folders
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        category = (
            _infer_category(raw)
            if default_category is None
            else default_category
        )
        if category is None:
            raise ValueError(f"Missing category for item '{name}'.")
        category = category.strip()
        if not category:
            raise ValueError(f"Missing category for item '{name}'.")

        base, key = _normalize_name(name)
        cat_key = (category.lower(), key)

        if cat_key in seen:
            continue

        download_url = raw.get("@microsoft.graph.downloadUrl") or raw.get("downloadUrl")
        if require_download and not isinstance(download_url, str):
            raise ValueError(f"Missing download URL for '{name}'.")

        seen.add(cat_key)
        results.append(
            DriveItem(
                raw_name=name,
                base_name=base,
                base_key=key,
                category=category,
                download_url=download_url if isinstance(download_url, str) else "",
            )
        )
    return results


def group_items_by_category(items: Iterable[DriveItem]) -> Dict[str, List[DriveItem]]:
    grouped: Dict[str, List[DriveItem]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)
    return grouped


def qdrant_records_for_category(
    client: QdrantClient,
    *,
    collection: str,
    category: str,
) -> Dict[str, models.Record]:
    """Return one representative record per filename for the category."""
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchValue(value=category),
            ),
        ]
    )

    results: Dict[str, models.Record] = {}
    offset: Optional[models.PointId] = None

    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        if not records:
            break
        for record in records:
            metadata = (record.payload or {}).get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            filename = metadata.get("filename")
            if isinstance(filename, str) and filename:
                key = filename.lower()
                results.setdefault(key, record)
        if offset is None:
            break
    return results


def compute_category_diffs(
    client: QdrantClient,
    *,
    collection: str,
    grouped_items: Dict[str, List[DriveItem]],
) -> List[CategoryDiff]:
    diffs: List[CategoryDiff] = []
    for category, items in grouped_items.items():
        qdrant_records = qdrant_records_for_category(
            client,
            collection=collection,
            category=category,
        )

        drive_keys = {item.base_key for item in items}
        qdrant_keys = set(qdrant_records.keys())

        stale_keys = sorted(qdrant_keys - drive_keys)
        missing_keys = sorted(drive_keys - qdrant_keys)
        item_lookup = {item.base_key: item for item in items}

        diffs.append(
            CategoryDiff(
                category=category,
                items=items,
                item_lookup=item_lookup,
                qdrant_records=qdrant_records,
                drive_keys=drive_keys,
                qdrant_keys=qdrant_keys,
                stale_keys=stale_keys,
                missing_keys=missing_keys,
            )
        )
    return diffs


__all__ = [
    "DriveItem",
    "CategoryDiff",
    "drive_items_from_payload",
    "group_items_by_category",
    "compute_category_diffs",
    "qdrant_records_for_category",
]
