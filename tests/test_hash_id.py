"""Tests for deterministic point identifier generation.

The ``point_id`` function should return the same integer for the same
combination of document ID, page number and chunk identifier, and
different integers for different combinations.
"""

from app.utils.hash_id import point_id


def test_point_id_deterministic() -> None:
    """Ensure that point identifiers are deterministic and unique per input."""
    id1 = point_id("doc123", 1, "1")
    id2 = point_id("doc123", 1, "1")
    id3 = point_id("doc123", 1, "2")
    id4 = point_id("doc123", 2, "1")
    # Deterministic: same inputs yield the same ID
    assert id1 == id2
    # Unique: different chunk or page yields different IDs
    assert id1 != id3
    assert id1 != id4
