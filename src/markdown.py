"""
Markdown composition helpers.

This module defines a simple dataclass representing a figure
description and a function to append a "Deskripsi Gambar" section to
the end of a markdown document.  It is identical to the upstream
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FigureDesc:
    index: int
    page: Optional[int]
    caption: str
    description: str


def compose_markdown(base_md: str, figures: List[FigureDesc]) -> str:
    """Append figure descriptions to the end of a markdown document.

    :param base_md: The base markdown content extracted from the document.
    :param figures: A list of :class:`FigureDesc` describing each figure.
    :return: A new markdown string with an appended "Deskripsi Gambar" section.
    """
    if not figures:
        return base_md
    parts = [base_md, "\n\n---\n\n## Deskripsi Gambar\n"]
    for f in figures:
        title = f"### Gambar {f.index}"
        if f.page:
            title += f" (Halaman {f.page})"
        parts.append(f"{title}\n\n")
        if f.caption.strip():
            parts.append(f"**Caption:** {f.caption.strip()}\n\n")
        parts.append(f"{f.description.strip() or '-'}\n\n")
    return "".join(parts)


__all__ = ["FigureDesc", "compose_markdown"]