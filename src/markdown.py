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