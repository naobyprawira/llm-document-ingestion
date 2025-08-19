"""
Structured text chunking for HR/Compliance documents.

Goals:
- Respect numbered headings like "6", "6.1", "6.1.1".
- Cap depth at 3. Fold deeper like "6.1.1.1" into "6.1.1".
- Never cut inside a paragraph. Split on blank lines only.
- Produce chunks sized for embeddings using word-count limits.
- Return simple tuples for easy wiring: (heading, content).

Public API:
- chunk_document(content: str, max_words: int = 900, min_words: int = 200)
    -> list[tuple[str, str]]
- chunk_by_heading(content: str) -> list[tuple[str, str]]
- smart_chunk(content: str, max_words: int = 900, min_words: int = 200)
    -> list[str]
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Matches "6", "6.1", "6.1.1" + title. Depth >3 is folded by trimming.
_HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,10})\s+(?P<title>.+)$")

def _fold_heading_number(num: str) -> str:
    parts = num.split(".")
    return ".".join(parts[:3])  # keep at most 3 levels

def _is_heading(line: str) -> tuple[bool, str, str]:
    m = _HEADING_RE.match(line.strip())
    if not m:
        return False, "", ""
    num = _fold_heading_number(m.group("num"))
    title = m.group("title").strip()
    return True, num, title

def _word_count(s: str) -> int:
    return len(s.split())

def _split_paragraphs(text: str) -> List[str]:
    # Normalize Windows/Mac line endings and collapse >2 blank lines to 2.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    # Split on blank lines. Keep paragraph integrity.
    paras = [p.strip() for p in re.split(r"\n\s*\n", normalized)]
    return [p for p in paras if p]

def smart_chunk(
    content: str,
    max_words: int = 900,
    min_words: int = 200,
) -> List[str]:
    """
    Paragraph-aware chunking. Never cuts inside a paragraph.
    Builds chunks by appending whole paragraphs until max_words is met.
    Ensures last tiny tail merges into previous chunk if < min_words.
    """
    paras = _split_paragraphs(content)
    if not paras:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_wc = 0

    for p in paras:
        wc = _word_count(p)
        if cur_wc and cur_wc + wc > max_words:
            # finalize current chunk
            chunks.append("\n\n".join(cur).strip())
            cur, cur_wc = [], 0
        cur.append(p)
        cur_wc += wc

    if cur:
        chunks.append("\n\n".join(cur).strip())

    # Merge tiny tail
    if len(chunks) >= 2 and _word_count(chunks[-1]) < min_words:
        last = chunks.pop()
        chunks[-1] = (chunks[-1] + "\n\n" + last).strip()

    return [c for c in chunks if c]

def chunk_by_heading(content: str) -> List[Tuple[str, str]]:
    """
    Split by numbered headings. Returns (heading, content) pairs.
    Heading is "6.1.1 Title" or "" when no heading seen yet.
    """
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    chunks: List[Tuple[str, str]] = []
    current_heading = ""
    buf: List[str] = []

    def flush():
        nonlocal buf, current_heading
        body = "\n".join(buf).strip()
        if body:
            chunks.append((current_heading, body))
        buf = []

    for raw in lines:
        line = raw.rstrip()
        is_h, num, title = _is_heading(line)
        if is_h:
            flush()
            current_heading = f"{num} {title}".strip()
        else:
            buf.append(line)

    flush()
    # Remove leading empty chunk if file starts with heading and no preface.
    return [(h, b) for (h, b) in chunks if b]

def chunk_document(
    content: str,
    max_words: int = 900,
    min_words: int = 200,
) -> List[Tuple[str, str]]:
    """
    Full policy:
    1) Try heading-based split.
    2) For each heading body, apply paragraph smart_chunk if oversized.
    3) If no headings at all, smart_chunk the whole document.
    """
    by_head = chunk_by_heading(content)
    if not by_head:
        return [("", c) for c in smart_chunk(content, max_words, min_words)]

    out: List[Tuple[str, str]] = []
    for heading, body in by_head:
        pieces = smart_chunk(body, max_words, min_words)
        if not pieces:
            continue
        if len(pieces) == 1:
            out.append((heading, pieces[0]))
        else:
            # Keep heading label for all sub-chunks
            out.extend((heading, p) for p in pieces)

    return out
