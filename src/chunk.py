"""
Structure‑aware chunker for Markdown documents.

This module exposes a single function, :func:`chunk_markdown`, which
splits a markdown string into a list of chunks suitable for
embedding.  The algorithm mirrors the upstream implementation and
operates synchronously.  Unlike the CLI script in the original
repository, this file does not include a command‑line interface.

The chunking strategy proceeds in four steps:

1. Build atomic blocks consisting of one or more headings followed by
   their associated body.  A preface before the first heading is
   treated as a block.
2. Merge heading‑only blocks into the next block when possible, or
   append them to the previous block when at the end of the document.
3. Greedily pack blocks into chunks without splitting a block across
   chunk boundaries.  Oversized single blocks are allowed to exceed
   ``max_chars`` so that the heading/body association is preserved.
4. Rebalance adjacent chunks by moving as many leading blocks as will
   fit from the next chunk into the current one.

See the upstream ``chunk.py`` for more details.
"""

from __future__ import annotations

import re
from typing import List


def chunk_markdown(text: str, max_chars: int = 1024) -> List[str]:
    """
    Split a markdown document into chunks of roughly ``max_chars``
    characters while respecting structural boundaries.

    :param text: The markdown string to split.
    :param max_chars: Maximum length (in characters) of each chunk.  A
        single block that exceeds this limit will form its own chunk.
    :return: A list of markdown strings.
    """

    def is_heading(line: str) -> bool:
        # Up to 3 leading spaces, then 1–6 '#' and a space
        return re.match(r'^\s{0,3}#{1,6}\s', line) is not None

    def join_blocks(blocks: List[str]) -> str:
        return "\n\n".join(blocks)

    def joined_len(blocks: List[str]) -> int:
        if not blocks:
            return 0
        # total content + separators of "\n\n" between blocks
        return sum(len(b) for b in blocks) + 2 * (len(blocks) - 1)

    # -------- Step 1: build atomic blocks (headings group + body) --------
    lines = text.split("\n")
    n = len(lines)
    blocks: List[str] = []
    i = 0
    # Preface before first heading
    preface_start = i
    while i < n and not is_heading(lines[i]):
        i += 1
    if i > preface_start:
        preface = "\n".join(lines[preface_start:i]).strip()
        if preface:
            blocks.append(preface)
    # Heading groups + bodies
    while i < n:
        # Collect consecutive headings
        h_start = i
        while i < n and is_heading(lines[i]):
            i += 1
        headings_block = "\n".join(lines[h_start:i]).strip()
        # Collect body until next heading or EOF
        b_start = i
        while i < n and not is_heading(lines[i]):
            i += 1
        body_block = "\n".join(lines[b_start:i]).strip()
        if headings_block and body_block:
            blocks.append(f"{headings_block}\n\n{body_block}")
        elif headings_block and not body_block:
            # heading‑only (possibly at EOF) — keep for now, fix next step
            blocks.append(headings_block)
        elif body_block:
            # body without heading (rare tail)
            blocks.append(body_block)
    # Strip empties
    blocks = [b for b in blocks if b and b.strip()]
    # -------- Step 2: fix heading‑only blocks (prefer merge into NEXT) --------
    def block_is_heading_only(block: str) -> bool:
        for ln in block.splitlines():
            if not ln.strip():
                continue
            if not is_heading(ln):
                return False
        return True
    fixed: List[str] = []
    idx = 0
    while idx < len(blocks):
        blk = blocks[idx]
        if block_is_heading_only(blk):
            if idx + 1 < len(blocks):
                # prepend heading‑only block to the next block
                merged = blk.rstrip() + "\n\n" + blocks[idx + 1].lstrip()
                fixed.append(merged)
                idx += 2  # consumed next
                continue
            elif fixed:
                # trailing heading‑only: append to previous
                fixed[-1] = fixed[-1].rstrip() + "\n\n" + blk
                idx += 1
                continue
            else:
                fixed.append(blk)
                idx += 1
                continue
        else:
            fixed.append(blk)
            idx += 1
    blocks = fixed
    # -------- Step 3: greedy pack blocks into chunks (no splits) --------
    chunk_blocks: List[List[str]] = []
    cur: List[str] = []
    for blk in blocks:
        if not cur:
            if len(blk) > max_chars:
                # Oversized single block stands alone (overflow allowed)
                chunk_blocks.append([blk])
            else:
                cur = [blk]
        else:
            candidate_len = joined_len(cur) + 2 + len(blk)
            if candidate_len <= max_chars:
                cur.append(blk)
            else:
                chunk_blocks.append(cur)
                if len(blk) > max_chars:
                    chunk_blocks.append([blk])
                    cur = []
                else:
                    cur = [blk]
    if cur:
        chunk_blocks.append(cur)
    # -------- Step 4: rebalance by prepending from next into current --------
    # For each boundary, move as many leading blocks as will fit from next chunk into current.
    i = 0
    while i < len(chunk_blocks) - 1:
        cur_blocks = chunk_blocks[i]
        nxt_blocks = chunk_blocks[i + 1]
        moved = False
        while nxt_blocks:
            candidate_len = (
                joined_len(cur_blocks) + 2 + len(nxt_blocks[0]) if cur_blocks else len(nxt_blocks[0])
            )
            if candidate_len <= max_chars:
                # move first block of next into end of current
                cur_blocks.append(nxt_blocks.pop(0))
                moved = True
            else:
                break
        # If next chunk becomes empty, remove it
        if not nxt_blocks:
            chunk_blocks.pop(i + 1)
            # Do not increment i; try rebalancing with the new neighbour as well
            moved = True
            continue
        # Otherwise, move to the next boundary
        i += 1
    # Join blocks into final chunk strings
    chunks = [join_blocks(cb) for cb in chunk_blocks if cb]
    return chunks


__all__ = ["chunk_markdown"]
