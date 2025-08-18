"""
Smart text chunking.

When PDF pages do not contain numbered headings or when the structure
cannot be reliably inferred, this module provides a fallback chunking
strategy. The text is split into paragraphs separated by blank lines,
and paragraphs are grouped until a token count limit is reached. The
token limit is approximate and based on the number of words; it does
not account for subword tokenisation but provides a reasonable upper
bound to keep embedding sizes manageable.
"""
from __future__ import annotations

import re
from typing import List


def smart_chunk(text: str, max_tokens: int = 900) -> List[str]:
    """
    Break a block of text into manageable chunks based on paragraph
    boundaries and a maximum token count. Paragraphs are separated by
    one or more blank lines.

    Parameters
    ----------
    text: str
        The input text to chunk.
    max_tokens: int
        Approximate maximum number of tokens (words) per chunk. Defaults
        to 900. If a single paragraph exceeds this limit it will be
        emitted as its own chunk.

    Returns
    -------
    List[str]
        A list of chunk strings, each containing one or more paragraphs.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        # Approximate token count as number of words
        token_count = len(para.split())
        if current_tokens + token_count > max_tokens and current_chunk:
            # Emit current chunk and start a new one
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [para]
            current_tokens = token_count
        else:
            current_chunk.append(para)
            current_tokens += token_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks
