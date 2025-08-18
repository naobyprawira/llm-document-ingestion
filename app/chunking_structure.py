"""
Structured text chunking based on hierarchical headings.

This module implements a parser that splits a block of text into
sections using headings of the form ``<n>.<m>.<p> ...``. Only up to
three levels of headings are recognised. Headings deeper than three
levels (e.g. ``6.1.1.1``) are assigned to their closest ancestor
heading (``6.1.1``). This ensures that nested subtopics remain within
the context of their parent section.

When no headings are detected in a given text block, the entire block
is returned as a single chunk.

Each returned chunk is a tuple ``(heading, body)``, where ``heading``
may be an empty string if no numbered heading was found, and ``body``
is the associated text.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple


_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+(.+)")


def chunk_by_heading(text: str) -> List[Tuple[str, str]]:
    """
    Split text into chunks based on numbered headings.

    Parameters
    ----------
    text: str
        Input text, potentially containing multiple sections with
        headings. Lines should be separated by newline characters.

    Returns
    -------
    List[Tuple[str, str]]
        A list of (heading, body) pairs. The heading is the
        hierarchical number (e.g. ``"6.1"``) and the body is the text
        following the heading up to but not including the next heading.
        If no headings are found, the list contains a single chunk
        with an empty heading and the entire input text as the body.
    """
    lines = text.splitlines()
    chunks: List[Tuple[str, List[str]]] = []
    current_heading: str | None = None
    current_body: List[str] = []

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            # Flush previous chunk
            if current_heading is not None or current_body:
                heading = current_heading or ""
                chunks.append((heading, current_body))
                current_body = []

            number = m.group(1)
            title = m.group(2)
            # Normalise the heading to at most 3 levels
            parts = number.split(".")
            if len(parts) > 3:
                number = ".".join(parts[:3])
            # Combine the number and title into a single heading string
            current_heading = number + " " + title
        else:
            current_body.append(line)

    # Append last chunk
    if current_heading is not None or current_body:
        heading = current_heading or ""
        chunks.append((heading, current_body))

    # Convert body lists to strings and strip leading/trailing whitespace
    result: List[Tuple[str, str]] = []
    for heading, body_lines in chunks:
        body_text = "\n".join(body_lines).strip()
        # If body_text is empty and heading has content, at least keep the heading
        result.append((heading, body_text))

    return result
