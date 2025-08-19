"""
Flowchart detection and VLM extraction (plain text mode).

- Detect likely-flowchart pages via vector primitives.
- Render page to PNG and ask Gemini to return ONLY a numbered list of steps:
  1. ...
  2. ...
  3. ...
- No JSON is parsed. We extract numbered lines with a regex and renumber.

Public API:
- detect_flowchart(doc, page_index) -> bool
- extract_flowchart_steps(doc, page_index, config, *, force=False) -> dict | None
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai  # pip install google-generativeai

from ..config import AppConfig
from ..parsers.pdf_text import render_page_to_png

log = logging.getLogger(__name__)

# Keep the instruction minimal to reduce extra prose.
FLOWCHART_STEPS_PROMPT = (
    "You will see a PNG of a process flowchart.\n"
    "OUTPUT FORMAT STRICTLY:\n"
    "1. <first step>\n2. <second step>\n3. <third step>\n"
    "Rules: numbered list only, one step per line, no headings, no JSON, no code fences, no extra text."
)

# Heuristics for detection using vector shapes.
_DRAW_MIN_BOXES = 4
_DRAW_MIN_LINES = 6


def _count_shapes(page) -> Tuple[int, int]:
    rects = 0
    lines = 0
    try:
        for item in page.get_drawings():
            if item.get("rect"):
                rects += 1
            for it in item.get("items", ()):
                if it and it[0] == "l":
                    lines += 1
    except Exception as e:  # pragma: no cover
        log.debug("get_drawings failed: %s", e)
    return rects, lines


def detect_flowchart(doc, page_index: int) -> bool:
    page = doc.load_page(page_index)
    boxes, segs = _count_shapes(page)
    return boxes >= _DRAW_MIN_BOXES and segs >= _DRAW_MIN_LINES


_NUM_LINE_RE = re.compile(r"^\s*(\d+)[\.\)\-]\s+(.*\S)\s*$")


def _extract_numbered_lines(raw: str) -> List[str]:
    """Extract '1. ...' lines and return the step texts."""
    lines = []
    for line in raw.splitlines():
        m = _NUM_LINE_RE.match(line)
        if m:
            lines.append(m.group(2).strip())
    return lines


def extract_flowchart_steps(
    doc,
    page_index: int,
    config: AppConfig,
    *,
    force: bool = False,
    timeout_s: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Returns:
      {"has_flowchart": False} when not detected or no valid steps.
      {"has_flowchart": True, "title": "", "steps_text": "1. ...\\n2. ..."} on success.
    """
    if not force and not detect_flowchart(doc, page_index):
        return {"has_flowchart": False}

    image_bytes = render_page_to_png(doc, page_index, dpi=config.image_dpi)

    genai.configure(api_key=config.google_api_key)
    # Plain text output; temperature 0 to reduce fluff.
    model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"temperature": 0.0})

    try:
        resp = model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},
                FLOWCHART_STEPS_PROMPT,
            ],
            request_options={"timeout": timeout_s},
        )
        raw = (resp.text or "").strip()
    except Exception as e:
        log.warning("Gemini VLM call failed on page %d: %s", page_index + 1, e)
        return None

    steps = _extract_numbered_lines(raw)

    # Require at least 2 steps to consider it a flowchart
    if len(steps) < 2:
        log.debug("No numbered steps extracted on page %d. Raw preview: %.120s", page_index + 1, raw.replace("\n", " "))
        return {"has_flowchart": False}

    # Renumber cleanly to avoid '1., 3., 4.' artifacts
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps[:50]))

    return {
        "has_flowchart": True,
        "title": "",            # keep field for payload schema compatibility
        "steps_text": steps_text,
    }


__all__ = ["detect_flowchart", "extract_flowchart_steps"]
