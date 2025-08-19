# app/vision/flowchart_extractor.py
"""
Flowchart detection and VLM extraction.

- Detects likely-flowchart pages using vector primitives from PyMuPDF.
- When detected (or forced), renders the page to PNG and asks Gemini VLM
  to return STRICT JSON with ordered steps.

Public API:
- detect_flowchart(doc, page_index) -> bool
- extract_flowchart_steps(doc, page_index, config) -> dict | None
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai  # pip install google-generativeai

from ..config import AppConfig
from ..parsers.pdf_text import render_page_to_png

log = logging.getLogger(__name__)

# Single instruction kept short to reduce hallucinations.
FLOWCHART_JSON_PROMPT = (
    "You are given a PNG of a flowchart. Return STRICT JSON only:\n"
    '{ "has_flowchart": true|false, "title": "string",'
    '  "steps": [ {"n": 1, "text": "..."}, {"n": 2, "text": "..."} ] }\n'
    'If no flowchart is present, return {"has_flowchart": false} only.'
)

# Heuristics for detection using vector shapes.
_DRAW_MIN_BOXES = 4
_DRAW_MIN_LINES = 6


def _count_shapes(page) -> Tuple[int, int]:
    """
    Count rectangles and line segments using PyMuPDF drawing objects.
    """
    rects = 0
    lines = 0
    try:
        for item in page.get_drawings():
            # item contains 'rect', 'items' etc.
            if item.get("rect"):
                rects += 1
            for it in item.get("items", ()):
                # ('l', p1, p2) is a line segment
                if it and it[0] == "l":
                    lines += 1
    except Exception as e:  # pragma: no cover
        log.debug("get_drawings failed: %s", e)
    return rects, lines


def detect_flowchart(doc, page_index: int) -> bool:
    """
    Lightweight detection: enough boxes and lines suggests a flowchart.
    """
    page = doc.load_page(page_index)
    boxes, segs = _count_shapes(page)
    return boxes >= _DRAW_MIN_BOXES and segs >= _DRAW_MIN_LINES


def extract_flowchart_steps(
    doc,
    page_index: int,
    config: AppConfig,
    *,
    force: bool = False,
    timeout_s: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    If page looks like a flowchart (or force=True), call Gemini VLM and
    return a dict:
      {
        "has_flowchart": bool,
        "title": str,
        "steps_text": "1. ...\\n2. ...",
      }
    Returns None on transport/parse errors.
    """
    if not force and not detect_flowchart(doc, page_index):
        return {"has_flowchart": False}

    # Render once to bytes
    image_bytes = render_page_to_png(doc, page_index, dpi=config.image_dpi)

    # Configure and call VLM
    genai.configure(api_key=config.google_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        resp = model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},
                FLOWCHART_JSON_PROMPT,
            ],
            request_options={"timeout": timeout_s},
        )
        text = (resp.text or "").strip()
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("VLM returned non-JSON on page %d", page_index + 1)
        return None
    except Exception as e:
        log.warning("Gemini VLM call failed: %s", e)
        return None

    if not isinstance(data, dict) or not data.get("has_flowchart"):
        return {"has_flowchart": False}

    steps = data.get("steps") or []
    steps_text = "\n".join(
        f"{int(s.get('n', i+1))}. {str(s.get('text','')).strip()}"
        for i, s in enumerate(steps)
        if isinstance(s, dict)
    )

    return {
        "has_flowchart": True,
        "title": str(data.get("title") or "").strip(),
        "steps_text": steps_text,
    }


__all__ = ["detect_flowchart", "extract_flowchart_steps"]
