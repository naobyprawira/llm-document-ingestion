"""
Flowchart detection and Vision Language Model (VLM) integration.

Detects flowcharts on PDF pages via PyMuPDF vector primitives.  If likely,
renders the page to PNG and queries Gemini to linearise steps as JSON.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
from typing import Any, Dict, Optional

import fitz  # type: ignore

# Prefer the new unified SDK; fall back to legacy if unavailable.
try:
    from google import genai as genai_unified  # google-genai
    from google.genai import types as unified_types
    _USE_UNIFIED = True
except Exception:  # pragma: no cover
    genai_unified = None
    unified_types = None
    _USE_UNIFIED = False

try:
    import google.generativeai as genai_legacy  # google-generativeai
    from google.generativeai.types import GenerationConfig as LegacyGenerationConfig  # type: ignore
except Exception:  # pragma: no cover
    genai_legacy = None
    LegacyGenerationConfig = None

from ..config import AppConfig
from ..parsers.pdf_text import render_page_to_png

logger = logging.getLogger(__name__)

_UNIFIED_MODEL = "gemini-2.5-flash"  # unified SDK naming
_LEGACY_MODEL = "gemini-1.5-flash"   # legacy SDK naming


def _count_vector_features(page: fitz.Page) -> Dict[str, int]:
    rects = ellipses = diamonds = lines = ortho = 0
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            itype = item[0]
            if itype == "re":
                rects += 1
            elif itype == "el":
                ellipses += 1
            elif itype == "l":
                _, p0, p1 = item
                lines += 1
                angle = abs(math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x))) % 90
                if angle < 6 or angle > 84:
                    ortho += 1
            elif itype in ("p", "qu", "c"):
                coords = []
                for segment in item[1:]:
                    if hasattr(segment, "x"):
                        coords.append((segment.x, segment.y))
                    elif isinstance(segment, tuple):
                        coords.extend([(segment[0].x, segment[0].y), (segment[1].x, segment[1].y)])
                if len(coords) == 4:
                    sides = []
                    for i in range(4):
                        x0, y0 = coords[i]
                        x1, y1 = coords[(i + 1) % 4]
                        sides.append(math.hypot(x1 - x0, y1 - y0))
                    mean_side = sum(sides) / 4.0
                    if mean_side > 0 and all(0.8 * mean_side <= s <= 1.2 * mean_side for s in sides):
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        ang = abs(math.degrees(math.atan2(y1 - y0, x1 - x0))) % 90
                        if 20 < ang < 70:
                            diamonds += 1
    return {"rects": rects, "ellipses": ellipses, "diamonds": diamonds, "lines": lines, "ortho": ortho}


def detect_flowchart(page: fitz.Page) -> bool:
    counts = _count_vector_features(page)
    rects = counts["rects"]
    diamonds = counts["diamonds"]
    ellipses = counts["ellipses"]
    lines = counts["lines"]
    ortho = counts["ortho"]
    arrow_proxy = 1 if (lines >= 6 and lines > 0 and (ortho / max(lines, 1)) > 0.65) else 0
    score = rects + 2 * diamonds + ellipses + arrow_proxy
    return score >= 3


def _extract_json_object(text: str) -> Optional[dict]:
    """Robustly extract a JSON object from free-form model text."""
    if not text:
        return None
    # Strip code fences
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        # remove language hint like ```json
        if "\n" in t:
            t = t.split("\n", 1)[1]
    # Find first/last braces
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(t[start : end + 1])
    except Exception:
        return None


def extract_flowchart_steps(
    doc: fitz.Document,
    page_index: int,
    config: AppConfig,
) -> Optional[Dict[str, Any]]:
    # Resolve API key from config or env.
    api_key = (
        getattr(config, "google_api_key", None)
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )
    if not api_key:
        logger.warning("VLM API key missing; skipping flowchart extraction.")
        return None

    if not (_USE_UNIFIED or genai_legacy):
        logger.warning("Gemini SDK not installed; skipping flowchart extraction.")
        return None

    # Render page
    png_bytes = render_page_to_png(doc, page_index, getattr(config, "image_dpi", 220))

    # Instruction prompt
    prompt = (
        "Return JSON ONLY with keys: has_flowchart (bool), title (string), "
        "steps_text (string), confidence (0..1).\n"
        "Rules:\n"
        "- If image has no flowchart → {\"has_flowchart\": false, \"title\":\"\", \"steps_text\":\"\", \"confidence\":0}.\n"
        "- steps_text must be numbered lines:\n"
        "  1) ...\n  2) ...\n"
        "  For decisions use branch bullets:\n"
        "    - If Yes → ...\n    - If No  → ...\n"
        "- Be faithful; if unreadable, use empty string for that fragment. No prose outside JSON."
    )

    try:
        if _USE_UNIFIED:
            # Unified SDK: use Part.from_bytes and JSON response MIME type.
            client = genai_unified.Client(api_key=api_key)
            img_part = unified_types.Part.from_bytes(data=png_bytes, mime_type="image/png")
            cfg = unified_types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.1, max_output_tokens=1024
            )
            resp = client.models.generate_content(
                model=_UNIFIED_MODEL,
                contents=[prompt, img_part],
                generation_config=cfg,
            )
            text = resp.text if hasattr(resp, "text") else str(resp)
        else:
            # Legacy SDK: use embed_content with image
            generation_config = LegacyGenerationConfig(response_mime_type="application/json", temperature=0.1)
            resp = genai_legacy.embed_content(
                model=_LEGACY_MODEL,
                content={"image": {"mime_type": "image/png", "data": base64.b64encode(png_bytes).decode()}},
                prompt=prompt,
                generation_config=generation_config,
            )
            text = resp["text"] if isinstance(resp, dict) else str(resp)
    except Exception as e:
        logger.error("Gemini VLM call failed: %s", e)
        return None

    data = _extract_json_object(text)
    if not data:
        return None
    # Basic validation
    has_flow = bool(data.get("has_flowchart"))
    steps_text = data.get("steps_text", "")
    if not has_flow or not steps_text.strip():
        return None
    return data
