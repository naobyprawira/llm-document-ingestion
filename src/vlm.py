#!/usr/bin/env python3
"""
Visual language model wrapper using Google’s Gemini API.

This module provides a ``GeminiVLM`` class that wraps the Google
Generative AI SDK (``google.genai``) to describe images using a
language model. It includes retry logic, streaming support and
auto‑continuation to handle truncated responses. Configuration is
controlled via the ``VLMConfig`` dataclass, which reads defaults
from environment variables. The implementation below matches the
upstream code from the reference repository with minor reformatting
for readability.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    _HAS_GOOGLE = True
except Exception:
    # When the Google Generative AI SDK is unavailable, stub out the imports
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GOOGLE = False
from .prompts import default_prompt

# Optional import of Pydantic for structured JSON validation
try:
    from pydantic import BaseModel, ValidationError  # type: ignore
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False


@dataclass
class VLMConfig:
    """Configuration for ``GeminiVLM``.

    All values can be overridden via environment variables. See the
    ``vlm.py`` in the upstream repository for further discussion of
    these parameters.
    """
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "1512"))
    temperature: float = float(os.getenv("VLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("VLM_TOP_P", "0.9"))
    top_k: int = int(os.getenv("VLM_TOP_K", "40"))
    candidate_count: int = int(os.getenv("VLM_CANDIDATE_COUNT", "1"))
    stop_sequences: List[str] = field(
        default_factory=lambda: (
            os.getenv("VLM_STOP", "").split("|") if os.getenv("VLM_STOP") else []
        )
    )
    retry_max: int = int(os.getenv("RETRY_MAX", "3"))
    retry_base: float = float(os.getenv("RETRY_BASE_SEC", "0.8"))


# System rules and task template used to frame prompts to the model
_SYSTEM_RULES = """\
You are an image description assistant.
Follow these rules:
- Be concise and factual. If unsure, say you are unsure.
- Do not include hidden reasoning or chain‑of‑thought.
- Do not invent text in the image. If text is unreadable, say it.
- Respect the requested output format exactly.
"""

_TASK_TEMPLATE = """\
TASK:
{task}

CONSTRAINTS:
- Language: {lang}
- Output format: {fmt}
- Max length guideline: {max_words} words (not enforced if JSON).
If you are uncertain, set fields to null or empty rather than guessing.
"""


class GeminiVLM:
    """Client for the Google Generative AI vision model.

    When the Google SDK is unavailable, the class falls back to a dummy
    implementation that returns generic descriptions instead of
    querying an external service. This behaviour allows the wider API
    to function in constrained environments (e.g. during tests).
    """

    def __init__(self, cfg: Optional[VLMConfig] = None) -> None:
        self.cfg = cfg or VLMConfig()
        if _HAS_GOOGLE:
            if not self.cfg.api_key:
                raise ValueError("GOOGLE_API_KEY is missing")
            self.client = genai.Client(api_key=self.cfg.api_key)
        else:
            # Dummy client placeholder; not used but kept for symmetry
            self.client = None  # type: ignore

    # Low‑level one‑shot generation (with optional streaming)
    def _gen_once(
        self,
        parts: list,
        *,
        response_mime_type: str | None = None,
        stream: bool = True,
    ) -> tuple[str, Optional[str]]:
        """Perform a single generation call, returning text and finish_reason.

        In stub mode (when the Google SDK is missing) this returns a
        dummy description and ``None`` for the finish reason.
        """
        if not _HAS_GOOGLE:
            # Return a generic message; the actual content of ``parts`` is ignored
            return "[vision service unavailable]", None
        gen_cfg = types.GenerateContentConfig(
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            candidate_count=self.cfg.candidate_count,
            stop_sequences=self.cfg.stop_sequences if self.cfg.stop_sequences else None,
        )
        if response_mime_type:
            try:
                gen_cfg.response_mime_type = response_mime_type  # type: ignore[attr-defined]
            except Exception:
                pass

        def _extract_nonstream(resp) -> tuple[str, Optional[str]]:
            text = (getattr(resp, "text", "") or "").strip()
            fr = None
            try:
                cand0 = resp.candidates[0] if getattr(resp, "candidates", None) else None
                fr = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
            except Exception:
                pass
            return text, fr

        # Try streaming first, then gracefully fall back
        if stream:
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model,
                    contents=parts,
                    config=gen_cfg,
                    stream=True,  # type: ignore[arg-type]
                )
                text, finish_reason = "", None
                for ev in resp:
                    if hasattr(ev, "text") and ev.text:
                        text += ev.text
                    if getattr(ev, "candidates", None):
                        try:
                            cand0 = ev.candidates[0]
                            fr = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
                            if fr:
                                finish_reason = fr
                        except Exception:
                            pass
                return text.strip(), finish_reason
            except TypeError:
                pass
            except AttributeError:
                pass
            except Exception:
                pass

        # Non‑streaming path
        resp = self.client.models.generate_content(
            model=self.cfg.model, contents=parts, config=gen_cfg
        )
        return _extract_nonstream(resp)

    # Heuristics for detecting incomplete text/JSON
    @staticmethod
    def _looks_incomplete_text(s: str) -> bool:
        if not s:
            return True
        trimmed = s.rstrip()
        return (len(trimmed) > 120) and (trimmed[-1] not in ".!?」”’\"]`}")

    @staticmethod
    def _looks_incomplete_json(s: str) -> bool:
        try:
            json.loads(s)
            return False
        except Exception:
            return s.count("{") != s.count("}")

    # Auto‑continuation loop
    def _continue_until_done(
        self,
        first_text: str,
        first_finish_reason: Optional[str],
        make_parts_for_continue,  # callable(prev_tail: str) -> List[Any]
        *,
        want_json: bool = False,
        max_continues: int = 4,
    ) -> str:
        out = first_text or ""
        if want_json:
            need_more = (
                (first_finish_reason == "MAX_TOKENS") or self._looks_incomplete_json(out)
            )
        else:
            need_more = (
                (first_finish_reason == "MAX_TOKENS") or self._looks_incomplete_text(out)
            )
        rounds = 0
        while need_more and rounds < max_continues:
            rounds += 1
            tail = out[-300:] if len(out) > 300 else out
            parts = make_parts_for_continue(tail)
            mime = "application/json" if want_json else None
            more, fr = self._gen_once(parts, response_mime_type=mime, stream=True)
            # Guard against repetition at the concat boundary
            if more and tail and more.startswith(tail[-50:]):
                more = more[len(tail[-50:]):].lstrip()
            out += ("\n" if out and more and not want_json else "") + (more or "")
            if want_json:
                need_more = (fr == "MAX_TOKENS") or self._looks_incomplete_json(out)
            else:
                need_more = (fr == "MAX_TOKENS") or self._looks_incomplete_text(out)
        return out

    # Public: plain text describe
    def describe(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        lang: str = "English",
        max_words: int = 120,
    ) -> str:
        """Generate a concise plain‑text description; auto‑continues if truncated."""
        instruction = _SYSTEM_RULES + "\n" + _TASK_TEMPLATE.format(
            task=prompt, lang=lang, fmt="Plain text", max_words=max_words
        )
        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.retry_max):
            try:
                parts = [
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    instruction,
                ]
                text, finish = self._gen_once(parts, response_mime_type=None, stream=True)

                def make_parts_for_continue(tail: str) -> List[Any]:
                    return [
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        (
                            f"{instruction}\n\n"
                            "Continue EXACTLY from where you stopped. Do not repeat earlier text. "
                            "Continue from this tail:\n<<<\n" + tail + "\n>>>"
                        ),
                    ]

                full = self._continue_until_done(
                    text, finish, make_parts_for_continue, want_json=False, max_continues=4
                )
                return full.strip()
            except Exception as e:
                last_exc = e
                time.sleep(self.cfg.retry_base * (2 ** attempt))
        return f"[VLM-ERROR] {last_exc}"

    # Public: structured JSON describe
    def describe_json(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        schema_model: Optional[Type[Any]] = None,
        lang: str = "English",
        examples: Optional[List[Dict[str, Any]]] = None,
        strict_json: bool = True,
        max_words_hint: int = 200,
    ) -> Dict[str, Any]:
        """Ask for JSON output; validate & auto‑repair. Auto‑continues when truncated."""
        want_schema = (schema_model is not None) and HAS_PYDANTIC
        json_rule = (
            "Return ONLY valid JSON. Do not include backticks, code fences, or extra text."
            if strict_json
            else "Return JSON first; any comments after the JSON will be ignored."
        )
        example_block = ""
        if examples:
            try:
                example_block = "\nEXAMPLE JSON:\n" + json.dumps(examples[0], ensure_ascii=False, indent=2)
            except Exception:
                pass
        base_instruction = (
            _SYSTEM_RULES
            + "\n"
            + _TASK_TEMPLATE.format(task=prompt, lang=lang, fmt="JSON", max_words=max_words_hint)
            + f"\n{json_rule}"
            + example_block
        )
        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.retry_max):
            try:
                parts = [
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    base_instruction,
                ]
                txt, finish = self._gen_once(parts, response_mime_type="application/json", stream=True)

                def make_parts_for_continue(tail: str) -> List[Any]:
                    return [
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        (
                            base_instruction
                            + "\nContinue the SAME JSON object EXACTLY from where you stopped "
                              "(do not repeat earlier keys/values). "
                              "Append only the missing part so that the final result is valid JSON.\n"
                              "Tail context:\n<<<\n"
                            + tail
                            + "\n>>>"
                        ),
                    ]
                txt = self._continue_until_done(
                    txt, finish, make_parts_for_continue, want_json=True, max_continues=5
                )
                cleaned = txt.strip()
                if strict_json and cleaned.startswith("```"):
                    first_nl = cleaned.find("\n")
                    if first_nl != -1:
                        cleaned = cleaned[first_nl + 1 :]
                if strict_json and cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                data = json.loads(cleaned)
                if want_schema:
                    try:
                        validated = schema_model.model_validate(data)  # type: ignore[attr-defined]
                        return json.loads(validated.model_dump_json())  # type: ignore[attr-defined]
                    except AttributeError:
                        validated = schema_model.parse_obj(data)  # type: ignore[attr-defined]
                        return validated.dict()  # type: ignore[attr-defined]
                return data
            except Exception as e:
                last_exc = e
                base_instruction += f"\nVALIDATION ERROR:\n{e}\nFix and output ONLY valid JSON."
                time.sleep(self.cfg.retry_base * (2 ** attempt))
        # Last resort: surface raw error
        return {"_raw": "", "_error": str(last_exc) if last_exc else "unknown"}


if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke test (requires GOOGLE_API_KEY and a valid JPEG bytes)
    import sys
    print("vlm.py loaded; instantiate GeminiVLM() and call describe()/describe_json().")
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "rb") as f:
            img = f.read()
        vlm = GeminiVLM()
        prompt = default_prompt()
        print(vlm.describe(img, prompt, lang="Indonesian"))