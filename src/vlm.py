"""
Visual language model wrapper using Google’s Gemini API.

This module provides a ``GeminiVLM`` class that wraps the Google
Generative AI SDK (``google.genai``) to describe images using a
language model.  It includes retry logic, streaming support and
auto-continuation to handle truncated responses.  Configuration is
controlled via the :class:`VLMConfig` dataclass, which reads defaults
from environment variables.  The implementation here is copied from
the upstream project and remains synchronous.

To use this class simply instantiate it with a configuration and call
:meth:`describe` with JPEG bytes and a prompt.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
import time
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type

# Attempt to import the Google generative AI SDK.  If unavailable
# fallback behaviours are provided.
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    try:
        from google.genai import errors as genai_errors  # type: ignore
    except Exception:
        genai_errors = None  # type: ignore
    _HAS_GOOGLE = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    genai_errors = None  # type: ignore
    _HAS_GOOGLE = False

# ✨ All prompt text now lives in prompts.py
from .prompts import (
    DEFAULT_SYSTEM_RULES,
    build_text_instruction,
    build_json_instruction,
    continue_text_instruction,
    continue_json_instruction,
)

# Optional import of Pydantic for structured JSON validation
try:
    from pydantic import BaseModel, ValidationError  # type: ignore
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False


@dataclass
class VLMConfig:
    """Configuration for :class:`GeminiVLM`.  Values are loaded from environment vars."""
    api_key: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
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
    retry_max: int = int(os.getenv("RETRY_MAX", "5"))
    retry_base: float = float(os.getenv("RETRY_BASE_SEC", "0.8"))
    timeout_ms: Optional[int] = int(os.getenv("GENAI_TIMEOUT_MS", "0")) or None


def _is_transient_error(e: Exception) -> bool:
    """Return True if exception is transient (safe to retry)."""
    if genai_errors and isinstance(e, genai_errors.APIError):
        code = getattr(e, "code", None)
        if code in (429, 500, 502, 503, 504):
            return True
    # Network-level hiccups/timeouts
    if isinstance(e, (TimeoutError, OSError)):
        return True
    return False


def _backoff_seconds(base: float, attempt: int, cap: float = 30.0) -> float:
    """Truncated exponential backoff with jitter."""
    return min(cap, base * (2 ** attempt)) * random.uniform(0.5, 1.5)


class GeminiVLM:
    """Client for the Google Generative AI vision model.

    When the Google SDK is unavailable, the class falls back to a dummy
    implementation that returns generic descriptions instead of
    querying an external service.
    """

    def __init__(self, cfg: Optional[VLMConfig] = None) -> None:
        self.cfg = cfg or VLMConfig()
        if _HAS_GOOGLE:
            if not self.cfg.api_key:
                raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is missing")
            # Optional client-level timeout via HttpOptions
            if self.cfg.timeout_ms:
                http_opts = types.HttpOptions(timeout=self.cfg.timeout_ms)
                self.client = genai.Client(api_key=self.cfg.api_key, http_options=http_opts)
            else:
                self.client = genai.Client(api_key=self.cfg.api_key)
        else:
            # Dummy client placeholder
            self.client = None  # type: ignore

    # Low-level one-shot generation (with optional streaming)
    def _gen_once(
        self,
        parts: list,
        *,
        response_mime_type: str | None = None,
        stream: bool = True,
    ) -> tuple[str, Optional[str]]:
        """Perform a single generation call, returning text and finish_reason.

        Uses the current google-genai surface:
          - client.models.generate_content(...)
          - client.models.generate_content_stream(...)
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
                gen_cfg.response_mime_type = response_mime_type  # e.g. "application/json"
            except Exception:
                pass

        if not stream:
            resp = self.client.models.generate_content(
                model=self.cfg.model,
                contents=parts,
                config=gen_cfg,
            )
            text = (getattr(resp, "text", "") or "").strip()
            fr = None
            try:
                cand0 = resp.candidates[0] if getattr(resp, "candidates", None) else None
                fr = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
            except Exception:
                pass
            return text, fr

        # Streaming
        out = ""
        finish = None
        stream_it = self.client.models.generate_content_stream(
            model=self.cfg.model,
            contents=parts,
            config=gen_cfg,
        )
        for event in stream_it:
            t = getattr(event, "text", None)
            if t:
                out += t
            try:
                cand0 = event.candidates[0] if getattr(event, "candidates", None) else None
                fr = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
                finish = finish or fr
            except Exception:
                pass
        return out, finish

    def _looks_incomplete_text(self, text: str) -> bool:
        """Heuristic: check if a text response looks incomplete."""
        return not text or len(text.strip()) < 3 or text.strip().endswith(",")

    def _looks_incomplete_json(self, text: str) -> bool:
        """Heuristic: check if a JSON response might be truncated."""
        txt = text.strip()
        return not txt or txt.count("{") > txt.count("}")

    def _continue_until_done(
        self,
        out: str,
        finish_reason: Optional[str],
        make_parts_for_continue,
        *,
        want_json: bool,
        max_continues: int,
    ) -> str:
        """Iteratively call the API until the output appears complete."""
        need_more = (
            (finish_reason == "MAX_TOKENS")
            or (self._looks_incomplete_json(out) if want_json else self._looks_incomplete_text(out))
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
        system_rules: str = DEFAULT_SYSTEM_RULES,
    ) -> str:
        """Generate a concise plain-text description; auto-continues if truncated.

        The instruction string is built via prompts.py and can be customized centrally.
        """
        instruction = build_text_instruction(
            task=prompt, lang=lang, max_words=max_words, system_rules=system_rules
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
                        continue_text_instruction(base_instruction=instruction, tail=tail),
                    ]
                full = self._continue_until_done(
                    text, finish, make_parts_for_continue, want_json=False, max_continues=4
                )
                return full.strip()
            except Exception as e:
                last_exc = e
                if _is_transient_error(e) and attempt < self.cfg.retry_max - 1:
                    time.sleep(_backoff_seconds(self.cfg.retry_base, attempt))
                    continue
                break
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
        system_rules: str = DEFAULT_SYSTEM_RULES,
    ) -> Dict[str, Any]:
        """Ask for JSON output; validate & auto-repair. Auto-continues when truncated.

        The instruction string is built via prompts.py and can be customized centrally.
        """
        want_schema = (schema_model is not None) and HAS_PYDANTIC
        base_instruction = build_json_instruction(
            task=prompt,
            lang=lang,
            max_words=max_words_hint,
            strict_json=strict_json,
            examples=examples,
            system_rules=system_rules,
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
                        continue_json_instruction(base_instruction=base_instruction, tail=tail),
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
                if _is_transient_error(e) and attempt < self.cfg.retry_max - 1:
                    # Add a hint to the instruction for the next round
                    base_instruction += f"\nVALIDATION ERROR:\n{e}\nFix and output ONLY valid JSON."
                    time.sleep(_backoff_seconds(self.cfg.retry_base, attempt))
                    continue
                break
        return {"_raw": "", "_error": str(last_exc) if last_exc else "unknown"}


__all__ = ["VLMConfig", "GeminiVLM"]
