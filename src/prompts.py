"""
Centralized prompt builders for Gemini VLM.

Edit these helpers (or add new ones) to change how the VLM is instructed.
`vlm.py` imports and uses these so you don't need to touch the VLM client code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# You can freely edit these rules/templates:
DEFAULT_SYSTEM_RULES = """\
You are an image description assistant.
Follow these rules:
- Be concise and factual. If unsure, say you are unsure.
- Do not include hidden reasoning or chain-of-thought.
- Do not invent text in the image. If text is unreadable, say it.
- Respect the requested output format exactly.
"""

TEXT_TASK_TEMPLATE = """\
TASK:
{task}

CONSTRAINTS:
- Language: {lang}
- Output format: Plain text
- Max length guideline: {max_words} words
"""

JSON_TASK_TEMPLATE = """\
TASK:
{task}

CONSTRAINTS:
- Language: {lang}
- Output format: JSON
- Max length guideline: {max_words} words (hint; not enforced)
"""

# --- Public builders --------------------------------------------------------

def build_text_instruction(
    *,
    task: str,
    lang: str = "English",
    max_words: int = 120,
    system_rules: str = DEFAULT_SYSTEM_RULES,
) -> str:
    """Return the full instruction block for plain-text output."""
    return system_rules + "\n" + TEXT_TASK_TEMPLATE.format(
        task=task, lang=lang, max_words=max_words
    )


def build_json_instruction(
    *,
    task: str,
    lang: str = "English",
    max_words: int = 200,
    strict_json: bool = True,
    examples: Optional[List[Dict[str, Any]]] = None,
    system_rules: str = DEFAULT_SYSTEM_RULES,
) -> str:
    """Return the full instruction block for JSON output."""
    base = system_rules + "\n" + JSON_TASK_TEMPLATE.format(
        task=task, lang=lang, max_words=max_words
    )
    json_rule = (
        "Return ONLY valid JSON. Do not include backticks, code fences, or extra text."
        if strict_json
        else "Return JSON first; any comments after the JSON will be ignored."
    )
    example_block = ""
    if examples:
        # Use just the first example for brevity; extend as needed.
        from json import dumps
        example_block = "\nEXAMPLE JSON:\n" + dumps(examples[0], ensure_ascii=False, indent=2)

    return base + "\n" + json_rule + example_block


# --- Continuation builders (used when responses are truncated) --------------

def continue_text_instruction(
    *,
    base_instruction: str,
    tail: str,
) -> str:
    """Instruction to continue a truncated plain-text answer."""
    return (
        f"{base_instruction}\n\n"
        "Continue EXACTLY from where you stopped. Do not repeat earlier text. "
        "Continue from this tail:\n<<<\n" + tail + "\n>>>"
    )


def continue_json_instruction(
    *,
    base_instruction: str,
    tail: str,
) -> str:
    """Instruction to continue a truncated JSON answer."""
    return (
        base_instruction
        + "\nContinue the SAME JSON object EXACTLY from where you stopped "
          "(do not repeat earlier keys/values). "
          "Append only the missing part so that the final result is valid JSON.\n"
          "Tail context:\n<<<\n"
        + tail
        + "\n>>>"
    )


# --- Back-compat convenience (optional) ------------------------------------

def default_prompt() -> str:
    """Legacy convenience function kept for compatibility."""
    return build_text_instruction(task="Describe the image.")
