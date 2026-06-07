"""
Translation pipeline for BFCL test cases.

LocalizationLevel controls which parts of a test case are translated:
- QUERY_ONLY  : Only the user-facing question is translated. Function names,
                parameter names, and descriptions stay in English. This tests
                whether a model can call English-defined APIs from a non-English query.
- FULL        : Both the query AND the function documentation (description,
                parameter descriptions) are translated. Function and parameter
                *names* are intentionally kept in English because real-world
                APIs rarely localize identifier names.
"""

from __future__ import annotations

import copy
import json
from enum import Enum
from typing import Any

import anthropic

from multilingual_bfcl.localization.locale_config import Locale

# Model used for translation. Claude is chosen because it supports the full
# set of target languages, produces high-quality translations, and is the
# same family used in many BFCL evaluations (reducing translation artifacts).
_TRANSLATION_MODEL = "claude-opus-4-8"

_SYSTEM_PROMPT = """\
You are a professional translator specializing in technical content for software \
developers. You will be given JSON content to translate into {language}. Follow \
these rules strictly:
1. Translate ONLY the values that are explicitly marked for translation.
2. Do NOT translate JSON keys, function names, parameter names, enum values, \
   code snippets, or anything inside backticks.
3. Preserve all JSON structure, whitespace, and escape sequences exactly.
4. Output ONLY valid JSON — no explanation, no markdown fences.
5. Preserve numerical values, units, and date formats according to locale: {hints}
"""


class LocalizationLevel(str, Enum):
    # Only the user-visible query is translated
    QUERY_ONLY = "query_only"
    # Query + function/parameter descriptions are translated
    FULL = "full"


def translate_test_case(
    test_case: dict[str, Any],
    locale: Locale,
    level: LocalizationLevel,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    """
    Return a new test case dict with the requested fields translated.

    The returned dict gains two extra fields:
      - locale: the BCP-47 code of the target language
      - localization_level: the level applied
      - source_id: original test case id (unchanged)
    The id is suffixed with the locale code.
    """
    if client is None:
        client = anthropic.Anthropic()

    result = copy.deepcopy(test_case)
    result["source_id"] = test_case["id"]
    result["id"] = f"{test_case['id']}_{locale.code}"
    result["locale"] = locale.code
    result["localization_level"] = level.value

    # Translate the conversation turns (user messages only)
    result["question"] = _translate_question(test_case["question"], locale, client)

    if level == LocalizationLevel.FULL:
        result["function"] = _translate_function_docs(test_case["function"], locale, client)

    return result


def _translate_question(
    question: list[list[dict[str, str]]],
    locale: Locale,
    client: anthropic.Anthropic,
) -> list[list[dict[str, str]]]:
    translated_turns = []
    for turn in question:
        translated_turn = []
        for msg in turn:
            if msg["role"] == "user":
                translated_turn.append({
                    "role": "user",
                    "content": _call_translation_api(msg["content"], locale, client),
                })
            else:
                translated_turn.append(msg)
        translated_turns.append(translated_turn)
    return translated_turns


def _translate_function_docs(
    functions: list[dict[str, Any]],
    locale: Locale,
    client: anthropic.Anthropic,
) -> list[dict[str, Any]]:
    """
    Translate description fields inside function docs.
    Deliberately preserves: name, type, required, enum, default.
    """
    translated = []
    for func in functions:
        func_copy = copy.deepcopy(func)
        if "description" in func_copy:
            func_copy["description"] = _call_translation_api(
                func_copy["description"], locale, client
            )
        if "parameters" in func_copy and "properties" in func_copy["parameters"]:
            for param_name, param_def in func_copy["parameters"]["properties"].items():
                if "description" in param_def:
                    param_def["description"] = _call_translation_api(
                        param_def["description"], locale, client
                    )
        translated.append(func_copy)
    return translated


def _call_translation_api(text: str, locale: Locale, client: anthropic.Anthropic) -> str:
    hints = "; ".join(locale.model_hints) if locale.model_hints else "none"
    system = _SYSTEM_PROMPT.format(language=locale.name, hints=hints)

    # Wrap in a simple JSON envelope so the model returns structured output
    # and we can extract the translated string cleanly.
    payload = json.dumps({"text": text}, ensure_ascii=False)

    response = client.messages.create(
        model=_TRANSLATION_MODEL,
        max_tokens=4096,
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Translate the value of the 'text' key into {locale.name}:\n{payload}",
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)
    return parsed["text"]
