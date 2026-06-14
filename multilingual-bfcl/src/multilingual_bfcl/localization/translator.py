"""
Translation transforms for BFCL test cases — batch-API ready.

This module no longer makes its own (synchronous) API calls. Instead it provides
the pure pieces the batch driver needs:

  - LocalizationLevel       : which parts of a test case get translated
  - build_input()           : one batch item = one (test_case, locale) — packs the
                              query messages and (for FULL) the textual ground-truth
                              parameter values into a single JSON payload
  - make_chain()/build_prompt(): the LangChain chain langasync wraps
  - parse_translation()     : parse the model's JSON reply
  - apply_translation()     : reassemble a translated test_case + possible_answer entry

LocalizationLevel controls scope:
  - QUERY : Only the user-facing query is translated. Ground-truth parameter
            values are left unchanged. Function definitions stay in English.
  - FULL  : The query AND the *textual* ground-truth parameter values are
            translated. Function names, parameter names, descriptions, enum
            values, numbers, dates, identifiers, etc. all stay unchanged — only
            natural-language argument values are localized.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from enum import Enum
from typing import Any

from multilingual_bfcl.localization.locale_config import Locale

# Default model for translation. Claude supports the full set of target
# languages and produces high-quality, low-artifact translations.
DEFAULT_TRANSLATION_MODEL = "claude-opus-4-8"


class LocalizationLevel(str, Enum):
    # Only the user-visible query is translated (ground truth untouched).
    QUERY = "query"
    # Query + textual ground-truth parameter values are translated.
    FULL = "full"


# ---------------------------------------------------------------------------
# Local "is this worth translating?" skip — mirrors classify_benchmark.py.
# Strings that are clearly language-invariant never reach the model: numbers,
# booleans, dates/times, and empty strings.
# ---------------------------------------------------------------------------
_NON_TRANSLATABLE_STRING_RE = re.compile(
    r"""^\s*(
        true|false
        |[-+]?\d+(\.\d+)?([eE][-+]?\d+)?                              # number
        |\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?  # ISO-ish date(/time)
        |\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?  # day-first/US date(/time)
        |\d{1,2}:\d{2}(:\d{2})?                                        # time
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_translatable_string(value: Any) -> bool:
    """True only for a non-empty string that is not a number/bool/date/time.

    The model still gets the final say (it is told to leave identifiers, enums,
    code, units, etc. unchanged); this just avoids sending obviously invariant
    scalars and keeps the payload small.
    """
    return (
        isinstance(value, str)
        and value.strip() != ""
        and _NON_TRANSLATABLE_STRING_RE.match(value) is None
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a professional translator localizing function-calling benchmark data \
into {language}. Follow these rules strictly:
1. Translate natural-language text faithfully and idiomatically into {language}.
2. Do NOT translate or alter: function names, parameter names, JSON keys, enum \
tokens, programming identifiers, code, mathematical expressions, file paths, \
URLs, units, numbers, booleans, or dates. Return such values exactly as given.
3. Do not add, remove, or reorder information.
4. Locale conventions for numbers/dates: {hints}
5. Output ONLY a JSON object — no explanation, no markdown fences."""

# Two slots are filled per-level with str.format(): {values_instruction} and
# {output_json}. The LangChain runtime variables are written doubled ({{language}},
# {{payload}}) so they survive that .format() pass as single-brace placeholders,
# and the literal JSON braces shown to the model come from the {output_json} value
# (also doubled, so LangChain renders them as a literal { } at invoke time).
HUMAN_TEMPLATE = """\
Translate the JSON below into {{language}}.

- "queries": user messages — translate each one fully into {{language}}.
{values_instruction}Return ONLY a JSON object with the SAME keys and the SAME number \
of items in each list, in the same order:
{output_json}

INPUT:
{{payload}}"""

_VALUES_INSTRUCTION = (
    '- "values": ground-truth argument values — translate ONLY those that are '
    "natural language (names, places, search terms, descriptions). Return any "
    "identifier, enum, code, number, or unit value unchanged.\n"
)

# Expected output object per level (doubled braces -> literal braces in the prompt).
_OUTPUT_JSON = {
    LocalizationLevel.QUERY: '{{"queries": [...]}}',
    LocalizationLevel.FULL: '{{"queries": [...], "values": [...]}}',
}


def build_human_template(level: LocalizationLevel) -> str:
    """Format HUMAN_TEMPLATE for a level.

    The "queries" line is always present; the "values" instruction line is only
    inserted for FULL, and the expected output JSON object is adjusted to match
    (queries-only vs queries+values).
    """
    return HUMAN_TEMPLATE.format(
        values_instruction=_VALUES_INSTRUCTION if level == LocalizationLevel.FULL else "",
        output_json=_OUTPUT_JSON[level],
    )


def build_prompt(level: LocalizationLevel):
    """The ChatPromptTemplate langasync wraps (variables: language, hints, payload)."""
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", build_human_template(level)),
    ])


def make_chain(provider: str, model_name: str, level: LocalizationLevel):
    """Build prompt | model | str. Switching providers is just this factory."""
    from langchain_core.output_parsers import StrOutputParser

    prompt = build_prompt(level)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=model_name, max_tokens=4096)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=model_name, temperature=0, max_tokens=4096)
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Choose 'anthropic' or 'openai'.")

    return prompt | model | StrOutputParser()


# ---------------------------------------------------------------------------
# Extracting the units to translate (deterministic order — submit & retrieve
# recompute the same lists, so results map back positionally).
# ---------------------------------------------------------------------------

def collect_query_texts(question: list[list[dict[str, str]]]) -> list[str]:
    """Ordered list of user-message contents across all turns."""
    return [
        msg["content"]
        for turn in question
        for msg in turn
        if msg.get("role") == "user"
    ]


def collect_textual_values(ground_truth: list[dict]) -> list[str]:
    """Ordered list of ground-truth argument values worth translating.

    Ground-truth format: [{func_name: {param: [acceptable_value, ...]}}].
    Only non-empty natural-language strings are collected; the order here is the
    exact order in which apply_translation() puts the translations back.
    """
    values: list[str] = []
    for call in ground_truth:
        for params in call.values():
            for value_list in params.values():
                for v in value_list:
                    if is_translatable_string(v):
                        values.append(v)
    return values


def build_input(
    test_case: dict[str, Any],
    answer: dict[str, Any] | None,
    locale: Locale,
    level: LocalizationLevel,
) -> dict[str, str]:
    """Build the chain-input dict for one (test_case, locale) batch item.

    The payload mirrors the level: QUERY sends only {"queries": [...]}, while FULL
    also sends {"values": [...]} with the textual ground-truth values.
    """
    queries = collect_query_texts(test_case.get("question", []))
    payload_obj: dict[str, list] = {"queries": queries}
    if level == LocalizationLevel.FULL:
        payload_obj["values"] = (
            collect_textual_values(answer.get("ground_truth", [])) if answer is not None else []
        )

    payload = json.dumps(payload_obj, ensure_ascii=False)
    hints = "; ".join(locale.model_hints) if locale.model_hints else "none"
    return {"language": locale.name, "hints": hints, "payload": payload}


# ---------------------------------------------------------------------------
# Parsing the model reply
# ---------------------------------------------------------------------------

def parse_translation(raw: str, entry_id: str) -> dict[str, list] | None:
    """Parse the model's JSON reply into {"queries": [...], "values": [...]}.

    Returns None (and warns) if the JSON cannot be parsed or is malformed.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start:end])
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse JSON for {entry_id}: {raw!r}", file=sys.stderr)
                return None
        else:
            print(f"[WARN] No JSON found for {entry_id}: {raw!r}", file=sys.stderr)
            return None

    queries = obj.get("queries")
    values = obj.get("values", [])
    if not isinstance(queries, list) or not isinstance(values, list):
        print(f"[WARN] {entry_id}: reply missing list 'queries'/'values': {obj!r}", file=sys.stderr)
        return None
    return {"queries": queries, "values": values}


# ---------------------------------------------------------------------------
# Reassembling translated test case + possible-answer entry
# ---------------------------------------------------------------------------

def _suffixed_id(base_id: str, locale: Locale) -> str:
    return f"{base_id}_{locale.code}"


def apply_translation(
    test_case: dict[str, Any],
    answer: dict[str, Any] | None,
    locale: Locale,
    level: LocalizationLevel,
    parsed: dict[str, list],
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """Build the translated (test_case_entry, possible_answer_entry).

    Returns None if the reply's item counts don't match what we sent (the caller
    should treat that as a failed entry). possible_answer_entry is None when no
    answer was provided.
    """
    queries = parsed["queries"]
    values = parsed["values"]

    expected_queries = collect_query_texts(test_case.get("question", []))
    if len(queries) != len(expected_queries):
        print(
            f"[WARN] {_suffixed_id(test_case['id'], locale)}: expected "
            f"{len(expected_queries)} query string(s), got {len(queries)}.",
            file=sys.stderr,
        )
        return None

    # --- translated question (function defs left untouched) ---
    new_question = copy.deepcopy(test_case.get("question", []))
    qi = iter(queries)
    for turn in new_question:
        for msg in turn:
            if msg.get("role") == "user":
                msg["content"] = next(qi)

    q_entry = copy.deepcopy(test_case)
    q_entry["question"] = new_question
    q_entry["source_id"] = test_case["id"]
    q_entry["id"] = _suffixed_id(test_case["id"], locale)
    q_entry["locale"] = locale.code
    q_entry["localization_level"] = level.value

    # --- translated possible-answer entry ---
    a_entry: dict[str, Any] | None = None
    if answer is not None:
        a_entry = copy.deepcopy(answer)
        a_entry["id"] = _suffixed_id(answer["id"], locale)

        if level == LocalizationLevel.FULL:
            expected_values = collect_textual_values(answer.get("ground_truth", []))
            if len(values) != len(expected_values):
                print(
                    f"[WARN] {a_entry['id']}: expected {len(expected_values)} "
                    f"value(s), got {len(values)}.",
                    file=sys.stderr,
                )
                return None
            vi = iter(values)
            for call in a_entry.get("ground_truth", []):
                for params in call.values():
                    for key, value_list in params.items():
                        params[key] = [
                            next(vi) if is_translatable_string(v) else v
                            for v in value_list
                        ]

    return q_entry, a_entry
