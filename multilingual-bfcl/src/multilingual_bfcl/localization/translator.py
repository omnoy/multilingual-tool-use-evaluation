"""
Translation transforms for BFCL test cases — batch-API ready.

This module no longer makes its own (synchronous) API calls. Instead it provides
the pure pieces the batch driver needs:

  - LocalizationLevel       : which parts of a test case get translated
  - build_input()           : one batch item = one (test_case, locale) — packs the
                              query messages and (for FULL) the textual ground-truth
                              parameter values into a single JSON payload
  - build_prompt()/render_messages(): the prompt template and the OpenAI-style
                              message list handed to the Azure Batch API
  - parse_translation()     : parse the model's JSON reply
  - apply_translation()     : reassemble a translated test_case (with inline ground_truth)

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
1. Translate ALL natural-language text into {language} faithfully and idiomatically. \
This INCLUDES proper nouns — names of people, places, countries, organizations, \
brands, teams, events, months, etc. Use the established {language} form when one \
exists; otherwise transliterate the word phonetically into the {language} script. \
Do NOT leave such words in English.
2. Leave UNCHANGED only genuinely non-linguistic tokens: JSON keys, function names, \
parameter names, programming identifiers and variable names, code, mathematical \
expressions, file paths, URLs, and pure numbers/booleans/dates.
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

# Note on braces: this string is inserted into HUMAN_TEMPLATE as a .format() VALUE,
# so its braces are NOT processed at that pass. Write {language} single (a LangChain
# runtime variable filled later) and any literal JSON braces doubled ({{ }}).
_VALUES_INSTRUCTION = (
    '- "values": ground-truth argument values. Each item is an object '
    '{{"param": <name>, "description": <what the parameter means>, "value": <the value>}} '
    "given only for context. Every item was pre-screened as a natural-language value, "
    'so translate its "value" into {language} — translate or transliterate proper nouns '
    "exactly as in rule 1. Keep a value unchanged ONLY if it is genuinely a code token, "
    "programming identifier, file path, or URL. Output \"values\" as a flat list of the "
    "translated value strings, one per input item, in the same order.\n"
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
    """The ChatPromptTemplate for translation (variables: language, hints, payload)."""
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", build_human_template(level)),
    ])


def render_messages(level: LocalizationLevel, variables: dict[str, str]) -> list[dict[str, str]]:
    """Render the prompt for one batch item into OpenAI-style chat messages.

    `variables` fills the template slots (language, hints, payload). The result is a
    list of {"role": ..., "content": ...} dicts ready to drop into a chat.completions
    request body (used for the Azure Batch API).
    """
    from langchain_core.messages import SystemMessage

    messages = build_prompt(level).format_messages(**variables)
    return [
        {"role": "system" if isinstance(m, SystemMessage) else "user", "content": m.content}
        for m in messages
    ]


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


def _param_descriptions(functions: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Map (function_name, param_name) -> parameter description for context lookup."""
    out: dict[tuple[str, str], str] = {}
    for func in functions:
        name = func.get("name", "")
        props = func.get("parameters", {}).get("properties", {})
        for param, pdef in props.items():
            out[(name, param)] = pdef.get("description", "")
    return out


def collect_textual_value_contexts(
    test_case: dict[str, Any], ground_truth: list[dict]
) -> list[dict[str, str]]:
    """Like collect_textual_values, but each value carries its parameter context.

    Returns objects {"param", "description", "value"} in the SAME order as
    collect_textual_values, so apply_translation can still map results back
    positionally. The description is looked up from the test case's function defs.
    """
    descs = _param_descriptions(test_case.get("function", []))
    items: list[dict[str, str]] = []
    for call in ground_truth:
        for func_name, params in call.items():
            for param, value_list in params.items():
                for v in value_list:
                    if is_translatable_string(v):
                        items.append({
                            "param": param,
                            "description": descs.get((func_name, param), ""),
                            "value": v,
                        })
    return items


# ---------------------------------------------------------------------------
# Language descriptors — annotate natural-language parameters with the set of
# languages their value may be given in (e.g. "Languages: English, Hebrew.").
#
# Detection combines two signals so that distractor functions — which are never
# referenced by the ground truth — are covered as well as the correct one:
#   - Ground truth refines: a parameter present in the entry's ground truth is
#     natural-language iff at least one of its GT values passes is_translatable_string
#     (the "basic regex" — not a number/bool/date). This correctly skips a
#     string-typed param whose actual value is really a number/date/code.
#   - Schema fallback: a parameter NOT in the ground truth (i.e. living only in a
#     distractor function) is natural-language iff its schema type is a string, or
#     an array/list of strings.
# Enums (including array-item enums) are never tagged.
# ---------------------------------------------------------------------------

def _has_enum(pdef: dict[str, Any]) -> bool:
    """True if the parameter (or, for arrays, its items) is constrained to an enum."""
    if "enum" in pdef:
        return True
    items = pdef.get("items")
    return isinstance(items, dict) and "enum" in items


def _is_textual_schema(pdef: dict[str, Any]) -> bool:
    """True if the schema type is a string, or an array/list whose items are strings."""
    ptype = pdef.get("type")
    if ptype == "string":
        return True
    if ptype in ("array", "list"):
        items = pdef.get("items")
        return isinstance(items, dict) and items.get("type") == "string"
    return False


def is_natural_language_param(
    pdef: dict[str, Any],
    gt_values: list | None,
) -> bool:
    """Whether a parameter should carry a language descriptor.

    `gt_values` is the parameter's ground-truth value list, or None when the
    parameter does not appear in the ground truth (a distractor param). Enums are
    always rejected; otherwise ground truth decides when present, and the schema
    type is the fallback when it is absent.
    """
    if _has_enum(pdef):
        return False
    if gt_values is not None:
        return any(is_translatable_string(v) for v in gt_values)
    return _is_textual_schema(pdef)


def add_language_descriptors_to_entry(
    entry: dict[str, Any],
    languages: str,
    template: str = "Languages: {languages}.",
) -> int:
    """Append a language descriptor to each natural-language parameter's description.

    A parameter is natural-language per is_natural_language_param: ground truth
    decides for params it references (the correct function), and the schema type is
    the fallback for distractor-function params it does not. Enums are never tagged.
    Mutates `entry` in place; returns the number of parameter descriptions modified.
    """
    functions = entry.get("function", [])

    # (func_name, param) -> ground-truth value list, used to refine detection.
    gt_values: dict[tuple[str, str], list] = {}
    for call in entry.get("ground_truth", []):
        for func_name, params in call.items():
            for param, value_list in params.items():
                gt_values[(func_name, param)] = value_list

    hint = template.format(languages=languages)
    modified = 0
    for func in functions:
        name = func.get("name", "")
        props = func.get("parameters", {}).get("properties", {})
        for param, pdef in props.items():
            if not is_natural_language_param(pdef, gt_values.get((name, param))):
                continue
            desc = pdef.get("description", "")
            if hint in desc:  # idempotent — never double-append
                continue
            pdef["description"] = f"{desc} {hint}".strip()
            modified += 1
    return modified


def messages_key(test_case: dict[str, Any]) -> str:
    """Key holding the conversation turns: "messages" (current format) or legacy "question"."""
    return "messages" if "messages" in test_case else "question"


def build_input(
    test_case: dict[str, Any],
    locale: Locale,
    level: LocalizationLevel,
) -> dict[str, str]:
    """Build the chain-input dict for one (test_case, locale) batch item.

    The payload mirrors the level: QUERY sends only {"queries": [...]}, while FULL
    also sends {"values": [...]} with the textual values of the entry's inline
    `ground_truth`.
    """
    queries = collect_query_texts(test_case.get(messages_key(test_case), []))
    payload_obj: dict[str, list] = {"queries": queries}
    if level == LocalizationLevel.FULL:
        payload_obj["values"] = collect_textual_value_contexts(
            test_case, test_case.get("ground_truth", [])
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

    # Be tolerant if the model echoed the value context objects instead of plain
    # strings, e.g. {"param":..., "value":"..."} — pull the translated value out.
    values = [v.get("value", "") if isinstance(v, dict) else v for v in values]
    return {"queries": queries, "values": values}


# ---------------------------------------------------------------------------
# Reassembling the translated test case (ground truth travels inline)
# ---------------------------------------------------------------------------

def _suffixed_id(base_id: str, locale: Locale) -> str:
    return f"{base_id}_{locale.code}"


def apply_translation(
    test_case: dict[str, Any],
    locale: Locale,
    level: LocalizationLevel,
    parsed: dict[str, list],
) -> dict[str, Any] | None:
    """Build the translated test-case entry, including its `ground_truth`.

    QUERY keeps the ground truth unchanged; FULL substitutes the translated
    textual values. Returns None if the reply's item counts don't match what we
    sent (the caller should treat that as a failed entry).
    """
    queries = parsed["queries"]
    values = parsed["values"]
    new_id = _suffixed_id(test_case["id"], locale)
    mkey = messages_key(test_case)

    expected_queries = collect_query_texts(test_case.get(mkey, []))
    if len(queries) != len(expected_queries):
        print(
            f"[WARN] {new_id}: expected "
            f"{len(expected_queries)} query string(s), got {len(queries)}.",
            file=sys.stderr,
        )
        return None

    # --- translated messages (function defs left untouched) ---
    new_messages = copy.deepcopy(test_case.get(mkey, []))
    qi = iter(queries)
    for turn in new_messages:
        for msg in turn:
            if msg.get("role") == "user":
                msg["content"] = next(qi)

    # --- ground truth (translated only for FULL) ---
    ground_truth = copy.deepcopy(test_case.get("ground_truth", []))
    if level == LocalizationLevel.FULL:
        expected_values = collect_textual_values(ground_truth)
        if len(values) != len(expected_values):
            print(
                f"[WARN] {new_id}: expected {len(expected_values)} "
                f"value(s), got {len(values)}.",
                file=sys.stderr,
            )
            return None
        vi = iter(values)
        for call in ground_truth:
            for params in call.values():
                for key, value_list in params.items():
                    params[key] = [
                        next(vi) if is_translatable_string(v) else v
                        for v in value_list
                    ]

    entry = {k: copy.deepcopy(v) for k, v in test_case.items() if k != "ground_truth"}
    entry[mkey] = new_messages
    entry["id"] = new_id
    entry["source_id"] = test_case["id"]
    entry["locale"] = locale.code
    entry["localization_level"] = level.value
    entry["ground_truth"] = ground_truth
    return entry
