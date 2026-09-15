"""
translate_function_descriptions.py — build a benchmark whose function/parameter
DESCRIPTIONS are translated into Hebrew, with everything else left in English.

This produces the "English query + non-English API" condition: the user query stays
exactly as in the base benchmark, but every function description and parameter
description in the tool definitions is translated. Function names, parameter names,
types, enum values, defaults, and the ground-truth answers are all left unchanged.

Efficiency: the base benchmark reuses the same functions across many entries (367
function instances, ~329 distinct definitions). The script first collects the set of
distinct function definitions, translates each one exactly once (cached to disk), and
only then rebuilds every entry from the translated definitions. Deduplication is keyed
on the full function JSON — NOT the name — because some names appear with several
different definitions (e.g. restaurant.find_nearby has 4).

Only strings under a "description" key are translated (function-level and any nested
parameter descriptions, at any depth). Positional mapping keeps them aligned.

Inputs (under data/benchmarks/multiple/):
  eng_translatable.json                      base entries (English)
  possible_answer/eng_translatable.json      base ground truth (English)

Outputs (under data/benchmarks/multiple/):
  heb/he_translated_function_descriptions.json                 translated benchmark
  possible_answer/heb/he_translated_function_descriptions.json English GT, re-keyed

Entries mirror the existing heb convention: id "<source>_he", plus source_id, locale,
and localization_level="function_descriptions". The query is byte-for-byte the base
English query; only the tool definitions change.

Usage:
    # Full build (uses the on-disk translation cache; safe to re-run / resume):
    python scripts/translate_function_descriptions.py

    # Cheap smoke test on the first few entries:
    python scripts/translate_function_descriptions.py --limit 5

    # Preview the first translation prompt without calling the API:
    python scripts/translate_function_descriptions.py --dry-run

Environment (multilingual-bfcl/.env):
    ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Make the package importable when run as a plain script.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
load_dotenv(PACKAGE_ROOT / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from multilingual_bfcl.localization.locale_config import get_locale  # noqa: E402

DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks" / "multiple"

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_SOURCE = "eng_translatable.json"
OUTPUT_STEM = "he_translated_function_descriptions"
LOCALIZATION_LEVEL = "function_descriptions"

SYSTEM_PROMPT = """\
You are a professional translator localizing function-calling API documentation into \
{language}. You will receive a JSON list of English strings: function descriptions and \
parameter descriptions from an API specification. Follow these rules strictly:
1. Translate every string into {language} faithfully, idiomatically, and completely. \
Translate proper nouns using the established {language} form when one exists; otherwise \
transliterate them into the {language} script. Do NOT leave natural-language words in English.
2. Leave UNCHANGED only genuinely non-linguistic tokens embedded in the text: code, \
programming identifiers, parameter/enum literal values, units, file paths, URLs, and \
pure numbers/booleans/dates.
3. Do not add, remove, or reorder information; preserve the meaning precisely.
4. Locale conventions: {hints}
5. Output ONLY a JSON object: {{"descriptions": [...]}} with the SAME number of items in \
the SAME order as the input. No markdown, no explanation."""

HUMAN_TEMPLATE = """\
Translate these {n} description string(s) into {language}. Return ONLY \
{{"descriptions": [...]}} with {n} item(s) in the same order.

INPUT:
{payload}"""


# ---------------------------------------------------------------------------
# Description collection / application (deterministic DFS over "description" keys)
# ---------------------------------------------------------------------------

def _is_translatable(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def collect_descriptions(func: dict) -> list[str]:
    """Every non-empty string under a "description" key, in deterministic DFS order.

    Covers the function-level description and all (possibly nested) parameter
    descriptions. A "description" string is a leaf — we do not recurse into it.
    """
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and _is_translatable(value):
                    out.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(func)
    return out


def apply_descriptions(func: dict, translations: list[str]) -> dict:
    """Return a deep copy of func with each translatable "description" replaced by the
    next item from `translations`, following the exact order of collect_descriptions."""
    new_func = copy.deepcopy(func)
    it = iter(translations)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "description" and _is_translatable(value):
                    node[key] = next(it)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(new_func)
    return new_func


def func_key(func: dict) -> str:
    """Canonical dedup key for a function definition (full definition, not just name)."""
    return json.dumps(func, sort_keys=True, ensure_ascii=False)


def _cache_key(descriptions: list[str], language: str) -> str:
    payload = json.dumps([language, descriptions], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def _parse_descriptions(raw: str, expected: int) -> list[str] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    items = obj.get("descriptions")
    if not isinstance(items, list) or len(items) != expected:
        return None
    return [str(x) for x in items]


async def translate_descriptions(
    client: AsyncAnthropic,
    model: str,
    descriptions: list[str],
    language: str,
    hints: str,
    max_tokens: int,
) -> list[str] | None:
    """Translate an ordered list of description strings. None on unrecoverable failure."""
    if not descriptions:
        return []
    system = SYSTEM_PROMPT.format(language=language, hints=hints)
    human = HUMAN_TEMPLATE.format(
        n=len(descriptions), language=language,
        payload=json.dumps(descriptions, ensure_ascii=False, indent=2),
    )
    # opus-4-8 rejects the temperature parameter, so we don't pass it.
    for _ in range(2):  # one retry on a malformed / wrong-length reply
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": human}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        parsed = _parse_descriptions(raw, len(descriptions))
        if parsed is not None:
            return parsed
    return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_cache(path: Path) -> dict[str, list[str]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    locale = get_locale("he")
    language = locale.name
    hints = "; ".join(locale.model_hints) if locale.model_hints else "none"

    source_path = DATA_ROOT / args.source
    answer_path = DATA_ROOT / "possible_answer" / args.source
    entries = _load_jsonl(source_path)
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
    answers = {a["id"]: a["ground_truth"] for a in _load_jsonl(answer_path)}

    # 1. Collect distinct function definitions across the (possibly limited) entries.
    unique: dict[str, dict] = {}
    for entry in entries:
        for func in entry["function"]:
            unique.setdefault(func_key(func), func)
    print(f"{len(entries)} entries -> {len(unique)} distinct function definitions "
          f"({sum(len(e['function']) for e in entries)} instances)")

    if args.dry_run:
        sample = next(iter(unique.values()))
        descs = collect_descriptions(sample)
        print("\n--- SYSTEM ---\n" + SYSTEM_PROMPT.format(language=language, hints=hints))
        print("\n--- HUMAN ---\n" + HUMAN_TEMPLATE.format(
            n=len(descs), language=language,
            payload=json.dumps(descs, ensure_ascii=False, indent=2)))
        return

    # 2. Translate each distinct definition once (cache-backed, concurrent).
    cache_path = PACKAGE_ROOT / args.cache
    cache = _load_cache(cache_path)
    client = AsyncAnthropic(max_retries=args.max_retries)
    semaphore = asyncio.Semaphore(args.concurrency)
    translated_funcs: dict[str, dict] = {}
    failures: list[str] = []
    done = 0

    async def worker(key: str, func: dict) -> None:
        nonlocal done
        descriptions = collect_descriptions(func)
        ck = _cache_key(descriptions, language)
        cached = cache.get(ck)
        if cached is not None and len(cached) == len(descriptions):
            result = cached
        else:
            async with semaphore:
                result = await translate_descriptions(
                    client, args.model, descriptions, language, hints, args.max_tokens
                )
            if result is None:
                failures.append(func.get("name", "<unknown>"))
                result = descriptions  # fall back to English so the pipeline still builds
            else:
                cache[ck] = result
        translated_funcs[key] = apply_descriptions(func, result)
        done += 1
        if done % 25 == 0 or done == len(unique):
            print(f"  translated {done}/{len(unique)} functions")

    await asyncio.gather(*(worker(k, f) for k, f in unique.items()))
    _save_cache(cache_path, cache)

    # 3. Rebuild every entry from the translated definitions; emit re-keyed GT.
    out_entries: list[dict] = []
    out_answers: list[dict] = []
    missing_gt = 0
    for entry in entries:
        new_entry = copy.deepcopy(entry)
        new_entry["function"] = [translated_funcs[func_key(f)] for f in entry["function"]]
        new_entry["source_id"] = entry["id"]
        new_entry["id"] = f"{entry['id']}_{locale.code}"
        new_entry["locale"] = locale.code
        new_entry["localization_level"] = LOCALIZATION_LEVEL
        out_entries.append(new_entry)

        gt = answers.get(entry["id"])
        if gt is None:
            missing_gt += 1
        else:
            out_answers.append({"id": new_entry["id"], "ground_truth": gt})

    out_bench = DATA_ROOT / "heb" / f"{OUTPUT_STEM}.json"
    out_gt = DATA_ROOT / "possible_answer" / "heb" / f"{OUTPUT_STEM}.json"
    _write_jsonl(out_bench, out_entries)
    _write_jsonl(out_gt, out_answers)

    print(f"\nwrote {len(out_entries)} entries -> {out_bench}")
    print(f"wrote {len(out_answers)} answers -> {out_gt}")
    if missing_gt:
        print(f"[WARN] {missing_gt} entries had no matching ground truth", file=sys.stderr)
    if failures:
        print(f"[WARN] {len(failures)} function(s) fell back to English "
              f"(translation failed): {', '.join(sorted(set(failures)))}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate function/parameter descriptions of the base benchmark into Hebrew.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Base benchmark filename under data/benchmarks/multiple/.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Translation model id.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N entries (0 = all).")
    parser.add_argument("--concurrency", type=int, default=6,
                        help="Max concurrent translation requests.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="max_tokens per translation request.")
    parser.add_argument("--max-retries", type=int, default=6,
                        help="SDK-level retries for rate limits / transient errors.")
    parser.add_argument("--cache", default="data/benchmarks/multiple/heb/.he_fn_desc_cache.json",
                        help="Translation cache path (relative to package root).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the first translation prompt and exit without calling the API.")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
