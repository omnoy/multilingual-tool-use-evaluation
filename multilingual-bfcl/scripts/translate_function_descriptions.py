"""
translate_function_descriptions.py — build a benchmark whose function/parameter
DESCRIPTIONS are translated into one or more locales, with everything else left in English.

This produces the "English query + non-English API" condition: the user query stays
exactly as in the base benchmark, but every function description and parameter
description in the tool definitions is translated. Function names, parameter names,
types, enum values, defaults, and the inline ground truth are all left unchanged.

Efficiency: the base benchmark reuses the same functions across many entries. The
script first collects the set of distinct function definitions, translates each one
exactly once per locale (cached to disk), and only then rebuilds every entry from the
translated definitions. Deduplication is keyed on the full function JSON — NOT the
name — because some names appear with several different definitions (e.g.
restaurant.find_nearby has 4). Only definitions missing from the cache are sent, so a
re-run after partial failures translates just what is still missing.

Only strings under a "description" key are translated (function-level and any nested
parameter descriptions, at any depth). Positional mapping keeps them aligned.

Like translate_benchmark.py, requests go through the Azure OpenAI Batch API by default
(one batch item per (distinct function, locale)); --no-batch sends them as plain
concurrent requests instead, for deployments without a Global/Data Zone Batch SKU.

Input  (under data/benchmarks/<category>/):  --source, default en/en_translatable.jsonl
Output (under data/benchmarks/<category>/):
  <locale>/<locale>_translatable_func_desc.jsonl     e.g. he/he_translatable_func_desc.jsonl
  <locale>/.<locale>_fn_desc_cache.json              translation cache
  translate_func_desc_manifest.json                  batch manifest (for --retrieve)

Entries follow the translate_benchmark convention: id "<source id>_<locale>", plus
source_id, locale, localization_level="function_descriptions", and the (unchanged)
ground_truth. The query is byte-for-byte the base English query.

Usage:
    # Submit and wait for results:
    python scripts/translate_function_descriptions.py --category bfcl_multiple --locales he --model gpt-5.6-terra

    # Submit only, then retrieve later:
    python scripts/translate_function_descriptions.py --category bfcl_multiple --locales he --model gpt-5.6-terra --submit-only
    python scripts/translate_function_descriptions.py --category bfcl_multiple --retrieve batch_01AbCd...

    # No Batch-SKU deployment available? Send plain concurrent requests (full price):
    python scripts/translate_function_descriptions.py --category bfcl_multiple --locales he --model gpt-5.6-terra --no-batch

    # Preview the first prompt and the cost estimate without calling the API:
    python scripts/translate_function_descriptions.py --category bfcl_multiple --locales he --model gpt-5.6-terra --dry-run

--model is required unless --retrieve; there is no default, to prevent accidental runs.

Cost: an estimate (tiktoken-based) is printed before submitting, and the actual cost
(from the API's reported token usage) after the results arrive. Both use
model_data.csv list prices, with the Batch API discount applied unless --no-batch.

Environment (multilingual-bfcl/.env):
    AZURE_OPENAI_ENDPOINT=...      # …/openai/v1 (or the resource root)
    AZURE_OPENAI_API_KEY=...
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

from multilingual_bfcl.azure_batch import (  # noqa: E402
    BatchItemResult,
    build_chat_request,
    collect_results,
    get_batch,
    is_finished,
    run_direct,
    status_line,
    submit_batch,
)
from multilingual_bfcl.azure_client import (  # noqa: E402
    OPENAI_MODEL_TYPES,
    ModelType,
    build_chat_params,
    make_sync_client,
)
from multilingual_bfcl.benchmark_builder import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SOURCE,
    load_jsonl,
    output_path,
    write_jsonl,
)
from multilingual_bfcl.localization.locale_config import Locale, get_locale  # noqa: E402
from multilingual_bfcl.pricing import count_tokens, format_cost  # noqa: E402

BENCHMARK_DIR = PACKAGE_ROOT / "data" / "benchmarks"

OUTPUT_SUFFIX = "func_desc"
LOCALIZATION_LEVEL = "function_descriptions"
MANIFEST_NAME = "translate_func_desc_manifest.json"

# Cap on translation output tokens per request.
MAX_TOKENS = 4096
# Seconds between batch status polls while waiting for results.
POLL_INTERVAL_S = 30
# Rough output/input-strings token ratio for the pre-submit estimate (the reply
# re-emits every description translated; non-Latin scripts tokenize less compactly).
OUTPUT_TOKEN_RATIO = 1.5

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
# Prompting / parsing
# ---------------------------------------------------------------------------

def _hints(locale: Locale) -> str:
    return "; ".join(locale.model_hints) if locale.model_hints else "none"


def render_messages(descriptions: list[str], locale: Locale) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT.format(language=locale.name, hints=_hints(locale))
    human = HUMAN_TEMPLATE.format(
        n=len(descriptions), language=locale.name,
        payload=json.dumps(descriptions, ensure_ascii=False, indent=2),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": human}]


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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def cache_path(category: str, locale_code: str) -> Path:
    return BENCHMARK_DIR / category / locale_code / f".{locale_code}_fn_desc_cache.json"


def _load_cache(path: Path) -> dict[str, list[str]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=0)


def manifest_path(category: str) -> Path:
    return BENCHMARK_DIR / category / MANIFEST_NAME


# ---------------------------------------------------------------------------
# Plan: which (distinct function, locale) pairs still need translating
# ---------------------------------------------------------------------------

def load_entries(category: str, source: str, limit: int | None) -> list[dict]:
    src = BENCHMARK_DIR / category / source
    if not src.exists():
        sys.exit(f"ERROR: {src} not found.")
    entries = load_jsonl(src)
    return entries[:limit] if limit else entries


def distinct_functions(entries: list[dict]) -> dict[str, dict]:
    unique: dict[str, dict] = {}
    for entry in entries:
        for func in entry["function"]:
            unique.setdefault(func_key(func), func)
    return unique


def descriptions_by_cache_key(
    unique: dict[str, dict], locale: Locale
) -> dict[str, list[str]]:
    """{cache key: descriptions} for every distinct function with something to translate."""
    out: dict[str, list[str]] = {}
    for func in unique.values():
        descs = collect_descriptions(func)
        if descs:
            out[_cache_key(descs, locale.name)] = descs
    return out


# ---------------------------------------------------------------------------
# Results -> cache -> output files
# ---------------------------------------------------------------------------

def merge_results(
    results: dict[str, BatchItemResult],
    units: list[dict[str, str]],
    category: str,
    source: str,
    limit: int | None,
) -> tuple[int, int]:
    """Parse results (keyed by unit index) into the per-locale caches.

    Returns (input_tokens, output_tokens) summed over all results.
    """
    entries = load_entries(category, source, limit)
    unique = distinct_functions(entries)
    by_locale = {u["locale"] for u in units}
    expected = {lc: descriptions_by_cache_key(unique, get_locale(lc)) for lc in by_locale}
    caches = {lc: _load_cache(cache_path(category, lc)) for lc in by_locale}

    n_fail = 0
    for cid, item in results.items():
        idx = int(cid)
        if not (0 <= idx < len(units)):
            print(f"[WARN] result {cid!r} out of range; skipping.", file=sys.stderr)
            continue
        unit = units[idx]
        descs = expected[unit["locale"]].get(unit["key"])
        if descs is None:
            print(f"[WARN] unit {idx}: function no longer in source; skipping.", file=sys.stderr)
            continue
        if not item.success:
            print(f"[ERR] unit {idx} ({unit['name']}, {unit['locale']}) — {item.error}",
                  file=sys.stderr)
            n_fail += 1
            continue
        parsed = _parse_descriptions(item.content or "", len(descs))
        if parsed is None:
            print(f"[WARN] unit {idx} ({unit['name']}, {unit['locale']}): malformed or "
                  f"wrong-length reply (expected {len(descs)} item(s)).", file=sys.stderr)
            n_fail += 1
            continue
        caches[unit["locale"]][unit["key"]] = parsed

    for lc, cache in caches.items():
        _save_cache(cache_path(category, lc), cache)
    if n_fail:
        print(f"  {n_fail} function translation(s) failed — not cached; re-run to retry them.",
              file=sys.stderr)

    in_tok = sum(item.input_tokens for item in results.values())
    out_tok = sum(item.output_tokens for item in results.values())
    return in_tok, out_tok


def write_outputs(category: str, source: str, locales: list[str], limit: int | None) -> None:
    """Rebuild every entry from cached translations and write one file per locale.

    Functions with no cached translation fall back to English (and are reported).
    """
    entries = load_entries(category, source, limit)
    unique = distinct_functions(entries)

    for lc in locales:
        locale = get_locale(lc)
        cache = _load_cache(cache_path(category, lc))
        translated: dict[str, dict] = {}
        fallbacks: list[str] = []
        for key, func in unique.items():
            descs = collect_descriptions(func)
            cached = cache.get(_cache_key(descs, locale.name)) if descs else []
            if cached is None or len(cached) != len(descs):
                fallbacks.append(func.get("name", "<unknown>"))
                cached = descs
            translated[key] = apply_descriptions(func, cached)

        out_entries = []
        for entry in entries:
            new_entry = {k: copy.deepcopy(v) for k, v in entry.items() if k != "ground_truth"}
            new_entry["function"] = [translated[func_key(f)] for f in entry["function"]]
            new_entry["id"] = f"{entry['id']}_{locale.code}"
            new_entry["source_id"] = entry["id"]
            new_entry["locale"] = locale.code
            new_entry["localization_level"] = LOCALIZATION_LEVEL
            if "ground_truth" in entry:
                new_entry["ground_truth"] = copy.deepcopy(entry["ground_truth"])
            out_entries.append(new_entry)

        out = output_path(category, source, locale.code, OUTPUT_SUFFIX)
        write_jsonl(out_entries, out)
        print(f"[done] {len(out_entries)} entries -> {out}")
        if fallbacks:
            print(f"[WARN] {lc}: {len(fallbacks)} function(s) left in English (no translation): "
                  f"{', '.join(sorted(set(fallbacks)))}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Submit / retrieve
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    category, source, limit = args.category, args.source, args.limit or None
    model_type = ModelType(args.model_type)
    use_batch = not args.no_batch

    entries = load_entries(category, source, limit)
    unique = distinct_functions(entries)

    # One unit per (distinct function, locale) not already in the cache.
    units: list[dict[str, str]] = []
    messages: list[list[dict[str, str]]] = []
    payload_tokens = 0
    for lc in args.locales:
        locale = get_locale(lc)
        cache = _load_cache(cache_path(category, lc))
        names = {_cache_key(collect_descriptions(f), locale.name): f.get("name", "")
                 for f in unique.values()}
        for key, descs in descriptions_by_cache_key(unique, locale).items():
            if len(cache.get(key, [])) == len(descs):
                continue
            units.append({"key": key, "locale": lc, "name": names[key]})
            messages.append(render_messages(descs, locale))
            payload_tokens += count_tokens(json.dumps(descs, ensure_ascii=False))

    print(f"Source     : {source}")
    print(f"Locales    : {', '.join(args.locales)}")
    print(f"Functions  : {len(entries)} entries -> {len(unique)} distinct definitions "
          f"({sum(len(e['function']) for e in entries)} instances)")
    print(f"To send    : {len(units)} (distinct function × locale, not yet cached)")
    print(f"Deployment : {args.model}")
    print(f"Mode       : {'Batch API' if use_batch else f'direct (no batch), concurrency {args.concurrency}'}")
    if units:
        in_tok = sum(count_tokens(m["content"]) for msgs in messages for m in msgs)
        est = format_cost(args.model, in_tok, int(payload_tokens * OUTPUT_TOKEN_RATIO),
                          batch=use_batch)
        print(f"Est. cost  : ~{est}")

    if args.dry_run:
        if messages:
            print("\n--- DRY RUN: first prompt ---")
            for msg in messages[0]:
                print(f"[{msg['role']}]\n{msg['content']}\n")
        print(f"\n(Would {'submit' if use_batch else 'send'} {len(units)} items "
              f"to Azure deployment {args.model!r})")
        return

    if not units:
        print("Everything already cached — rebuilding output files.")
        write_outputs(category, source, args.locales, limit)
        return

    params = [
        build_chat_params(args.model, msgs, model_type=model_type, max_tokens=MAX_TOKENS)
        for msgs in messages
    ]

    if not use_batch:
        print(f"Sending {len(params)} requests directly to Azure deployment {args.model!r}...")
        results = await run_direct(params, args.concurrency, args.max_retries)
        in_tok, out_tok = merge_results(results, units, category, source, limit)
        write_outputs(category, source, args.locales, limit)
        print(f"Cost       : {format_cost(args.model, in_tok, out_tok, batch=False)}")
        return

    # custom_id is the unit index, matching the manifest's `units` order.
    requests = [build_chat_request(idx, p) for idx, p in enumerate(params)]
    client = make_sync_client()
    print(f"Submitting batch of {len(requests)} items to Azure deployment {args.model!r}...")
    batch = submit_batch(client, requests,
                         metadata={"task": "translate_func_desc", "category": category})
    batch_id = batch.id
    print(f"Batch job submitted.\n  Batch ID : {batch_id}\n  Category : {category}")

    mpath = manifest_path(category)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({
        "batch_id": batch_id,
        "task": "translate_func_desc",
        "model": args.model,
        "model_type": model_type.value,
        "category": category,
        "source": source,
        "limit": limit,
        "locales": args.locales,
        "units": units,
    }, indent=2), encoding="utf-8")
    print(f"  Manifest : {mpath}")

    if args.submit_only:
        print(
            "\nRetrieve results when the batch finishes:\n"
            f"  python scripts/translate_function_descriptions.py --category {category} "
            f"--retrieve {batch_id}"
        )
        return

    print("Waiting for results (can take up to 24h for large batches)...")
    while True:
        batch = get_batch(client, batch_id)
        if is_finished(batch):
            break
        print(f"  {status_line(batch)} — checking again in {POLL_INTERVAL_S}s", file=sys.stderr)
        await asyncio.sleep(POLL_INTERVAL_S)

    retrieve_and_write(batch_id, category, client)


def retrieve_and_write(batch_id: str, category: str, client=None) -> None:
    """Fetch a completed batch, cache its translations, write outputs, report cost."""
    mpath = manifest_path(category)
    if not mpath.exists():
        sys.exit(f"ERROR: {mpath} not found (written at submit time; required to map results).")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("batch_id") not in (None, batch_id):
        print(f"[WARN] Manifest is for batch {manifest['batch_id']!r}, retrieving {batch_id!r}.",
              file=sys.stderr)

    client = client or make_sync_client()
    print(f"Retrieving results for batch {batch_id!r}...")
    results = collect_results(client, batch_id)
    in_tok, out_tok = merge_results(results, manifest["units"], category,
                                    manifest["source"], manifest.get("limit"))
    write_outputs(category, manifest["source"], manifest["locales"], manifest.get("limit"))
    print(f"Cost       : {format_cost(manifest.get('model', ''), in_tok, out_tok, batch=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate function/parameter descriptions of a benchmark into target locales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category", required=True,
                        help="Benchmark category folder under data/benchmarks/.")
    parser.add_argument("--locales", nargs="+", default=None,
                        help="Target locale codes, e.g. he zh-CN. Required unless --retrieve.")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Input .jsonl path relative to data/benchmarks/<category>/.")
    parser.add_argument("--model", default=None,
                        help="Azure deployment name to translate with. Required unless "
                             "--retrieve (no default, to prevent accidental runs).")
    parser.add_argument("--model-type", choices=[mt.value for mt in OPENAI_MODEL_TYPES],
                        default=ModelType.STANDARD.value,
                        help="Deployment kind: 'standard' (temperature + max_tokens) or "
                             "'reasoning' (no temperature, max_completion_tokens).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N source entries.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the first prompt and the cost estimate, then exit.")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit the batch, print the batch ID, then exit. Use --retrieve later.")
    parser.add_argument("--retrieve", metavar="BATCH_ID", default=None,
                        help="Retrieve a previously submitted Azure batch and write outputs.")
    parser.add_argument("--no-batch", action="store_true",
                        help="Skip the Batch API: send plain concurrent chat.completions "
                             "requests (for deployments without a Global/Data Zone Batch "
                             "SKU). Full price, results written immediately.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="--no-batch only: max requests in flight.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="--no-batch only: SDK retries per request (backoff on 429/5xx).")
    args = parser.parse_args()

    if args.submit_only and args.retrieve:
        parser.error("--submit-only and --retrieve are mutually exclusive.")
    if args.retrieve and args.dry_run:
        parser.error("--dry-run has no effect with --retrieve.")
    if args.no_batch and (args.submit_only or args.retrieve):
        parser.error("--no-batch cannot be combined with --submit-only or --retrieve.")

    if args.retrieve:
        retrieve_and_write(args.retrieve, args.category)
        return

    if not args.locales:
        parser.error("--locales is required unless using --retrieve.")
    if not args.model:
        parser.error("--model is required unless using --retrieve.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
