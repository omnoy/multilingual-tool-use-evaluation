"""
translate_nested_ground_truth.py — Retroactively translate nested ground-truth values.

Before the translator learned to walk nested values, `--level full` translated only
top-level string candidates. Strings nested inside array or object candidates stayed
in English, e.g.

    "elements": [["Sam", "Alice", "Jack"]]
    "items":    [["tomatoes", "lettuce"], ["lettuce", "tomatoes"]]

This script finds those strings in every full-level file (localization_level
"full", "full_full_full", "full_and_function_descriptions", ...), translates them
with the same Azure deployment setup as translate_benchmark.py, and writes an exact
copy of the benchmark folder with the translations filled in. The source folder is
never modified.

What counts as untranslated: a translatable string (see is_translatable_string)
nested inside an array/object candidate that also appears verbatim somewhere in the
English source entry's ground truth (matched by `source_id`). Top-level string
candidates are never touched (the model deliberately kept some of them, e.g. "New
York" in German), and nested values that were already translated (WildToolBench) do
not match the English source, so they are skipped too.

One request per (category, locale, source entry): the derived files of a locale
(full, full_lang_desc, full_func_desc, full_full_full) share the same ground truth,
so they all receive the same translation. Each request carries the entry's already
translated user query as context so the values use the query's wording and
transliterations. Repeated strings within an entry are sent once.

The same pass also repairs enum params whose ground truth holds a translated value
that is not a member of the enum (the old translator sent enum values to the model,
e.g. complexity: ["mittel", ""] for enum low/medium/high): the param's candidates
are replaced with the English source entry's candidates. This needs no API call and
is listed in the dry run.

After the translations are applied, each changed parameter is deduplicated the same
way as dedupe_ground_truth.py does (e.g. ["Criminal Law"] and ["criminal law"] can
both become the same Hebrew string).

Every run that calls the API saves the raw string mapping to --translations-out, so
the files can be rebuilt (or the mapping reviewed / hand-edited and re-applied)
without calling the API again via --translations.

Usage:
    # Preview targets, cost and the first prompt (no API calls, nothing written):
    python scripts/translate_nested_ground_truth.py --model gpt-5.6-terra --dry-run

    # Translate (concurrent chat.completions requests) and write data/benchmarks:
    python scripts/translate_nested_ground_truth.py --model gpt-5.6-terra

    # Lower the parallelism if the deployment rate-limits:
    python scripts/translate_nested_ground_truth.py --model gpt-5.6-terra --concurrency 4

    # Re-apply a saved (possibly hand-edited) mapping without calling the API:
    python scripts/translate_nested_ground_truth.py --translations data/benchmarks_nested_translations.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
load_dotenv(PACKAGE_ROOT / ".env")

from multilingual_bfcl.azure_client import OPENAI_MODEL_TYPES, ModelType  # noqa: E402
from multilingual_bfcl.localization.locale_config import Locale, get_locale  # noqa: E402
from multilingual_bfcl.localization.translator import (  # noqa: E402
    SYSTEM_PROMPT,
    _param_schemas,
    collect_query_texts,
    collect_textual_values,
    messages_key,
    walk_value,
)

_MAX_TOKENS = 2048
_OUTPUT_TOKEN_RATIO = 1.5
# Emails and URLs never need translating; keep them out of the requests.
_URL_OR_EMAIL_RE = re.compile(r"://|^www\.|^[^\s@]+@[^\s@]+\.[^\s@]+$")
# Locale codes used in entry metadata that differ from locale_config's codes.
_LOCALE_ALIASES = {"zh": "zh-CN"}

HUMAN_TEMPLATE = """\
Translate the ground-truth argument values below into {language}.

- "query": the user request, ALREADY translated into {language}. It is context \
only: do not translate it and do not return it. Whenever a value also appears in \
the query, use exactly the same wording, spelling and transliteration as the query.
- "values": ground-truth argument values. Each item is an object \
{{"param": <name>, "description": <what the parameter means>, "value": <the value>}} \
given only for context. Every item was pre-screened as a natural-language value, so \
translate its "value" into {language} — translate or transliterate proper nouns \
exactly as in rule 1. Keep a value unchanged ONLY if it is genuinely a code token, \
programming identifier, file path, or URL.

Return ONLY a JSON object {{"values": [...]}}: a flat list of the translated value \
strings, one per input item, in the same order.

INPUT:
{payload}"""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """One translation request: the nested English strings of one localized entry."""
    category: str
    locale_code: str          # as written in the entry ("zh", "he", ...)
    source_id: str
    query: list[str]
    items: list[dict[str, str]]  # {"param", "description", "value"}, unique values
    files: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.category}|{self.locale_code}|{self.source_id}"


def resolve_locale(code: str) -> Locale | None:
    try:
        return get_locale(_LOCALE_ALIASES.get(code, code))
    except ValueError:
        return None


def nested_strings(entry: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(param_label, description, string) for every translatable string nested
    inside an array/object candidate. Top-level string candidates are skipped."""
    schemas = _param_schemas(entry.get("function", []))
    found: list[tuple[str, str, str]] = []

    def visit(label: str, desc: str, s: str) -> str:
        found.append((label, desc, s))
        return s

    for call in entry.get("ground_truth") or []:
        for func_name, params in call.items():
            if not isinstance(params, dict):
                continue
            for param, candidates in params.items():
                if not isinstance(candidates, list):
                    continue
                schema = schemas.get((func_name, param), {})
                for cand in candidates:
                    if isinstance(cand, (list, dict)):
                        walk_value(cand, schema, param, schema.get("description", ""), visit)
    return found


def map_nested(entry_gt: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Copy of the ground truth with nested strings replaced via `mapping`."""
    def visit(_label: str, _desc: str, s: str) -> str:
        return mapping.get(s, s)

    out = []
    for call in entry_gt:
        new_call = {}
        for func_name, params in call.items():
            if not isinstance(params, dict):
                new_call[func_name] = params
                continue
            new_call[func_name] = {
                param: [
                    walk_value(c, {}, param, "", visit) if isinstance(c, (list, dict)) else c
                    for c in candidates
                ] if isinstance(candidates, list) else candidates
                for param, candidates in params.items()
            }
        out.append(new_call)
    return out


def dedupe_candidates(gt: list[dict]) -> tuple[list[dict], int]:
    """Drop exact-duplicate candidates per parameter (first occurrence kept).

    Same rule as dedupe_ground_truth.py, top level only: translation can map two
    English candidates (["Criminal Law"], ["criminal law"]) to the same string.
    Returns (new_gt, number_of_candidates_removed).
    """
    removed = 0
    out = []
    for call in gt:
        new_call = {}
        for func_name, params in call.items():
            if not isinstance(params, dict):
                new_call[func_name] = params
                continue
            new_params = {}
            for param, candidates in params.items():
                if not isinstance(candidates, list):
                    new_params[param] = candidates
                    continue
                seen: set[str] = set()
                kept = []
                for c in candidates:
                    k = json.dumps(c, ensure_ascii=False, sort_keys=True)
                    if k not in seen:
                        seen.add(k)
                        kept.append(c)
                removed += len(candidates) - len(kept)
                new_params[param] = kept
            new_call[func_name] = new_params
        out.append(new_call)
    return out, removed


def _enum_values(schema: dict) -> list | None:
    """The allowed values of an enum param (or of an array param's items), else None."""
    if isinstance(schema.get("enum"), list):
        return schema["enum"]
    items = schema.get("items")
    if isinstance(items, dict) and isinstance(items.get("enum"), list):
        return items["enum"]
    return None


def _invalid_enum_strings(candidates: list, allowed: list) -> list[str]:
    """Non-empty strings among the candidates (or inside list candidates) not in `allowed`."""
    bad = []
    for c in candidates:
        for s in (c if isinstance(c, list) else [c]):
            if isinstance(s, str) and s != "" and s not in allowed:
                bad.append(s)
    return bad


def restore_enums(entry: dict[str, Any], en_entry: dict[str, Any] | None,
                  ) -> tuple[list[dict], list[tuple[str, Any, Any]], list[str]]:
    """Put the English candidates back for enum params whose ground truth holds a
    translated value that is not a member of the enum.

    The old translator sent enum values to the model, which sometimes translated
    them (complexity: ["mittel", ""] for enum low/medium/high). The fix replaces that
    param's whole candidate list with the English source entry's list (matched by
    function name and call position). Returns (new_gt, fixes, warnings) where each
    fix is (label, old_candidates, new_candidates).
    """
    gt = entry.get("ground_truth") or []
    schemas = _param_schemas(entry.get("function", []))
    en_gt = (en_entry or {}).get("ground_truth") or []
    fixes: list[tuple[str, Any, Any]] = []
    warnings: list[str] = []
    seen_calls: dict[str, int] = {}
    out = []
    for call in gt:
        new_call = {}
        for func_name, params in call.items():
            if not isinstance(params, dict):
                new_call[func_name] = params
                continue
            # The n-th call to func_name pairs with the n-th English call to it.
            n = seen_calls.get(func_name, 0)
            seen_calls[func_name] = n + 1
            en_calls = [c[func_name] for c in en_gt if func_name in c]
            en_params = en_calls[n] if n < len(en_calls) else {}
            new_params = {}
            for param, candidates in params.items():
                allowed = _enum_values(schemas.get((func_name, param), {}))
                if (allowed is None or not isinstance(candidates, list)
                        or not _invalid_enum_strings(candidates, allowed)):
                    new_params[param] = candidates
                    continue
                label = f"{func_name}.{param}"
                en_cands = en_params.get(param)
                # Restore when at least one English candidate is a real enum member
                # (BFCL sometimes adds case variants: ["Bronze", "bronze"]).
                en_valid = isinstance(en_cands, list) and any(
                    s in allowed
                    for c in en_cands for s in (c if isinstance(c, list) else [c])
                    if isinstance(s, str) and s != ""
                )
                if not en_valid:
                    warnings.append(f"{entry.get('id')} {label}: no valid English candidates "
                                    f"to restore (English: {en_cands!r}); left as is")
                    new_params[param] = candidates
                    continue
                new_params[param] = json.loads(json.dumps(en_cands))
                fixes.append((label, candidates, new_params[param]))
            new_call[func_name] = new_params
        out.append(new_call)
    return out, fixes, warnings


def category_dirs(src: Path, categories: list[str] | None) -> list[Path]:
    """Benchmark subfolders to process: `categories` if given, else all but backups/."""
    dirs = sorted(p for p in src.iterdir() if p.is_dir() and p.name != "backups")
    if categories is None:
        return dirs
    return [p for p in dirs if p.name in categories]


def load_english(src: Path, categories: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """{category: {id: English source entry}} for every category with an en file."""
    out: dict[str, dict[str, dict]] = {}
    for cat_dir in category_dirs(src, categories):
        en_path = cat_dir / "en" / "en_translatable.jsonl"
        if en_path.exists():
            out[cat_dir.name] = {e["id"]: e for _, e in iter_jsonl(en_path) if e and "id" in e}
    return out


def iter_jsonl(path: Path):
    """Yield (line, parsed_or_None) for every line, keeping the raw text."""
    with path.open(encoding="utf-8", newline="") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                yield line, None
                continue
            try:
                yield line, json.loads(stripped)
            except json.JSONDecodeError:
                yield line, None


def is_full_level(entry: dict[str, Any]) -> bool:
    return str(entry.get("localization_level", "")).startswith("full")


def discover(src: Path, categories: list[str] | None = None,
             ) -> tuple[dict[str, Unit], list[tuple[str, str, str, Any, Any]], list[str]]:
    """Collect translation units and enum fixes from every full-level file under `src`.

    Enum fixes are (file, entry_id, param_label, old_candidates, new_candidates).
    """
    units: dict[str, Unit] = {}
    enum_fixes: list[tuple[str, str, str, Any, Any]] = []
    warnings: list[str] = []
    english_by_cat = load_english(src, categories)
    for cat_dir in category_dirs(src, categories):
        english = english_by_cat.get(cat_dir.name)
        if english is None:
            warnings.append(f"{cat_dir.name}: no en/en_translatable.jsonl, category skipped")
            continue
        for path in sorted(cat_dir.glob("*/*.jsonl")):
            if path.parent.name == "en":
                continue
            rel = path.relative_to(src).as_posix()
            for _, entry in iter_jsonl(path):
                if not entry or not is_full_level(entry) or "ground_truth" not in entry:
                    continue
                src_entry = english.get(entry.get("source_id"))
                if src_entry is None:
                    warnings.append(f"{rel}: {entry.get('id')} has no English source entry")
                    continue
                _, fixes, fix_warnings = restore_enums(entry, src_entry)
                enum_fixes += [(rel, entry.get("id"), *f) for f in fixes]
                warnings += [f"{rel}: {w}" for w in fix_warnings]
                english_strings = set(collect_textual_values(src_entry.get("ground_truth", [])))
                targets = [t for t in nested_strings(entry)
                           if t[2] in english_strings and not _URL_OR_EMAIL_RE.search(t[2])]
                if not targets:
                    continue
                key = f"{cat_dir.name}|{entry.get('locale')}|{entry['source_id']}"
                unit = units.get(key)
                if unit is None:
                    seen: set[str] = set()
                    items = []
                    for label, desc, s in targets:
                        if s not in seen:
                            seen.add(s)
                            items.append({"param": label, "description": desc, "value": s})
                    unit = Unit(
                        category=cat_dir.name,
                        locale_code=str(entry.get("locale")),
                        source_id=entry["source_id"],
                        query=collect_query_texts(entry.get(messages_key(entry), [])),
                        items=items,
                    )
                    units[key] = unit
                else:
                    # Derived files should share the ground truth; pick up any extra strings.
                    have = {i["value"] for i in unit.items}
                    for label, desc, s in targets:
                        if s not in have:
                            have.add(s)
                            unit.items.append({"param": label, "description": desc, "value": s})
                unit.files.add(rel)
    return units, enum_fixes, warnings


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def render(unit: Unit, locale: Locale) -> list[dict[str, str]]:
    hints = "; ".join(locale.model_hints) if locale.model_hints else "none"
    payload = json.dumps({"query": unit.query, "values": unit.items}, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=locale.name, hints=hints)},
        {"role": "user", "content": HUMAN_TEMPLATE.format(language=locale.name, payload=payload)},
    ]


def parse_values(raw: str, n_expected: int, key: str) -> list[str] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        try:
            obj = json.loads(text[start:end]) if start != -1 and end > start else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("values"), list):
        print(f"[WARN] {key}: could not parse reply: {raw!r}", file=sys.stderr)
        return None
    values = [v.get("value", "") if isinstance(v, dict) else v for v in obj["values"]]
    if len(values) != n_expected or not all(isinstance(v, str) for v in values):
        print(f"[WARN] {key}: expected {n_expected} string value(s), got {values!r}",
              file=sys.stderr)
        return None
    return values


# ---------------------------------------------------------------------------
# Applying + copying
# ---------------------------------------------------------------------------

def write_copy(src: Path, dst: Path, translations: dict[str, dict[str, str]],
               verbose: bool, categories: list[str] | None = None) -> None:
    """Copy src -> dst, rewriting only entries whose ground truth changes
    (nested translations applied and/or translated enum values restored).
    Folders outside `categories` are copied verbatim."""
    per_file: dict[str, list[int]] = {}
    changes: list[tuple[str, str, Any, Any]] = []
    dedup_removed = 0
    english_by_cat = load_english(src, categories)
    selected = {p.name for p in category_dirs(src, categories)}
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(src)
        out = dst / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".jsonl" or rel_path.parts[0] not in selected:
            shutil.copy2(path, out)
            continue
        rel = rel_path.as_posix()
        category = rel_path.parts[0]
        lines = []
        n_changed = n_strings = n_enum = 0
        for line, entry in iter_jsonl(path):
            if not (entry and is_full_level(entry) and "ground_truth" in entry):
                lines.append(line)
                continue
            mapping = translations.get(f"{category}|{entry.get('locale')}|{entry.get('source_id')}") or {}
            en_entry = english_by_cat.get(category, {}).get(entry.get("source_id"))
            old_gt = entry["ground_truth"]
            new_gt = map_nested(old_gt, mapping) if mapping else old_gt
            if en_entry is not None:
                new_gt, fixes, _ = restore_enums({**entry, "ground_truth": new_gt}, en_entry)
                n_enum += len(fixes)
            if new_gt == old_gt:
                lines.append(line)
                continue
            new_gt, n_removed = dedupe_candidates(new_gt)
            dedup_removed += n_removed
            n_strings += sum(1 for s in nested_strings(entry) if s[2] in mapping)
            entry["ground_truth"] = new_gt
            n_changed += 1
            changes.append((rel, entry.get("id"), old_gt, new_gt))
            ending = line[len(line.rstrip("\r\n")):]
            lines.append(json.dumps(entry, ensure_ascii=False) + ending)
        with out.open("w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        if n_changed:
            per_file[rel] = [n_changed, n_strings, n_enum]

    print("\n=== Written ===")
    print(f"{'file':<70} {'entries':>8} {'strings':>8} {'enums':>6}")
    for rel, (n, s, e) in per_file.items():
        print(f"{rel:<70} {n:>8} {s:>8} {e:>6}")
    print(f"duplicate candidates removed after translation: {dedup_removed}")
    print(f"destination: {dst}")
    if verbose:
        print("\n=== Changes ===")
        for rel, eid, old, new in changes:
            print(f"[{rel}] {eid}")
            print(f"    before: {json.dumps(old, ensure_ascii=False)}")
            print(f"    after:  {json.dumps(new, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate nested ground-truth values into a copy of the benchmark folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", default="data/benchmarks",
                        help="Source benchmark folder (relative to the package root).")
    parser.add_argument("--dst", default="data/benchmarks",
                        help="Destination folder (relative to the package root).")
    parser.add_argument("--categories", nargs="+", default=["bfcl_multiple"],
                        help="Benchmark subfolders to process; all other folders are "
                             "copied unchanged. Pass 'all' for every subfolder except backups/.")
    parser.add_argument("--model", help="Azure deployment name. Required unless --translations.")
    parser.add_argument("--model-type", choices=[mt.value for mt in OPENAI_MODEL_TYPES],
                        default=ModelType.STANDARD.value)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max requests in flight.")
    parser.add_argument("--max-retries", type=int, default=6,
                        help="SDK retries per request (backoff on 429/5xx).")
    parser.add_argument("--translations", metavar="JSON",
                        help="Apply a saved mapping instead of calling the API.")
    parser.add_argument("--translations-out", default=None,
                        help="Where to save the mapping (default: <dst>_translations.json).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show targets, cost estimate and the first prompt; call nothing, write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the destination folder if it exists.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every target string / every changed entry.")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    src = (PACKAGE_ROOT / args.src).resolve()
    dst = (PACKAGE_ROOT / args.dst).resolve()
    tr_out = Path(args.translations_out) if args.translations_out else dst.with_name(
        dst.name + "_translations.json")
    if not src.is_dir():
        sys.exit(f"Source folder not found: {src}")
    if dst == src or src in dst.parents:
        sys.exit("Destination must not be the source folder or inside it.")
    if not args.translations and not args.model:
        parser.error("--model is required unless --translations is given.")
    if not args.dry_run and dst.exists() and not args.overwrite:
        sys.exit(f"Destination exists: {dst} (pass --overwrite to replace it)")

    categories = None if args.categories == ["all"] else args.categories
    if categories is not None:
        missing = [c for c in categories if not (src / c).is_dir()]
        if missing:
            sys.exit(f"Category folder(s) not found under {src}: {', '.join(missing)}")

    units, enum_fixes, warnings = discover(src, categories)
    unit_list = sorted(units.values(), key=lambda u: u.key)

    # --- report targets ---
    print(f"source:      {src}")
    print(f"categories:  {', '.join(categories) if categories else 'all'}")
    print(f"destination: {'(dry run, nothing written)' if args.dry_run else dst}")
    by_group: dict[tuple[str, str], list[Unit]] = {}
    for u in unit_list:
        by_group.setdefault((u.category, u.locale_code), []).append(u)
    print("\n=== Untranslated nested values ===")
    print(f"{'category':<16} {'locale':<7} {'entries':>8} {'strings':>8}  files")
    for (cat, loc), us in by_group.items():
        files = sorted({f.split("/")[-1] for u in us for f in u.files})
        print(f"{cat:<16} {loc:<7} {len(us):>8} {sum(len(u.items) for u in us):>8}  {', '.join(files)}")
    print(f"total: {len(unit_list)} requests, {sum(len(u.items) for u in unit_list)} unique strings")
    shown = unit_list if args.verbose else unit_list[:15]
    print("\n=== Targets ===" + ("" if args.verbose else " (first 15; --verbose for all)"))
    for u in shown:
        vals = ", ".join(f"{i['param']}={i['value']!r}" for i in u.items)
        print(f"  {u.category}/{u.locale_code}/{u.source_id}: {vals}")

    # Enum fixes repeat across the derived files of a locale; list each entry once.
    print("\n=== Translated enum values restored to English (no API needed) ===")
    by_file: dict[str, int] = {}
    unique: dict[tuple[str, str], tuple[Any, Any]] = {}
    for rel, eid, label, old, new in enum_fixes:
        by_file[rel] = by_file.get(rel, 0) + 1
        unique.setdefault((eid, label), (old, new))
    for rel, n in by_file.items():
        print(f"  {rel:<70} {n:>4} param(s)")
    for (eid, label), (old, new) in sorted(unique.items()):
        print(f"  {eid:<18} {label:<45} {json.dumps(old, ensure_ascii=False)} -> "
              f"{json.dumps(new, ensure_ascii=False)}")
    print(f"total: {len(unique)} param(s) across {len(enum_fixes)} file entries")
    for w in warnings:
        print(f"[WARN] {w}", file=sys.stderr)

    # --- locales ---
    locales: dict[str, Locale] = {}
    runnable: list[Unit] = []
    for u in unit_list:
        loc = resolve_locale(u.locale_code)
        if loc is None:
            print(f"[WARN] unsupported locale {u.locale_code!r}; skipping {u.key}", file=sys.stderr)
            continue
        locales[u.key] = loc
        runnable.append(u)

    # --- translations from file ---
    if args.translations:
        translations = json.loads(Path(args.translations).read_text(encoding="utf-8"))["translations"]
        print(f"\nLoaded {len(translations)} mapping(s) from {args.translations}")
        if args.dry_run:
            return
        if dst.exists():
            shutil.rmtree(dst)
        write_copy(src, dst, translations, args.verbose, categories)
        return

    from multilingual_bfcl.azure_batch import run_direct
    from multilingual_bfcl.azure_client import build_chat_params
    from multilingual_bfcl.pricing import count_tokens, format_cost

    messages = [render(u, locales[u.key]) for u in runnable]
    in_tok = sum(count_tokens(m["content"]) for msgs in messages for m in msgs)
    out_tok = int(sum(count_tokens(json.dumps([i["value"] for i in u.items], ensure_ascii=False))
                      for u in runnable) * _OUTPUT_TOKEN_RATIO) + 10 * len(runnable)
    print(f"\nDeployment : {args.model}")
    print(f"Mode       : concurrent requests (concurrency {args.concurrency}, "
          f"{args.max_retries} retries each)")
    print(f"Est. cost  : ~{format_cost(args.model, in_tok, out_tok, batch=False)}")

    if args.dry_run:
        if messages:
            print(f"\n--- DRY RUN: first prompt ({runnable[0].key}) ---")
            for m in messages[0]:
                print(f"[{m['role']}]\n{m['content']}\n")
        print(f"(Would send {len(messages)} request(s); nothing written.)")
        return
    if not runnable:
        print("Nothing to translate; writing the enum fixes only.")
        if dst.exists():
            shutil.rmtree(dst)
        write_copy(src, dst, {}, args.verbose, categories)
        return

    params = [build_chat_params(args.model, m, model_type=ModelType(args.model_type),
                                max_tokens=_MAX_TOKENS) for m in messages]
    print(f"Sending {len(params)} request(s) to {args.model!r}...")
    results = asyncio.run(run_direct(params, args.concurrency, args.max_retries))

    translations: dict[str, dict[str, str]] = {}
    n_fail = 0
    in_used = out_used = 0
    for idx, u in enumerate(runnable):
        item = results.get(str(idx))
        if item is None or not item.success:
            print(f"[ERR] {u.key}: {getattr(item, 'error', 'no result')}", file=sys.stderr)
            n_fail += 1
            continue
        in_used += item.input_tokens
        out_used += item.output_tokens
        values = parse_values(item.content or "", len(u.items), u.key)
        if values is None:
            n_fail += 1
            continue
        translations[u.key] = {i["value"]: v for i, v in zip(u.items, values)}

    tr_out.parent.mkdir(parents=True, exist_ok=True)
    tr_out.write_text(json.dumps({"model": args.model, "src": str(src),
                                  "translations": translations},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(translations)} mapping(s) -> {tr_out}")
    print(f"Cost       : {format_cost(args.model, in_used, out_used, batch=False)}")
    if n_fail:
        print(f"[WARN] {n_fail} request(s) failed; their entries stay untranslated.", file=sys.stderr)

    if dst.exists():
        shutil.rmtree(dst)
    write_copy(src, dst, translations, args.verbose, categories)


if __name__ == "__main__":
    main()
