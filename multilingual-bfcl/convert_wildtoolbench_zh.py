"""Convert a WildToolBench zh 'combined' file into the he_translatable-style format.

Input:  WildToolBench_zh.bfcl.combined.translatable.jsonl  (id, messages, function, ground_truth)
Output: zh_translatable.jsonl with the he_translatable schema:
    id                  = <source_id>_<locale>
    messages, function  = carried over (parameters normalized to type/properties/required)
    source_id           = original id
    locale              = "zh"
    localization_level  = per-entry: "full"  if ground_truth params are Chinese,
                                     "query" if they are English
    ground_truth        = carried over

localization_level is decided per entry by scanning the ground_truth's string
values recursively: any CJK char => "full" (fully localized answer), otherwise any
Latin letter => "query" (only the query is localized, answer values stay English).

Usage:
    python convert_wildtoolbench_zh.py [--input <file>] [--locale zh] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
LATIN_RE = re.compile(r"[A-Za-z]")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return records


def iter_strings(value) -> "list[str]":
    """Recursively collect every string leaf inside a nested structure."""
    out = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(iter_strings(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(iter_strings(v))
    return out


def classify_ground_truth(ground_truth) -> str:
    """"full" if any Chinese value, "query" if any Latin value, else "query"."""
    strings = iter_strings(ground_truth)
    if any(CJK_RE.search(s) for s in strings):
        return "full"
    if any(LATIN_RE.search(s) for s in strings):
        return "query"
    return "query"  # no language-bearing values; treat as untranslated answer


def normalize_parameters(params: dict) -> dict:
    return {
        "type": params.get("type", "dict"),
        "properties": params.get("properties", {}),
        "required": params.get("required", []),
    }


def normalize_function(func: dict) -> dict:
    return {
        "name": func.get("name"),
        "description": func.get("description"),
        "parameters": normalize_parameters(func.get("parameters", {})),
    }


def convert(record: dict, locale: str) -> dict:
    source_id = record["id"]
    return {
        "id": f"{source_id}_{locale}",
        "messages": record["messages"],
        "function": [normalize_function(f) for f in record.get("function", [])],
        "source_id": source_id,
        "locale": locale,
        "localization_level": classify_ground_truth(record["ground_truth"]),
        "ground_truth": record["ground_truth"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_input = (Path(__file__).resolve().parent / "data" / "benchmarks"
                     / "wildtoolbench" / "zh"
                     / "WildToolBench_zh.bfcl.combined.translatable.jsonl")
    parser.add_argument("--input", type=Path, default=default_input,
                        help=f"Combined source file (default: {default_input})")
    parser.add_argument("--locale", default="zh", help="Locale code (default: zh)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the tally without writing the output file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    records = load_jsonl(args.input)
    converted = [convert(r, args.locale) for r in records]

    tally = {"full": 0, "query": 0}
    for rec in converted:
        tally[rec["localization_level"]] += 1

    out_file = args.input.parent / f"{args.locale}_translatable.jsonl"
    print(f"Input:  {args.input.name}")
    print(f"Output: {out_file.name}{'  (dry run)' if args.dry_run else ''}")
    print(f"Converted {len(converted)} record(s).")
    print(f"localization_level tally:  full (Chinese params) = {tally['full']}, "
          f"query (English params) = {tally['query']}")

    if not args.dry_run:
        with out_file.open("w", encoding="utf-8") as f:
            for rec in converted:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
