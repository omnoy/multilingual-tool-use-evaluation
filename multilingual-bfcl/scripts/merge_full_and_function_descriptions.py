"""
merge_full_and_function_descriptions.py — build the fully-translated benchmark.

Combines two existing Hebrew benchmarks into a single "everything translated"
condition (no API calls — pure deterministic merge):

  - function/parameter DESCRIPTIONS  <- he_translated_function_descriptions.json
  - user QUERY                       <- he_translatable_full.json
  - ground-truth parameter VALUES    <- possible_answer/heb/he_translatable_full.json

Function names, parameter names, and types stay English (they are identical across
both sources). The result is: Hebrew query + Hebrew API descriptions + Hebrew ground
truth — the maximal language-mismatch-free "all Hebrew" baseline.

Both sources derive from the same 132 base entries and share ids (multiple_X_he), so
the merge is a straight per-id combine. The script verifies that the two sources'
function definitions are structurally identical (ignoring descriptions) before merging.

Outputs (under data/benchmarks/multiple/):
  heb/he_translated_full_and_function_descriptions.json
  possible_answer/heb/he_translated_full_and_function_descriptions.json

Usage:
    python scripts/merge_full_and_function_descriptions.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks" / "multiple"

DEFAULT_DESC_SOURCE = "he_translated_function_descriptions"   # Hebrew descriptions
DEFAULT_QUERY_SOURCE = "he_translatable_full"                 # Hebrew query + Hebrew GT
DEFAULT_OUTPUT = "he_translated_full_and_function_descriptions"
DEFAULT_LEVEL = "full_and_function_descriptions"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _index(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_descriptions(obj: Any) -> Any:
    """Copy of a structure with every "description" value blanked, for comparison."""
    if isinstance(obj, dict):
        return {k: ("" if k == "description" else _strip_descriptions(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_descriptions(x) for x in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge translated function descriptions with a translated query/GT benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--desc-source", default=DEFAULT_DESC_SOURCE,
                        help="Benchmark stem (under heb/) providing translated function defs.")
    parser.add_argument("--query-source", default=DEFAULT_QUERY_SOURCE,
                        help="Benchmark stem (under heb/) providing the translated query and, "
                             "via its possible_answer file, the translated ground truth.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output benchmark stem (under heb/ and possible_answer/heb/).")
    parser.add_argument("--level", default=DEFAULT_LEVEL,
                        help="localization_level tag written onto each merged entry.")
    args = parser.parse_args()

    desc = _index(_load_jsonl(DATA_ROOT / "heb" / f"{args.desc_source}.json"))
    query = _index(_load_jsonl(DATA_ROOT / "heb" / f"{args.query_source}.json"))
    query_gt = _index(_load_jsonl(
        DATA_ROOT / "possible_answer" / "heb" / f"{args.query_source}.json"))

    ids = [i for i in desc if i in query]  # preserve desc-source order
    missing = sorted(set(desc) ^ set(query))
    if missing:
        print(f"[WARN] {len(missing)} id(s) not in both sources; using intersection "
              f"({len(ids)} entries).", file=sys.stderr)

    out_entries: list[dict] = []
    out_answers: list[dict] = []
    struct_mismatch = 0
    missing_gt = 0

    for entry_id in ids:
        d_entry = desc[entry_id]
        q_entry = query[entry_id]

        # Sanity: the function definitions must be identical apart from descriptions.
        if _strip_descriptions(d_entry["function"]) != _strip_descriptions(q_entry["function"]):
            struct_mismatch += 1
            print(f"[WARN] {entry_id}: function structure differs between sources; skipping.",
                  file=sys.stderr)
            continue

        merged = copy.deepcopy(d_entry)          # keeps Hebrew-description functions
        merged["question"] = q_entry["question"]  # Hebrew query
        merged["localization_level"] = args.level
        out_entries.append(merged)

        gt = query_gt.get(entry_id)               # Hebrew ground-truth values
        if gt is None:
            missing_gt += 1
        else:
            out_answers.append({"id": entry_id, "ground_truth": gt["ground_truth"]})

    out_bench = DATA_ROOT / "heb" / f"{args.output}.json"
    out_gt = DATA_ROOT / "possible_answer" / "heb" / f"{args.output}.json"
    _write_jsonl(out_bench, out_entries)
    _write_jsonl(out_gt, out_answers)

    print(f"wrote {len(out_entries)} entries -> {out_bench}")
    print(f"wrote {len(out_answers)} answers -> {out_gt}")
    if struct_mismatch:
        print(f"[WARN] skipped {struct_mismatch} entries with mismatched function structure",
              file=sys.stderr)
    if missing_gt:
        print(f"[WARN] {missing_gt} merged entries had no matching ground truth", file=sys.stderr)


if __name__ == "__main__":
    main()
