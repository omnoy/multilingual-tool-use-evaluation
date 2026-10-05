"""
merge_full_and_function_descriptions.py — build the fully-translated benchmark.

Combines two existing Hebrew benchmarks into a single "everything translated"
condition (no API calls — pure deterministic merge):

  - function/parameter DESCRIPTIONS  <- --desc-source  (e.g. he_translatable_func_desc)
  - user QUERY (messages)            <- --query-source (e.g. he_translatable_full)
  - ground-truth parameter VALUES    <- --query-source's inline "ground_truth" field

Function names, parameter names, and types stay English (they are identical across
both sources). The result is: Hebrew query + Hebrew API descriptions + Hebrew ground
truth — the maximal language-mismatch-free "all Hebrew" baseline.

Both sources derive from the same 132 base entries and share ids (multiple_X_he), so
the merge is a straight per-id combine. Ground truth is stored inline on each entry
(there is no separate possible_answer/ file). The script verifies that the two sources'
function definitions are structurally identical (ignoring descriptions) before merging.

Output (under data/benchmarks/bfcl_multiple/):
  <locale>/<output>.jsonl

Usage:
    python scripts/merge_full_and_function_descriptions.py --locale he         --desc-source he_translatable_func_desc --query-source he_translatable_full         --output he_translatable_full_func_desc
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks" / "bfcl_multiple"

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
    parser.add_argument("--locale", required=True,
                        help="Language for translated function definitions.")
    parser.add_argument("--desc-source", required=True,
                        help="Benchmark stem (under <locale>/) providing translated function defs.")
    parser.add_argument("--query-source", required=True,
                        help="Benchmark stem (under <locale>/) providing the translated query "
                             "and its inline translated ground truth.")
    parser.add_argument("--output", required=True,
                        help="Output benchmark stem (under <locale>/).")
    parser.add_argument("--level", default=DEFAULT_LEVEL,
                        help="localization_level tag written onto each merged entry.")
    args = parser.parse_args()

    desc = _index(_load_jsonl(DATA_ROOT / args.locale / f"{args.desc_source}.jsonl"))
    query = _index(_load_jsonl(DATA_ROOT / args.locale / f"{args.query_source}.jsonl"))

    ids = [i for i in desc if i in query]  # preserve desc-source order
    missing = sorted(set(desc) ^ set(query))
    if missing:
        print(f"[WARN] {len(missing)} id(s) not in both sources; using intersection "
              f"({len(ids)} entries).", file=sys.stderr)

    out_entries: list[dict] = []
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

        gt = q_entry.get("ground_truth")          # translated ground-truth values
        if gt is None:
            missing_gt += 1
            print(f"[WARN] {entry_id}: no ground_truth in query source; skipping.",
                  file=sys.stderr)
            continue

        merged = copy.deepcopy(d_entry)           # keeps translated-description functions
        merged["messages"] = q_entry["messages"]  # translated query
        merged["ground_truth"] = gt
        merged["localization_level"] = args.level
        out_entries.append(merged)

    out_bench = DATA_ROOT / args.locale / f"{args.output}.jsonl"
    _write_jsonl(out_bench, out_entries)

    print(f"wrote {len(out_entries)} entries -> {out_bench}")
    if struct_mismatch:
        print(f"[WARN] skipped {struct_mismatch} entries with mismatched function structure",
              file=sys.stderr)
    if missing_gt:
        print(f"[WARN] skipped {missing_gt} entries with no ground truth", file=sys.stderr)


if __name__ == "__main__":
    main()
