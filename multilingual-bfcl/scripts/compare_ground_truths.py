"""
compare_ground_truths.py — Measure how much a locale's translated ground truth
still overlaps with the original (English) ground truth.

For each entry, the ground truth of the ``<locale>_translatable_query.jsonl``
file (original-language values) is compared parameter-by-parameter with the
ground truth of ``<locale>_translatable_full.jsonl`` (translated values).
Entries are matched by ``source_id``.

Each ground-truth parameter holds a list of acceptable values. A parameter is:
    fully equal     — both acceptable-value lists contain the same values
    partially equal — the lists share at least one value, but not all
    different       — the lists share no value

Parameters with no string content anywhere (numbers, bools, ...) are never
translated, so they are reported separately and excluded from the headline
counts.

Usage:
    python scripts/compare_ground_truths.py --category bfcl_multiple --locale de
    python scripts/compare_ground_truths.py --category bfcl_multiple --locale de \
        --output results/de_gt_overlap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent           # multilingual-bfcl/
DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_string(value: Any) -> bool:
    """True if a non-empty string appears anywhere inside ``value``."""
    if isinstance(value, str):
        return value != ""
    if isinstance(value, list):
        return any(has_string(v) for v in value)
    if isinstance(value, dict):
        return any(has_string(v) for v in value.values())
    return False


def canon(value: Any) -> str:
    """Hashable canonical form of a JSON value."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def iter_params(ground_truth: list[dict[str, Any]]):
    """Yield (call_idx, func_name, param_name, acceptable_values)."""
    for call_idx, call in enumerate(ground_truth):
        for func_name, params in call.items():
            for param_name, values in params.items():
                yield call_idx, func_name, param_name, values


def classify(src_values: list[Any], tgt_values: list[Any]) -> tuple[str, list[Any]]:
    # The empty string marks an optional param; it is not a real translated value.
    src = {canon(v): v for v in src_values if v != ""}
    tgt = {canon(v): v for v in tgt_values if v != ""}
    shared = [src[k] for k in src if k in tgt]
    if src.keys() == tgt.keys():
        return "full", shared
    if shared:
        return "partial", shared
    return "different", shared


def compare(src_entries, tgt_entries) -> tuple[list[dict[str, Any]], Counter, Counter]:
    tgt_by_id = {e["source_id"]: e for e in tgt_entries}
    rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    non_string: Counter = Counter()

    for src in src_entries:
        sid = src["source_id"]
        tgt = tgt_by_id.get(sid)
        if tgt is None:
            print(f"warning: {sid} missing from target file", file=sys.stderr)
            continue
        tgt_params = {
            (ci, fn, pn): vals for ci, fn, pn, vals in iter_params(tgt["ground_truth"])
        }
        for ci, fn, pn, src_vals in iter_params(src["ground_truth"]):
            tgt_vals = tgt_params.get((ci, fn, pn))
            if tgt_vals is None:
                print(f"warning: {sid} {fn}.{pn} missing from target GT", file=sys.stderr)
                continue
            status, shared = classify(src_vals, tgt_vals)
            if not (has_string(src_vals) or has_string(tgt_vals)):
                non_string[status] += 1
                continue
            counts[status] += 1
            rows.append({
                "source_id": sid,
                "function": fn,
                "param": pn,
                "status": status,
                "source_values": src_vals,
                "target_values": tgt_vals,
                "shared_values": shared,
            })
    return rows, counts, non_string


def fmt(values: list[Any]) -> str:
    return ", ".join(json.dumps(v, ensure_ascii=False) for v in values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="bfcl_multiple")
    ap.add_argument("--locale", required=True, help="Target locale, e.g. 'de'")
    ap.add_argument("--source-level", default="query",
                    help="Localization level holding original GT (default: query)")
    ap.add_argument("--target-level", default="full",
                    help="Localization level holding translated GT (default: full)")
    ap.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = ap.parse_args()

    # Windows consoles default to a legacy codepage; force UTF-8 for non-ASCII values.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    loc_dir = DATA_ROOT / args.category / args.locale
    src_path = loc_dir / f"{args.locale}_translatable_{args.source_level}.jsonl"
    tgt_path = loc_dir / f"{args.locale}_translatable_{args.target_level}.jsonl"
    rows, counts, non_string = compare(load_jsonl(src_path), load_jsonl(tgt_path))

    total = sum(counts.values())
    pct = lambda n: f"{100 * n / total:.1f}%" if total else "n/a"
    print(f"{src_path.name}  vs  {tgt_path.name}")
    print(f"String-valued GT parameters compared: {total}")
    for status in ("full", "partial", "different"):
        print(f"  {status:<10} {counts[status]:>4}  ({pct(counts[status])})")
    print(f"Non-string parameters skipped: {sum(non_string.values())}")

    for status, title in (("full", "FULLY EQUAL"), ("partial", "PARTIALLY EQUAL")):
        print(f"\n=== {title} ({counts[status]}) ===")
        for r in rows:
            if r["status"] != status:
                continue
            print(f"{r['source_id']:<14} {r['function']}.{r['param']}")
            if status == "full":
                print(f"    values: {fmt(r['source_values'])}")
            else:
                print(f"    {args.source_level}: {fmt(r['source_values'])}")
                print(f"    {args.target_level}: {fmt(r['target_values'])}")
                print(f"    shared: {fmt(r['shared_values'])}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "source_file": str(src_path),
            "target_file": str(tgt_path),
            "counts": dict(counts),
            "non_string_counts": dict(non_string),
            "params": [r for r in rows if r["status"] != "different"],
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
