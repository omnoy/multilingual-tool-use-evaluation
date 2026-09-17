"""Find entries that are identical between zh_translatable_full and zh_translatable_query.

A full-localization record (Chinese params) and its query-localization counterpart
(English params) should differ in ground_truth and/or message text. Any source_id
whose `messages` AND `ground_truth` are byte-identical across the two files is
suspect: it means the conversion left that entry untouched in one direction.

Reports, per source_id, whether messages match, ground_truth matches, or both.

Usage:
    python find_identical_full_query.py [--dir data/benchmarks/wildtoolbench/zh]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_by_source_id(path: Path) -> dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["source_id"]] = rec
    return out


def canon(value) -> str:
    """Stable canonical form for comparison (order-sensitive, whitespace-exact)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = (Path(__file__).resolve().parent / "data" / "benchmarks"
                   / "wildtoolbench" / "zh")
    parser.add_argument("--dir", type=Path, default=default_dir)
    parser.add_argument("--full", default="zh_translatable_full.jsonl")
    parser.add_argument("--query", default="zh_translatable_query.jsonl")
    args = parser.parse_args()

    full_path = args.dir / args.full
    query_path = args.dir / args.query
    for p in (full_path, query_path):
        if not p.exists():
            print(f"error: not found: {p}", file=sys.stderr)
            return 1

    full = load_by_source_id(full_path)
    query = load_by_source_id(query_path)

    only_full = sorted(set(full) - set(query))
    only_query = sorted(set(query) - set(full))
    if only_full:
        print(f"WARN source_ids only in full: {only_full}")
    if only_query:
        print(f"WARN source_ids only in query: {only_query}")

    common = set(full) & set(query)

    def num(sid: str) -> int:
        digits = "".join(c for c in sid if c.isdigit())
        return int(digits) if digits else 0

    identical_both = []
    same_msg_only = []
    same_gt_only = []
    for sid in sorted(common, key=num):
        f, q = full[sid], query[sid]
        msg_same = canon(f["messages"]) == canon(q["messages"])
        gt_same = canon(f["ground_truth"]) == canon(q["ground_truth"])
        if msg_same and gt_same:
            identical_both.append(sid)
        elif msg_same:
            same_msg_only.append(sid)
        elif gt_same:
            same_gt_only.append(sid)

    print(f"\nComparing {len(common)} shared source_id(s).\n")
    print(f"IDENTICAL in BOTH message and ground_truth ({len(identical_both)}):")
    for sid in identical_both:
        gt = json.dumps(full[sid]["ground_truth"], ensure_ascii=False)
        lvl = full[sid].get("localization_level")
        print(f"  {sid}  (orig? gt={gt})")

    print(f"\nSame MESSAGE only, ground_truth differs ({len(same_msg_only)}):")
    for sid in same_msg_only:
        print(f"  {sid}")

    print(f"\nSame GROUND_TRUTH only, message differs ({len(same_gt_only)}):")
    for sid in same_gt_only:
        fg = json.dumps(full[sid]["ground_truth"], ensure_ascii=False)
        print(f"  {sid}  gt={fg}")

    print(f"\nSummary: {len(identical_both)} fully identical, "
          f"{len(same_msg_only)} same-message-only, "
          f"{len(same_gt_only)} same-ground_truth-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
