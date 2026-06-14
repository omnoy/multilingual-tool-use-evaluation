"""
filter_translatable.py — Write translatable-only copies of a benchmark.

Builds the *effective* translatability labels by merging the model output
(base_classifications_params.csv) with the human corrections overlay
(base_classifications_params.corrections.csv, overriding by id), then writes
copies of the benchmark that keep only the entries whose effective
translatable_params == "true":

    data/benchmarks/<category>/eng_base.json
        -> data/benchmarks/<category>/eng_base_translatable.json
    data/benchmarks/<category>/possible_answer/eng_base.json
        -> data/benchmarks/<category>/possible_answer/eng_base_translatable.json

Both source files are newline-delimited JSON (one object per line); the outputs
preserve that format and the original ordering.

Usage:
    python scripts/filter_translatable.py --category multiple
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent           # multilingual-bfcl/
DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks"

BASE_CSV = "base_classifications_params.csv"
OVERLAY_CSV = "base_classifications_params.corrections.csv"
TRANSLATABLE_SUFFIX = "eng_base_translatable.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a file where each line is a JSON object."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as newline-delimited JSON, matching the source format."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_translatable_params(csv_path: Path) -> dict[str, str]:
    """Read an {id: translatable_params} map from a classification CSV."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "translatable_params" not in reader.fieldnames:
            sys.exit(
                f"ERROR: {csv_path} has no 'translatable_params' column "
                f"(found {reader.fieldnames})."
            )
        for row in reader:
            mapping[row["id"]] = (row.get("translatable_params") or "").strip().lower()
    return mapping


def effective_translatable_ids(bench_dir: Path) -> set[str]:
    """
    Merge the base classifications with the corrections overlay (overlay wins per
    id) and return the set of ids whose effective translatable_params == "true".
    """
    base_path = bench_dir / BASE_CSV
    if not base_path.exists():
        sys.exit(f"ERROR: {base_path} not found.")

    labels = load_translatable_params(base_path)

    overlay_path = bench_dir / OVERLAY_CSV
    n_overlay = 0
    if overlay_path.exists():
        overlay = load_translatable_params(overlay_path)
        labels.update(overlay)
        n_overlay = len(overlay)
        print(f"Merged corrections overlay: {n_overlay} row(s) from {overlay_path.name}")
    else:
        print(f"No corrections overlay at {overlay_path.name} — using base labels only.")

    return {eid for eid, val in labels.items() if val == "true"}


def filter_file(src: Path, keep_ids: set[str], dst: Path) -> tuple[int, int]:
    """Filter a JSONL benchmark file to entries whose id is in keep_ids."""
    if not src.exists():
        sys.exit(f"ERROR: {src} not found.")
    records = load_jsonl(src)
    kept = [r for r in records if r.get("id") in keep_ids]
    write_jsonl(kept, dst)
    return len(kept), len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write translatable-only copies of a benchmark "
                    "(eng_base_translatable.json) based on the merged "
                    "translatable_params classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        default="multiple",
        help="Benchmark category folder under data/benchmarks/.",
    )
    args = parser.parse_args()

    bench_dir = DATA_ROOT / args.category
    if not bench_dir.is_dir():
        sys.exit(f"ERROR: {bench_dir} not found.")

    keep_ids = effective_translatable_ids(bench_dir)
    print(f"Translatable entries (effective): {len(keep_ids)}")

    source_src = bench_dir / "eng_base.json"
    source_dst = bench_dir / TRANSLATABLE_SUFFIX
    answer_src = bench_dir / "possible_answer" / "eng_base.json"
    answer_dst = bench_dir / "possible_answer" / TRANSLATABLE_SUFFIX

    s_kept, s_total = filter_file(source_src, keep_ids, source_dst)
    a_kept, a_total = filter_file(answer_src, keep_ids, answer_dst)

    print(f"\nWrote {s_kept}/{s_total} entries -> {source_dst}")
    print(f"Wrote {a_kept}/{a_total} entries -> {answer_dst}")

    if s_kept != a_kept:
        print(
            f"[WARN] entry count ({s_kept}) and answer count ({a_kept}) differ — "
            "some ids may be missing from one of the source files.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
