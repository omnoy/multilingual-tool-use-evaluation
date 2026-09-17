"""Merge WildToolBench benchmark + answer files into a single bfcl_multiple-shaped jsonl.

WildToolBench stores each locale as two files in the same folder:
    <dir>/WildToolBench_<lang>.bfcl.translatable.jsonl         (id, question, function, translatable_values)
    <dir>/WildToolBench_<lang>.bfcl_answer.translatable.jsonl  (id, ground_truth)

This merges the ``ground_truth`` onto each benchmark record (keyed by id) and
writes ``<dir>/<lang>_translatable.jsonl`` normalized to the exact bfcl_multiple
schema:
    top-level:  id, question, function, ground_truth   (translatable_values dropped)
    function:   name, description, parameters
    parameters: type, properties, required             (required defaults to [])

Usage:
    python merge_wildtoolbench.py [--dir data/benchmarks/wildtoolbench/en] [--lang en] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def normalize_parameters(params: dict) -> dict:
    """Reorder to bfcl_multiple's type/properties/required, defaulting required to []."""
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


def to_bfcl_multiple(record: dict, ground_truth: object) -> dict:
    """Shape a WildToolBench record into the bfcl_multiple schema (drops extra fields)."""
    return {
        "id": record["id"],
        "question": record["question"],
        "function": [normalize_function(f) for f in record.get("function", [])],
        "ground_truth": ground_truth,
    }


def find_source_files(directory: Path, lang: str) -> tuple[Path, Path]:
    bench = directory / f"WildToolBench_{lang}.bfcl.translatable.jsonl"
    answer = directory / f"WildToolBench_{lang}.bfcl_answer.translatable.jsonl"
    return bench, answer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = (Path(__file__).resolve().parent
                   / "data" / "benchmarks" / "wildtoolbench" / "en")
    parser.add_argument("--dir", type=Path, default=default_dir,
                        help=f"Locale directory (default: {default_dir})")
    parser.add_argument("--lang", default="en",
                        help="Language code in the WildToolBench filenames (default: en)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written without writing files")
    args = parser.parse_args()

    directory: Path = args.dir
    bench_file, answer_file = find_source_files(directory, args.lang)

    for f in (bench_file, answer_file):
        if not f.exists():
            print(f"error: source file not found: {f}", file=sys.stderr)
            return 1

    benchmarks = load_jsonl(bench_file)
    answers = load_jsonl(answer_file)

    gt_by_id: dict[str, object] = {}
    for rec in answers:
        rec_id = rec.get("id")
        if rec_id is None:
            print(f"  WARN  {answer_file.name}: answer entry without 'id'")
            continue
        if "ground_truth" not in rec:
            print(f"  WARN  {answer_file.name}: id={rec_id!r} has no 'ground_truth'")
            continue
        gt_by_id[rec_id] = rec["ground_truth"]

    merged = []
    missing = 0
    for rec in benchmarks:
        rec_id = rec.get("id")
        if rec_id not in gt_by_id:
            missing += 1
            print(f"  WARN  {bench_file.name}: id={rec_id!r} has no matching ground_truth")
            continue
        merged.append(to_bfcl_multiple(rec, gt_by_id[rec_id]))

    out_file = directory / f"{args.lang}_translatable.jsonl"
    print(f"Benchmark: {bench_file.name}")
    print(f"Answers:   {answer_file.name}")
    print(f"Merged {len(merged)}/{len(benchmarks)} record(s) -> {out_file.name}"
          + (f"  ({missing} without ground_truth skipped)" if missing else "")
          + (" (dry run)" if args.dry_run else ""))

    if not args.dry_run:
        with out_file.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
