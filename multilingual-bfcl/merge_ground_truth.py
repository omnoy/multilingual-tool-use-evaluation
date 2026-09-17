"""Merge ground_truth from possible_answer files into their benchmark files.

For every benchmark file at ``<category>/<locale>/<name>.json`` under the
benchmarks root, this looks up the matching possible-answer file at
``<category>/possible_answer/<locale>/<name>.json``, adds each entry's
``ground_truth`` (keyed by ``id``) onto the corresponding benchmark object,
and writes the result to ``<category>/<locale>/<name>.jsonl``.

The source files are JSON-lines (one object per line) despite the .json
extension. Originals and the possible_answer/ subtrees are left untouched.

Usage:
    python merge_ground_truth.py [--benchmarks-dir data/benchmarks] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

POSSIBLE_ANSWER_DIRNAME = "possible_answer"

# Trailing language-code suffix, e.g. the "_he" in "multiple_2_he". Stripped as a
# fallback so a benchmark id ("multiple_2_he") can match a possible_answer whose
# ids were left in canonical form ("multiple_2"), as happens for the query files.
_LANG_SUFFIX_RE = re.compile(r"_[A-Za-z]+$")


def normalize_id(rec_id: str) -> str:
    return _LANG_SUFFIX_RE.sub("", rec_id)


def load_jsonl(path: Path) -> list[dict]:
    """Read a file of one JSON object per line, ignoring blank lines."""
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


def find_benchmark_files(benchmarks_dir: Path) -> list[Path]:
    """All .json benchmark files, excluding anything under a possible_answer/ dir."""
    files = []
    for path in sorted(benchmarks_dir.rglob("*.json")):
        if POSSIBLE_ANSWER_DIRNAME in path.relative_to(benchmarks_dir).parts:
            continue
        # Skip manifest/config files at the category root (e.g. batch_manifest.json).
        # Benchmark files live one level deeper, under a locale subfolder.
        rel_parts = path.relative_to(benchmarks_dir).parts
        if len(rel_parts) < 3:
            continue
        files.append(path)
    return files


def possible_answer_path_for(benchmark_file: Path, benchmarks_dir: Path) -> Path:
    """Map <category>/<locale>/<name>.json -> <category>/possible_answer/<locale>/<name>.json."""
    rel = benchmark_file.relative_to(benchmarks_dir)
    category, *rest = rel.parts  # rest = [locale, ..., name.json]
    return benchmarks_dir.joinpath(category, POSSIBLE_ANSWER_DIRNAME, *rest)


def merge_file(benchmark_file: Path, benchmarks_dir: Path, dry_run: bool) -> bool:
    """Merge one benchmark file with its possible_answer. Returns True on success."""
    pa_file = possible_answer_path_for(benchmark_file, benchmarks_dir)
    rel = benchmark_file.relative_to(benchmarks_dir)

    if not pa_file.exists():
        print(f"  SKIP  {rel}: no matching possible_answer at "
              f"{pa_file.relative_to(benchmarks_dir)}")
        return False

    benchmarks = load_jsonl(benchmark_file)
    answers = load_jsonl(pa_file)

    gt_by_id: dict[str, object] = {}
    gt_by_norm: dict[str, object] = {}
    for rec in answers:
        rec_id = rec.get("id")
        if rec_id is None:
            print(f"  WARN  {pa_file.name}: possible_answer entry without 'id'")
            continue
        if "ground_truth" not in rec:
            print(f"  WARN  {pa_file.name}: id={rec_id!r} has no 'ground_truth'")
            continue
        gt_by_id[rec_id] = rec["ground_truth"]
        gt_by_norm.setdefault(normalize_id(rec_id), rec["ground_truth"])

    merged = []
    missing = 0
    fuzzy = 0
    for rec in benchmarks:
        rec_id = rec.get("id")
        if rec_id in gt_by_id:
            gt = gt_by_id[rec_id]
        elif normalize_id(rec_id) in gt_by_norm:
            gt = gt_by_norm[normalize_id(rec_id)]
            fuzzy += 1
        else:
            gt = None
            missing += 1
            print(f"  WARN  {rel}: id={rec_id!r} has no matching ground_truth")
        if gt is not None:
            # Append ground_truth as a new key, preserving existing field order.
            rec = {**rec, "ground_truth": gt}
        merged.append(rec)

    out_file = benchmark_file.with_suffix(".jsonl")
    matched = len(merged) - missing
    fuzzy_note = f", {fuzzy} via suffix-strip" if fuzzy else ""
    print(f"  OK    {rel} -> {out_file.name}  "
          f"({matched}/{len(merged)} matched{fuzzy_note})")

    if not dry_run:
        with out_file.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = Path(__file__).resolve().parent / "data" / "benchmarks"
    parser.add_argument("--benchmarks-dir", type=Path, default=default_dir,
                        help=f"Benchmarks root (default: {default_dir})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written without writing files")
    args = parser.parse_args()

    benchmarks_dir: Path = args.benchmarks_dir
    if not benchmarks_dir.is_dir():
        print(f"error: benchmarks dir not found: {benchmarks_dir}", file=sys.stderr)
        return 1

    benchmark_files = find_benchmark_files(benchmarks_dir)
    if not benchmark_files:
        print(f"No benchmark .json files found under {benchmarks_dir}")
        return 0

    print(f"Benchmarks root: {benchmarks_dir}")
    print(f"Found {len(benchmark_files)} benchmark file(s)"
          f"{' (dry run)' if args.dry_run else ''}:")
    processed = 0
    for bf in benchmark_files:
        if merge_file(bf, benchmarks_dir, args.dry_run):
            processed += 1

    print(f"\nDone: {processed}/{len(benchmark_files)} file(s) merged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
