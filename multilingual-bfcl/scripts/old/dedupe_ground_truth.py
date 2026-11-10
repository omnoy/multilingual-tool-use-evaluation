"""
dedupe_ground_truth.py — Copy the benchmarks folder, dropping duplicate ground-truth candidates.

Translation often maps several English candidates to the same target-language string, e.g.

    "platform": ["פלייסטיישן", "פלייסטיישן"]

This script writes an exact copy of the source folder in which every parameter's candidate
list keeps only the first occurrence of each value (order preserved). Two candidates count
as duplicates only when they are exactly equal as JSON values (no case or whitespace
normalization). Dict-valued candidates (BFCL nested parameters such as
{"min": [300000], "max": [400000]}) are deduplicated recursively. List-valued candidates
(array parameters) are compared as whole values, never deduplicated internally, since
["Lakers", "Lakers"] could be a legitimate array.

Only *.jsonl files are processed. Lines whose ground truth is unchanged are written back
byte-for-byte; changed lines are re-serialized with json.dumps(ensure_ascii=False). Every
other file (.json manifests, .tsv, caches) is copied verbatim, as are lines that fail to
parse (reported as warnings). The backups/ subfolder is copied verbatim unless
--include-backups is given.

The source folder is never modified.

Usage:
    # Preview what would be removed (writes nothing):
    python scripts/dedupe_ground_truth.py --dry-run

    # Write data/benchmarks:
    python scripts/dedupe_ground_truth.py

    # Custom paths, list every removed value:
    python scripts/dedupe_ground_truth.py --src data/benchmarks --dst data/benchmarks_clean --verbose
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Removal:
    file: str
    entry_id: str
    path: str  # e.g. game_rewards.get.platform
    original: list
    kept: list


@dataclass
class Stats:
    removals: list[Removal] = field(default_factory=list)
    entries_changed: Counter = field(default_factory=Counter)  # file -> n entries
    entries_total: Counter = field(default_factory=Counter)
    bad_lines: list[tuple[str, int]] = field(default_factory=list)


def _key(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def dedupe_params(params: dict, prefix: str, sink: list[tuple[str, list, list]]) -> dict:
    """Dedupe each param's candidate list; recurse into dict-valued candidates."""
    out = {}
    for name, candidates in params.items():
        path = f"{prefix}.{name}"
        if not isinstance(candidates, list):
            out[name] = candidates
            continue
        processed = [
            dedupe_params(c, path, sink) if isinstance(c, dict) else c for c in candidates
        ]
        seen: set[str] = set()
        kept = []
        for c in processed:
            k = _key(c)
            if k not in seen:
                seen.add(k)
                kept.append(c)
        if len(kept) != len(processed):
            sink.append((path, processed, kept))
        out[name] = kept
    return out


def dedupe_ground_truth(gt, sink):
    if not isinstance(gt, list):
        return gt
    new_gt = []
    for call in gt:
        if isinstance(call, dict):
            new_gt.append({
                fn: dedupe_params(params, fn, sink) if isinstance(params, dict) else params
                for fn, params in call.items()
            })
        else:
            new_gt.append(call)
    return new_gt


def process_jsonl(src: Path, dst: Path | None, rel: str, stats: Stats) -> None:
    out_lines = []
    with src.open(encoding="utf-8", newline="") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                out_lines.append(line)
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                stats.bad_lines.append((rel, lineno))
                out_lines.append(line)
                continue
            if not isinstance(entry, dict) or "ground_truth" not in entry:
                out_lines.append(line)
                continue
            stats.entries_total[rel] += 1
            sink: list[tuple[str, list, list]] = []
            entry["ground_truth"] = dedupe_ground_truth(entry["ground_truth"], sink)
            if not sink:
                out_lines.append(line)
                continue
            stats.entries_changed[rel] += 1
            entry_id = str(entry.get("id", f"line {lineno}"))
            for path, original, kept in sink:
                stats.removals.append(Removal(rel, entry_id, path, original, kept))
            ending = line[len(stripped):]
            out_lines.append(json.dumps(entry, ensure_ascii=False) + ending)
    if dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8", newline="") as f:
            f.writelines(out_lines)


def fmt(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def report(stats: Stats, verbose: bool) -> None:
    removals = stats.removals
    n_values = sum(len(r.original) - len(r.kept) for r in removals)

    print("\n=== Per file ===")
    print(f"{'file':<70} {'entries':>8} {'changed':>8} {'params':>7} {'removed':>8}")
    for rel in sorted(stats.entries_total):
        file_rem = [r for r in removals if r.file == rel]
        n_rem = sum(len(r.original) - len(r.kept) for r in file_rem)
        print(f"{rel:<70} {stats.entries_total[rel]:>8} {stats.entries_changed[rel]:>8} "
              f"{len(file_rem):>7} {n_rem:>8}")

    print("\n=== Totals ===")
    print(f"files processed:        {len(stats.entries_total)}")
    print(f"entries with ground_truth: {sum(stats.entries_total.values())}")
    print(f"entries changed:        {sum(stats.entries_changed.values())}")
    print(f"parameters changed:     {len(removals)}")
    print(f"candidate values removed: {n_values}")

    dup_counter: Counter = Counter()
    for r in removals:
        counts = Counter(_key(c) for c in r.original)
        for k, c in counts.items():
            if c > 1:
                dup_counter[k] += c - 1
    if dup_counter:
        print("\n=== Most frequently removed values ===")
        for k, c in dup_counter.most_common(25):
            print(f"{c:>5}  {k}")

    if removals:
        print("\n=== Removed duplicates ===" + ("" if verbose else " (first 40; --verbose for all)"))
        shown = removals if verbose else removals[:40]
        for r in shown:
            print(f"[{r.file}] {r.entry_id}  {r.path}")
            print(f"    before: {fmt(r.original)}")
            print(f"    after:  {fmt(r.kept)}")
        if not verbose and len(removals) > 40:
            print(f"... {len(removals) - 40} more")

    if stats.bad_lines:
        print("\n=== WARNING: unparseable lines (copied unchanged) ===")
        for rel, lineno in stats.bad_lines:
            print(f"  {rel}:{lineno}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the benchmarks folder without duplicate ground-truth candidates.")
    parser.add_argument("--src", default="data/benchmarks",
                        help="Source folder (relative to the package root).")
    parser.add_argument("--dst", default="data/benchmarks",
                        help="Destination folder (relative to the package root).")
    parser.add_argument("--include-backups", action="store_true",
                        help="Also dedupe files under backups/ (default: copy them verbatim).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the destination folder if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="List every removal.")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    src = (PACKAGE_ROOT / args.src).resolve()
    dst = (PACKAGE_ROOT / args.dst).resolve()
    if not src.is_dir():
        sys.exit(f"Source folder not found: {src}")
    if dst == src or src in dst.parents:
        sys.exit("Destination must not be the source folder or inside it.")
    if not args.dry_run and dst.exists():
        if not args.overwrite:
            sys.exit(f"Destination exists: {dst} (pass --overwrite to replace it)")
        shutil.rmtree(dst)

    stats = Stats()
    copied = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(src)
        rel = rel_path.as_posix()
        out = None if args.dry_run else dst / rel_path
        in_backups = rel_path.parts[0] == "backups"
        if path.suffix == ".jsonl" and (args.include_backups or not in_backups):
            process_jsonl(path, out, rel, stats)
        elif out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            copied += 1

    print(f"source:      {src}")
    print(f"destination: {'(dry run, nothing written)' if args.dry_run else dst}")
    if not args.dry_run:
        print(f"files copied verbatim: {copied}")
    report(stats, args.verbose)


if __name__ == "__main__":
    main()
