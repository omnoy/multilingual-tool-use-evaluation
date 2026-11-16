"""Rename a benchmark inside file contents under a results folder (default: results_robustness).

Replaces the old name inside text files (summary.json "category", summary_table.csv, ...),
skipping the top-level "old" directory. Directory/file names are NOT renamed.

Nothing is changed unless --apply is passed; the default is a dry run that reports
what would change.

Usage:
    python scripts/rename_benchmark.py                 # dry run, summary only
    python scripts/rename_benchmark.py --verbose       # dry run, list every file/line
    python scripts/rename_benchmark.py --apply         # actually do it
    python scripts/rename_benchmark.py --old X --new Y --root some_dir --exclude old --exclude tmp
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OLD = "bfcl_multiple_fixed_desc_gt"
DEFAULT_NEW = "bfcl_multiple"
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"}


def name_pattern(old: str) -> re.Pattern[str]:
    # Match the name only as a whole token, so e.g. "bfcl_multiple_fixed_desc_gt2" is left alone.
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])")


def is_excluded(path: Path, root: Path, excludes: set[str]) -> bool:
    rel = path.relative_to(root)
    return bool(rel.parts) and rel.parts[0] in excludes


def collect(root: Path, pat: re.Pattern[str], excludes: set[str]):
    edits: dict[Path, int] = {}      # file -> number of occurrences in content
    skipped_binary: list[Path] = []
    for p in sorted(root.rglob("*")):
        if is_excluded(p, root, excludes) or not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary.append(p)
            continue
        n = len(pat.findall(text))
        if n:
            edits[p] = n
    return edits, skipped_binary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=DEFAULT_OLD)
    ap.add_argument("--new", default=DEFAULT_NEW)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "results_robustness")
    ap.add_argument("--exclude", action="append", default=None,
                    help="Top-level subdirectory of --root to skip (repeatable; default: old)")
    ap.add_argument("--apply", action="store_true", help="Actually perform the changes (default: dry run)")
    ap.add_argument("--verbose", "-v", action="store_true", help="List every affected file and line")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    excludes = set(args.exclude) if args.exclude is not None else {"old"}
    pat = name_pattern(args.old)
    if not root.is_dir():
        print(f"Root not found: {root}", file=sys.stderr)
        return 1

    edits, skipped = collect(root, pat, excludes)
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] '{args.old}' -> '{args.new}' under {root} (excluding: {', '.join(sorted(excludes)) or 'nothing'})\n")

    by_top = Counter()
    for f, n in edits.items():
        by_top[f.relative_to(root).parts[0]] += n
    print(f"Files with content changes: {len(edits)}  (total occurrences: {sum(edits.values())})")
    for top, n in sorted(by_top.items()):
        n_files = sum(1 for f in edits if f.relative_to(root).parts[0] == top)
        print(f"  {top}: {n_files} files, {n} occurrences")
    if args.verbose:
        for f, n in edits.items():
            print(f"    {f.relative_to(root)}  ({n})")
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    print(f"      L{i}: {line.strip()[:160]}")
    if skipped:
        print(f"\nSkipped (not UTF-8): {len(skipped)}")
        for f in skipped:
            print(f"  {f.relative_to(root)}")

    # Warn about rows in summary tables that already use the new name (would become ambiguous).
    for f in edits:
        text = f.read_text(encoding="utf-8")
        existing = len(name_pattern(args.new).findall(text))
        if existing:
            print(f"\nWARNING: {f.relative_to(root)} already contains {existing} occurrence(s) of '{args.new}'. "
                  f"After the rename those can't be told apart from the renamed ones. If it's a generated file "
                  f"(e.g. summary_table.csv), consider regenerating it instead.")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to perform these changes.")
        return 0

    for f in edits:
        raw = f.read_bytes().decode("utf-8")  # bytes round-trip preserves line endings
        f.write_bytes(pat.sub(args.new, raw).encode("utf-8"))
    print(f"\nDone: edited {len(edits)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
