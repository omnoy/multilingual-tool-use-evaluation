"""Rewrite each entry's `localization_level` field to its experiment name.

    query                          -> native_query_eng_target
    full                           -> native_query_native_target   (or native_qt_lang_hints, see below)
    full_and_function_descriptions -> native_qt_func_descs
    full_full_full                 -> native_qt_all_cues
    function_descriptions          -> native_query_func_descs
    (a trailing "_shuffled" is kept)

The new value is taken from the file's own name when it is a known experiment
(he/he_native_qt_lang_hints.jsonl -> native_qt_lang_hints), or from the nearest parent folder that
is one (results .../he_native_qt_all_cues/transcripts/x.json -> native_qt_all_cues). This matters
because *_native_qt_lang_hints files were written with "full", same as native_query_native_target.
Files whose path names no experiment (e.g. zh_translatable.jsonl) fall back to mapping each value
via experiment_names.LEGACY_LEVEL_VALUES.

In .json/.jsonl files only the `"localization_level": "..."` text is replaced, so formatting, key
order and line endings are untouched. In .csv/.tsv files (e.g. results statistics.csv) the
`localization_level` column is rewritten; a file is skipped (and reported) unless it round-trips
byte-for-byte through the csv module. Running it twice is a no-op.

Nothing is changed unless --apply is passed.

Usage (from multilingual-bfcl/):
    python scripts/relabel_localization_levels.py                       # dry run on data/benchmarks
    python scripts/relabel_localization_levels.py --apply
    python scripts/relabel_localization_levels.py --root results_robustness --exclude old --apply
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from multilingual_bfcl.experiment_names import EXPERIMENT_ORDER, normalize_level_value  # noqa: E402

FIELD_RE = re.compile(r'("localization_level"\s*:\s*")([^"]*)(")')
LOCALE_PREFIX_RE = re.compile(r"^[a-z]{2}(?:-[A-Za-z]{2,4})?_")
KNOWN = set(EXPERIMENT_ORDER)


def experiment_from_name(name: str) -> str | None:
    """'he_native_qt_lang_hints' / 'he_native_query_native_target_shuffled' -> experiment, else None.

    A trailing "_old" (archived runs, e.g. he_native_qt_lang_hints_old) is ignored.
    """
    rest = LOCALE_PREFIX_RE.sub("", name, count=1)
    if rest == name:
        return None
    if rest.endswith("_old"):
        rest = rest[: -len("_old")]
    base = rest[: -len("_shuffled")] if rest.endswith("_shuffled") else rest
    return rest if base in KNOWN else None


def experiment_from_path(path: Path, root: Path) -> str | None:
    for part in [path.stem, *(p.name for p in path.relative_to(root).parents if p.name)][: 4]:
        exp = experiment_from_name(part)
        if exp:
            return exp
    return None


def _write_table(rows: list[list[str]], delimiter: str, lineterminator: str) -> str:
    buf = io.StringIO()
    csv.writer(buf, delimiter=delimiter, lineterminator=lineterminator).writerows(rows)
    return buf.getvalue()


def relabel_table(text: str, delimiter: str, relabel) -> str | None:
    """Relabel the `localization_level` column of a CSV/TSV.

    Returns the new text, the unchanged text if there is no such header column, or None if the
    file doesn't round-trip byte-for-byte through the csv module (then it is left alone).
    """
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not rows or "localization_level" not in rows[0]:
        return text
    lineterminator = "\r\n" if "\r\n" in text else "\n"
    if _write_table(rows, delimiter, lineterminator) != text:
        return None
    col = rows[0].index("localization_level")
    for row in rows[1:]:
        if col < len(row):
            row[col] = relabel(row[col])
    return _write_table(rows, delimiter, lineterminator)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append",
                    help="Folder to scan (repeatable). Default: data/benchmarks. Relative to multilingual-bfcl/.")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Folder name to skip anywhere under the root (repeatable), e.g. old.")
    ap.add_argument("--apply", action="store_true", help="Actually write the changes (default: dry run).")
    ap.add_argument("--verbose", "-v", action="store_true", help="List every changed file.")
    args = ap.parse_args()
    roots = [(r if r.is_absolute() else PACKAGE_ROOT / r).resolve()
             for r in (args.root or [Path("data/benchmarks")])]
    skip = set(args.exclude) | {".git", "__pycache__", ".ipynb_checkpoints"}

    changes: dict[Path, tuple[str, int]] = {}
    by_pair: Counter = Counter()
    by_source: Counter = Counter()
    unmapped: Counter = Counter()
    csv_skipped: list[Path] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for name in filenames:
                if not name.endswith((".jsonl", ".json", ".csv", ".tsv")):
                    continue
                p = Path(dirpath) / name
                text = p.read_bytes().decode("utf-8")
                if "localization_level" not in text:
                    continue
                from_path = experiment_from_path(p, root)
                n = 0

                def relabel(old: str) -> str:
                    nonlocal n
                    if not old:
                        return old  # e.g. en_baseline rows have no level
                    new = from_path or normalize_level_value(old)
                    if from_path is None and new == old and old.removesuffix("_shuffled") not in KNOWN:
                        unmapped[f"{old}  ({p.relative_to(root.parent)})"] += 1
                    if new != old:
                        n += 1
                        by_pair[f"{old} -> {new}"] += 1
                        by_source["file/folder name" if from_path else "value map"] += 1
                    return new

                if name.endswith((".csv", ".tsv")):
                    new_text = relabel_table(text, "\t" if name.endswith(".tsv") else ",", relabel)
                    if new_text is None:
                        csv_skipped.append(p)
                        continue
                else:
                    new_text = FIELD_RE.sub(lambda m: m.group(1) + relabel(m.group(2)) + m.group(3), text)
                if new_text != text:
                    changes[p] = (new_text, n)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] roots: {', '.join(str(r) for r in roots)}"
          + (f"  (skipping folders: {', '.join(sorted(args.exclude))})" if args.exclude else ""))
    print(f"\n{len(changes)} files, {sum(n for _, n in changes.values())} entries")
    for k, v in sorted(by_pair.items()):
        print(f"    {k:70s} {v:>7}")
    print("  new value taken from:")
    for k, v in sorted(by_source.items()):
        print(f"    {k:70s} {v:>7}")
    if args.verbose:
        print("\nFiles:")
        for p, (_, n) in sorted(changes.items()):
            print(f"    {p}  ({n})")
    if csv_skipped:
        print(f"\nCSV/TSV files skipped (don't round-trip exactly through the csv module): {len(csv_skipped)}")
        for p in csv_skipped:
            print(f"    {p}")
    if unmapped:
        print("\nValues with no known experiment (left as is):")
        for k, v in sorted(unmapped.items()):
            print(f"    {k}  x{v}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write these changes.")
        return 0
    for p, (new_text, _) in changes.items():
        p.write_bytes(new_text.encode("utf-8"))
    print(f"\nDone: rewrote {len(changes)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
