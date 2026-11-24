"""
remove_robustness_entries.py — Delete entries' transcripts from robustness runs, across languages.

Matches results_robustness/<model>/<category>/<lang-dir>/<benchmark>/transcripts/<entry>.json
across all models, categories and benchmarks. Archived runs under results_robustness/old
sit one level deeper and are not matched.

Entry ids can be given in two forms:
  * a base id (multiple_2) matches the entry in every language: multiple_2.json under en,
    multiple_2_de.json under de, multiple_2_he.json under he, and so on;
  * a localized id (multiple_2_de) matches only that file, so only that language.
Matching is exact, so multiple_2 never touches multiple_21 or multiple_21_de.

By default every language folder is searched; --lang-dir narrows it to one or more.

Only previews by default; pass --apply to delete.

Deleting a transcript does not update statistics.csv or summary.json. Rebuild those
afterwards with run_robustness_sweep.py --recompute-stats (the script prints the command
for each affected language).

Usage:
    # Preview removing multiple_2 and multiple_7 from every language:
    python scripts/remove_robustness_entries.py --entry multiple_2 multiple_7

    # Only from German and Hebrew, and actually delete:
    python scripts/remove_robustness_entries.py --entry multiple_2 --lang-dir de he --apply

    # A single localized entry:
    python scripts/remove_robustness_entries.py --entry multiple_2_de --apply
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def candidate_names(entry: str, lang_dir: str) -> set[str]:
    """File names under <lang_dir> that the given id refers to."""
    return {f"{entry}.json", f"{entry}_{lang_dir}.json"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete entries' transcripts from robustness runs, across languages.",
    )
    parser.add_argument("--entry", required=True, nargs="+",
                        help="Entry ids: base (multiple_2, all languages) or localized "
                             "(multiple_2_de, that language only).")
    parser.add_argument("--lang-dir", nargs="+", default=None,
                        help="Language subfolders to search, e.g. de he. Default: all.")
    parser.add_argument("--output-dir", default="results_robustness",
                        help="Results root (relative to the package root).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete the files. Without this, only list them.")
    args = parser.parse_args()

    entries = []
    for raw in args.entry:
        entry = raw.removesuffix(".json")
        if not entry or entry != Path(entry).name or entry in {".", ".."}:
            parser.error(f"--entry must be a plain entry id, not a path: {raw!r}")
        entries.append(entry)

    root = PACKAGE_ROOT / args.output_dir
    transcript_dirs = [d for d in root.glob("*/*/*/*/transcripts") if d.is_dir()]
    if args.lang_dir:
        wanted = set(args.lang_dir)
        transcript_dirs = [d for d in transcript_dirs if d.parent.parent.name in wanted]

    matches: list[Path] = []
    found_ids: set[str] = set()
    for tdir in sorted(transcript_dirs):
        lang = tdir.parent.parent.name
        for entry in entries:
            for name in sorted(candidate_names(entry, lang)):
                path = tdir / name
                if path.is_file():
                    matches.append(path)
                    found_ids.add(entry)

    scope = ", ".join(args.lang_dir) if args.lang_dir else "all languages"
    if not matches:
        print(f"No transcripts for {', '.join(entries)} ({scope}) under {root}.")
        return

    per_lang: dict[str, int] = defaultdict(int)
    for path in matches:
        rel = path.relative_to(root)
        per_lang[rel.parts[2]] += 1
        if args.apply:
            path.unlink()
        print(f"{'deleted' if args.apply else 'would delete'}: {rel}")

    print("\nPer language: " + ", ".join(f"{lang}={n}" for lang, n in sorted(per_lang.items())))
    missing = [e for e in entries if e not in found_ids]
    if missing:
        print(f"Warning: no transcripts found for: {', '.join(missing)}")

    if args.apply:
        print(f"\nDeleted {len(matches)} file(s). Rebuild the stats with:")
        for lang in sorted(per_lang):
            print(f"  python scripts/run_robustness_sweep.py --lang-dir {lang} --recompute-stats")
    else:
        print(f"\n{len(matches)} file(s) matched. Nothing deleted; pass --apply to delete them.")


if __name__ == "__main__":
    main()
