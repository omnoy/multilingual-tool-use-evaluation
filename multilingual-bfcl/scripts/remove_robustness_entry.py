"""
remove_robustness_entry.py — Delete one entry's transcript from every robustness run of a language.

Matches results_robustness/<model>/<category>/<lang-dir>/<benchmark>/transcripts/<entry>.json
across all models, categories and benchmarks. The match on the entry name is exact, so
multiple_2_de does not touch multiple_21_de. Archived runs under results_robustness/old
sit one level deeper and are not matched.

Only previews by default; pass --apply to delete. Entry ids carry the language suffix for
localized files (multiple_2_de, multiple_2_he) but not for English (multiple_2).

Deleting a transcript does not update statistics.csv or summary.json. Rebuild those
afterwards with run_robustness_sweep.py --recompute-stats.

Usage:
    # List the files that would be deleted:
    python scripts/remove_robustness_entry.py --lang-dir de --entry multiple_2_de

    # Delete them:
    python scripts/remove_robustness_entry.py --lang-dir de --entry multiple_2_de --apply
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete one entry's transcript from every robustness run of a language.",
    )
    parser.add_argument("--lang-dir", required=True, help="Language subfolder, e.g. de, he, en.")
    parser.add_argument("--entry", required=True,
                        help="Entry id (transcript file stem), e.g. multiple_2_de.")
    parser.add_argument("--output-dir", default="results_robustness",
                        help="Results root (relative to the package root).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete the files. Without this, only list them.")
    args = parser.parse_args()

    entry = args.entry.removesuffix(".json")
    if not entry or entry != Path(entry).name or entry in {".", ".."}:
        parser.error(f"--entry must be a plain entry id, not a path: {args.entry!r}")

    root = PACKAGE_ROOT / args.output_dir
    pattern = f"*/*/{glob.escape(args.lang_dir)}/*/transcripts/{glob.escape(entry)}.json"
    matches = sorted(root.glob(pattern))

    if not matches:
        print(f"No transcripts named {entry}.json for language '{args.lang_dir}' under {root}.")
        return

    for path in matches:
        if args.apply:
            path.unlink()
        print(f"{'deleted' if args.apply else 'would delete'}: {path.relative_to(root)}")

    if args.apply:
        print(f"\nDeleted {len(matches)} file(s). Rebuild the stats with:")
        print(f"  python scripts/run_robustness_sweep.py --lang-dir {args.lang_dir} --recompute-stats")
    else:
        print(f"\n{len(matches)} file(s) matched. Nothing deleted; pass --apply to delete them.")


if __name__ == "__main__":
    main()
