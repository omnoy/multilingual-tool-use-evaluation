"""
move_old_results.py — move superseded result folders (named *_old) into
results_robustness/old/, keeping their relative folder structure.

  results_robustness/<model>/<benchmark>/<locale>/<stem>_old
    -> results_robustness/old/<model>/<benchmark>/<locale>/<stem>_old

Parent folders that already exist under old/ are merged into; a folder whose
destination already exists is skipped (never overwritten). Anything already
inside old/ is left alone.

Usage:
  python scripts/move_old_results.py                 # dry run: show what would move
  python scripts/move_old_results.py --apply         # do the move
  python scripts/move_old_results.py --suffix _old --results-dir results_robustness --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def find_old_dirs(results_dir: Path, suffix: str) -> list[Path]:
    """Top-most directories whose name ends with `suffix`, outside results_dir/old."""
    found: list[Path] = []
    stack = [p for p in results_dir.iterdir() if p.is_dir() and p.name != "old"]
    while stack:
        d = stack.pop()
        if d.name.endswith(suffix):
            found.append(d)  # don't descend: the whole folder moves as one
            continue
        stack.extend(p for p in d.iterdir() if p.is_dir())
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=PACKAGE_ROOT / "results_robustness")
    parser.add_argument("--suffix", default="_old",
                        help="Move directories whose name ends with this (default: _old).")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the move (default is a dry run).")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    old_root = results_dir / "old"
    moved = skipped = 0
    for src in find_old_dirs(results_dir, args.suffix):
        rel = src.relative_to(results_dir)
        dst = old_root / rel
        if dst.exists():
            print(f"SKIP (exists)  {rel}  ->  old/{rel.as_posix()}")
            skipped += 1
            continue
        print(f"{'MOVE' if args.apply else 'would move'}  {rel.as_posix()}  ->  old/{rel.as_posix()}")
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1

    verb = "moved" if args.apply else "to move (dry run; pass --apply)"
    print(f"\n{moved} folder(s) {verb}, {skipped} skipped.")


if __name__ == "__main__":
    main()
