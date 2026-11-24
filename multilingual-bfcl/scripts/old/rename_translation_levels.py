"""Rename experiment names (translation levels) repo-wide.

An experiment name is "<lang><suffix>", where <lang> is a 2-letter code optionally with a region
(he, de, zh-CN) and <suffix> is the level, e.g. "_native_qt_max_context". This script maps old
suffixes to new ones, keeping the language and an optional trailing "_shuffled":

    he_native_qt_max_context           -> he_native_qt_all_cues
    he_native_qt_max_context_shuffled  -> he_native_qt_all_cues_shuffled
    native_qt_max_context              -> native_qt_all_cues   (bare experiment name, e.g. in
                                                                experiment_names.py)

The mapping is RENAMES below; override it on the command line with one or more
`--map OLD_SUFFIX:NEW_SUFFIX` (e.g. --map _native_qt_max_context:_native_qt_all_cues).

Past renames done with earlier versions of this script (for reference):
    bfcl_multiple[_fixed_desc[_gt]]    -> bfcl_multiple_translatable[...]
    en_translatable                    -> en_baseline
    <lang>_translatable_query          -> <lang>_native_query_eng_target
    <lang>_translatable_full           -> <lang>_native_query_native_target
    <lang>_translatable_full_func_desc -> <lang>_native_qt_func_descs
    <lang>_translatable_full_lang_desc -> <lang>_native_qt_lang_hints
    <lang>_translatable_full_full_full -> <lang>_native_qt_max_context
    <lang>_translatable_func_desc      -> <lang>_native_query_func_descs

Applied to file CONTENTS (text files) and to file/folder NAMES, across the whole repo
(data/, results_robustness/, old/ folders, scripts, analysis gui, ...). Skips .git, virtualenvs,
IDE folders and caches. Names are matched as whole tokens only (no letter/digit right before the
language code, nothing but an optional "_shuffled" after the suffix), so e.g.
`he_native_qt_max_context_v2` is untouched. Running it twice is a no-op as long as no new suffix
is also an old one (checked at startup).

Code that builds names on the fly (f"{lang}_native_qt_{level}") is not fixed by this script and
needs a manual edit.

Nothing is changed unless --apply is passed; the default is a dry run that reports what would change.

Usage:
    python scripts/rename_translation_levels.py              # dry run, summary
    python scripts/rename_translation_levels.py --verbose    # dry run, list every file / rename
    python scripts/rename_translation_levels.py --apply      # do it
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

SELF = Path(__file__).resolve()
REPO_ROOT = SELF.parents[2]

SKIP_DIRS = {".git", ".venv", "venv", ".idea", ".vscode", "__pycache__", "node_modules",
             ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache"}
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml",
                 ".ipynb", ".html", ".toml", ".cfg", ".ini", ".sh", ".ps1", ".bat", ".log"}
TEXT_NAMES = {".gitignore", "Makefile"}

# Old level suffix -> new level suffix (both include the leading underscore).
RENAMES: dict[str, str] = {
    "_native_qt_max_context": "_native_qt_all_cues",
}

LANG = r"[a-z]{2}(?:-[A-Za-z]{2,4})?"
# Right boundary: end of token, or a kept "_shuffled" suffix.
END = r"(?=(?:_shuffled)?(?![A-Za-z0-9_]))"


class Renamer:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        # Longest first so a longer suffix wins over a prefix of it. Keys are matched without
        # their leading "_", either after "<lang>_" or as a bare token (e.g. "native_qt_max_context"
        # in experiment_names.py).
        alts = "|".join(re.escape(k[1:]) for k in sorted(mapping, key=len, reverse=True))
        self.regex = re.compile(
            rf"(?:(?<![A-Za-z0-9])(?P<lang>{LANG})_|(?<![A-Za-z0-9_]))(?P<level>{alts}){END}"
        )

    def might_match(self, text: str) -> bool:
        return any(k[1:] in text for k in self.mapping)

    def __call__(self, text: str, counter: Counter | None = None) -> str:
        def sub(m: re.Match[str]) -> str:
            lang, level = m.group("lang"), "_" + m.group("level")
            new = self.mapping[level][1:]
            if counter is not None:
                counter[f"<lang>{level} -> <lang>{self.mapping[level]}" if lang else f"{level[1:]} -> {new}"] += 1
            return f"{lang}_{new}" if lang else new

        return self.regex.sub(sub, text)


def parse_map(items: list[str]) -> dict[str, str]:
    mapping = {}
    for item in items:
        if item.count(":") != 1:
            sys.exit(f"--map expects OLD:NEW, got {item!r}")
        old, new = item.split(":")
        if not old.startswith("_") or not new.startswith("_"):
            sys.exit(f"--map suffixes must start with '_', got {item!r}")
        mapping[old] = new
    return mapping


def walk(root: Path):
    """Yield (dirpath, dirnames, filenames), pruning SKIP_DIRS."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        yield Path(dirpath), dirnames, filenames


def area(p: Path, root: Path) -> str:
    """Grouping key for the report: first path components under the repo."""
    parts = p.relative_to(root).parts
    return "/".join(parts[:3] if parts[0] == "multilingual-bfcl" and len(parts) > 2 else parts[:2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--map", action="append", default=[], metavar="OLD:NEW",
                    help="Suffix rename, e.g. _native_qt_max_context:_native_qt_all_cues. "
                         "Repeatable; replaces the built-in RENAMES when given.")
    ap.add_argument("--apply", action="store_true", help="Actually perform the changes (default: dry run).")
    ap.add_argument("--verbose", "-v", action="store_true", help="List every affected file and rename.")
    args = ap.parse_args()
    root = args.root.resolve()

    mapping = parse_map(args.map) if args.map else dict(RENAMES)
    if not mapping:
        sys.exit("No renames configured.")
    chained = set(mapping.values()) & set(mapping)
    if chained:
        sys.exit(f"New suffix(es) also appear as old suffix(es), the rename would not be stable: {chained}")
    rename = Renamer(mapping)

    # ---- 1. content edits -------------------------------------------------
    content_edits: dict[Path, int] = {}
    by_mapping: Counter = Counter()
    by_area_files: Counter = Counter()
    by_area_occ: Counter = Counter()
    not_utf8: list[Path] = []
    for d, _, files in walk(root):
        for name in files:
            p = d / name
            if p.resolve() == SELF:
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES and name not in TEXT_NAMES:
                continue
            try:
                text = p.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                not_utf8.append(p)
                continue
            if not rename.might_match(text):
                continue
            c: Counter = Counter()
            if rename(text, c) != text:
                n = sum(c.values())
                content_edits[p] = n
                by_mapping.update(c)
                by_area_files[area(p, root)] += 1
                by_area_occ[area(p, root)] += n

    # ---- 2. path renames (deepest first so parents are renamed last) -----
    renames: list[tuple[Path, Path]] = []
    rename_mapping: Counter = Counter()
    for d, dirnames, files in walk(root):
        for name in dirnames + files:
            c = Counter()
            new = rename(name, c)
            if new != name:
                renames.append((d / name, d / new))
                rename_mapping.update(c)
    renames.sort(key=lambda t: len(t[0].parts), reverse=True)

    collisions = [(s, t) for s, t in renames if t.exists()]

    # ---- 3. leftovers: old suffixes that weren't matched as whole tokens --
    leftovers: Counter = Counter()
    for d, dirnames, files in walk(root):
        for name in dirnames + files:
            after = rename(name)
            if rename.might_match(after):
                leftovers[f"[path] {after}"] += 1

    # ---- report ------------------------------------------------------------
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] repo root: {root}")
    for old, new in mapping.items():
        print(f"    <lang>{old}  ->  <lang>{new}")
    print(f"\nCONTENT: {len(content_edits)} files, {sum(content_edits.values())} occurrences")
    for k, v in sorted(by_mapping.items(), key=lambda kv: -kv[1]):
        print(f"    {k:60s} {v:>8}")
    print("  by area (files / occurrences):")
    for a in sorted(by_area_files):
        print(f"    {a:60s} {by_area_files[a]:>6} / {by_area_occ[a]:>8}")

    n_dirs = sum(1 for s, _ in renames if s.is_dir())
    print(f"\nPATHS: {len(renames)} renames ({n_dirs} folders, {len(renames) - n_dirs} files)")
    for k, v in sorted(rename_mapping.items(), key=lambda kv: -kv[1]):
        print(f"    {k:60s} {v:>8}")
    by_area_ren = Counter(area(s, root) for s, _ in renames)
    print("  by area:")
    for a, v in sorted(by_area_ren.items()):
        print(f"    {a:60s} {v:>6}")

    if args.verbose:
        print("\nContent edits:")
        for p, n in sorted(content_edits.items()):
            print(f"    {p.relative_to(root)}  ({n})")
        print("\nRenames:")
        for s, t in sorted(renames):
            print(f"    {s.relative_to(root)}  ->  {t.name}")

    if leftovers:
        print("\nPath names still containing an old suffix (not a whole-token match, left as is):")
        for k, v in sorted(leftovers.items()):
            print(f"    {k}  x{v}")
    if not_utf8:
        print(f"\nSkipped (not UTF-8): {len(not_utf8)}")
        for p in not_utf8[:20]:
            print(f"    {p.relative_to(root)}")
    if collisions:
        print(f"\nERROR: {len(collisions)} rename target(s) already exist:")
        for s, t in collisions:
            print(f"    {s.relative_to(root)} -> {t.name}")
        print("Nothing changed.")
        return 1

    if not args.apply:
        print("\nDry run only. Re-run with --apply to perform these changes.")
        return 0

    for p in content_edits:
        raw = p.read_bytes().decode("utf-8")  # bytes round-trip keeps line endings
        p.write_bytes(rename(raw).encode("utf-8"))
    for s, t in renames:
        s.rename(t)
    print(f"\nDone: edited {len(content_edits)} file(s), renamed {len(renames)} path(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
