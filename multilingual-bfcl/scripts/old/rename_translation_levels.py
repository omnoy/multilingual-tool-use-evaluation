"""Rename the bfcl_multiple benchmark and its experiment names (translation levels) repo-wide.

Benchmark:
    bfcl_multiple                  -> bfcl_multiple_translatable
    bfcl_multiple_fixed_desc[_gt]  -> bfcl_multiple_translatable_fixed_desc[_gt]

Experiments (<lang> = 2-letter code, optionally with a region, e.g. he, de, zh-CN):
    en_translatable                    -> en_baseline
    <lang>_translatable_query          -> <lang>_native_query_eng_target
    <lang>_translatable_full           -> <lang>_native_query_native_target
    <lang>_translatable_full_func_desc -> <lang>_native_qt_func_descs
    <lang>_translatable_full_lang_desc -> <lang>_native_qt_lang_hints
    <lang>_translatable_full_full_full -> <lang>_native_qt_max_context
    <lang>_translatable_func_desc      -> <lang>_native_query_func_descs
    (a trailing "_shuffled" is kept, e.g. he_translatable_full_shuffled -> he_native_query_native_target_shuffled)

Applied to file CONTENTS (text files) and to file/folder NAMES, across the whole repo
(data/, results_robustness/, old/ folders, scripts, notebooks, ...). Skips .git, virtualenvs,
IDE folders and caches. Names are matched as whole tokens only, so e.g. `eng_translatable`,
`zh_translatable`, `translatable_params` and `to_bfcl_multiple` are untouched. Running it twice
is a no-op (the new names don't match any old pattern).

Not changed: the `localization_level` field values inside entries ("query", "full", ...) and the
CLI --level values. Code that builds names on the fly (f"{lang}_translatable_{level}") is not
fixed by this script and needs a manual edit.

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
from collections import Counter, defaultdict
from pathlib import Path

SELF = Path(__file__).resolve()
REPO_ROOT = SELF.parents[2]

SKIP_DIRS = {".git", ".venv", "venv", ".idea", ".vscode", "__pycache__", "node_modules",
             ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache"}
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml",
                 ".ipynb", ".html", ".toml", ".cfg", ".ini", ".sh", ".ps1", ".bat", ".log"}
TEXT_NAMES = {".gitignore", "Makefile"}

# Level suffix -> new suffix. Longest first so _full_func_desc wins over _full.
LEVELS = {
    "_full_full_full": "_native_qt_max_context",
    "_full_func_desc": "_native_qt_func_descs",
    "_full_lang_desc": "_native_qt_lang_hints",
    "_func_desc": "_native_query_func_descs",
    "_query": "_native_query_eng_target",
    "_full": "_native_query_native_target",
}
LANG = r"[a-z]{2}(?:-[A-Za-z]{2,4})?"
# Right boundary: end of token, or a kept "_shuffled" suffix.
END = r"(?=(?:_shuffled)?(?![A-Za-z0-9_]))"

EXPERIMENT_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<lang>{LANG})_translatable"
    rf"(?P<level>{'|'.join(re.escape(k) for k in LEVELS)})?{END}"
)
BENCHMARK_RE = re.compile(r"(?<![A-Za-z0-9_])bfcl_multiple(?=(?:_fixed_desc(?:_gt)?)?(?![A-Za-z0-9_]))")
# Anything still containing "translatable" after the rename, to report leftovers.
LEFTOVER_RE = re.compile(r"[A-Za-z0-9_.\-]*_translatable[A-Za-z0-9_]*|[A-Za-z0-9]+_translatable\b")


def _exp_sub(m: re.Match[str]) -> str:
    lang, level = m.group("lang"), m.group("level")
    if level is None:
        return "en_baseline" if lang == "en" else m.group(0)  # e.g. zh_translatable: unmapped, keep
    return lang + LEVELS[level]


def rename_text(text: str, counter: Counter | None = None) -> str:
    def exp(m: re.Match[str]) -> str:
        new = _exp_sub(m)
        if counter is not None and new != m.group(0):
            lang, level = m.group("lang"), m.group("level")
            counter[f"<lang>_translatable{level}" if level else "en_translatable"] += 1
        return new

    def bench(m: re.Match[str]) -> str:
        if counter is not None:
            counter["bfcl_multiple"] += 1
        return "bfcl_multiple_translatable"

    text = EXPERIMENT_RE.sub(exp, text)
    return BENCHMARK_RE.sub(bench, text)


def walk(root: Path):
    """Yield (dirpath, dirnames, filenames), pruning SKIP_DIRS."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        yield Path(dirpath), dirnames, filenames


def area(p: Path) -> str:
    """Grouping key for the report: first two path components under the repo."""
    parts = p.relative_to(REPO_ROOT).parts
    return "/".join(parts[:3] if parts[0] == "multilingual-bfcl" and len(parts) > 2 else parts[:2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--apply", action="store_true", help="Actually perform the changes (default: dry run).")
    ap.add_argument("--verbose", "-v", action="store_true", help="List every affected file and rename.")
    args = ap.parse_args()
    root = args.root.resolve()

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
            if "translatable" not in text and "bfcl_multiple" not in text:
                continue
            c: Counter = Counter()
            if rename_text(text, c) != text:
                n = sum(c.values())
                content_edits[p] = n
                by_mapping.update(c)
                by_area_files[area(p)] += 1
                by_area_occ[area(p)] += n

    # ---- 2. path renames (deepest first so parents are renamed last) -----
    renames: list[tuple[Path, Path]] = []
    rename_mapping: Counter = Counter()
    for d, dirnames, files in walk(root):
        for name in dirnames + files:
            c: Counter = Counter()
            new = rename_text(name, c)
            if new != name:
                renames.append((d / name, d / new))
                rename_mapping.update(c)
    renames.sort(key=lambda t: len(t[0].parts), reverse=True)

    collisions = [(s, t) for s, t in renames if t.exists()]

    # ---- 3. leftovers: tokens with "translatable" the mapping doesn't cover
    leftovers: Counter = Counter()
    for d, dirnames, files in walk(root):
        for name in dirnames + files:
            for tok in LEFTOVER_RE.findall(rename_text(name)):
                if tok.startswith("bfcl_multiple_translatable"):
                    continue  # the new benchmark name
                leftovers[f"[path] {tok}"] += 1

    # ---- report ------------------------------------------------------------
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] repo root: {root}\n")
    print(f"CONTENT: {len(content_edits)} files, {sum(content_edits.values())} occurrences")
    for k, v in sorted(by_mapping.items(), key=lambda kv: -kv[1]):
        print(f"    {k:42s} {v:>8}")
    print("  by area (files / occurrences):")
    for a in sorted(by_area_files):
        print(f"    {a:60s} {by_area_files[a]:>6} / {by_area_occ[a]:>8}")

    n_dirs = sum(1 for s, _ in renames if s.is_dir())
    print(f"\nPATHS: {len(renames)} renames ({n_dirs} folders, {len(renames) - n_dirs} files)")
    for k, v in sorted(rename_mapping.items(), key=lambda kv: -kv[1]):
        print(f"    {k:42s} {v:>8}")
    by_area_ren = Counter(area(s) for s, _ in renames)
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
        print("\nPath names still containing 'translatable' after the rename (left as is):")
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
        p.write_bytes(rename_text(raw).encode("utf-8"))
    for s, t in renames:
        s.rename(t)
    print(f"\nDone: edited {len(content_edits)} file(s), renamed {len(renames)} path(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
