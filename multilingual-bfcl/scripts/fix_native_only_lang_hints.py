"""
fix_native_only_lang_hints.py — rewrite existing language descriptors so they name
only the native (target) language instead of English + the native language.

Affects the two experiment levels that carry a language descriptor:

  <locale>_native_qt_lang_hints.jsonl    "Languages: English, Hebrew."  -> "Languages: Hebrew."
  <locale>_native_qt_all_cues.jsonl   "שפות: אנגלית, עברית."         -> "שפות: עברית."

Only the trailing descriptor of a parameter description is touched; the rest of the
entry is left byte-for-byte as it was (same key order, same JSON formatting). Each
file is backed up before being rewritten, to

  data/old/backups/<benchmark>_en_native_hints/<locale>/<file>

(the benchmark files are not tracked by git, so this is the only way back).
Running the script twice is a no-op: already-fixed descriptors are left alone.

Usage:
  python scripts/fix_native_only_lang_hints.py                       # all locales, dry run
  python scripts/fix_native_only_lang_hints.py --apply               # write changes
  python scripts/fix_native_only_lang_hints.py --locales he de --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = PACKAGE_ROOT / "data" / "benchmarks"
BACKUP_ROOT = PACKAGE_ROOT / "data" / "old" / "backups"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from multilingual_bfcl.experiment_names import experiment_stem  # noqa: E402
from multilingual_bfcl.localization.locale_config import get_locale  # noqa: E402
from build_native_max_context import LOCALIZED_HINTS  # noqa: E402


def english_rewrite(locale: str) -> tuple[re.Pattern, str]:
    """'Languages: English, Hebrew.' -> 'Languages: Hebrew.' (end of description)."""
    native = get_locale(locale).name
    pat = re.compile(r"Languages: English, " + re.escape(native) + r"\.$")
    return pat, f"Languages: {native}."


def localized_rewrite(locale: str) -> tuple[re.Pattern, str]:
    """'שפות: אנגלית, עברית.' -> 'שפות: עברית.' (end of description)."""
    cfg = LOCALIZED_HINTS[locale]
    eng = cfg["names"]["English"]
    native = cfg["names"][get_locale(locale).name]
    head = cfg["label"] + cfg["colon"]
    pat = re.compile(re.escape(head + eng + cfg["sep"] + native + cfg["end"]) + "$")
    return pat, head + native + cfg["end"]


def fix_descriptions(obj, pat: re.Pattern, repl: str) -> int:
    """Rewrite every matching "description" value in place; return the count."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                new, c = pat.subn(repl, v)
                if c:
                    obj[k] = new
                    n += c
            else:
                n += fix_descriptions(v, pat, repl)
    elif isinstance(obj, list):
        for x in obj:
            n += fix_descriptions(x, pat, repl)
    return n


def process_file(path: Path, pat: re.Pattern, repl: str, apply: bool, backup_dir: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, changed, entries = [], 0, 0
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        n = fix_descriptions(entry.get("function", []), pat, repl)
        changed += n
        entries += bool(n)
        out.append(json.dumps(entry, ensure_ascii=False) if n else line)

    print(f"  {path.name}: {changed} descriptor(s) in {entries} entrie(s)"
          f"{'' if apply else ' [dry run]'}")
    if apply and changed:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / path.name
        if not backup.exists():  # keep the original if the script is re-run
            shutil.copy2(path, backup)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", "-b", default="bfcl_multiple_translatable")
    parser.add_argument("--locales", "-l", nargs="+", default=None,
                        help="Locales to fix (default: every locale dir except en).")
    parser.add_argument("--apply", action="store_true",
                        help="Write the changes (default is a dry run).")
    args = parser.parse_args()

    bench_dir = BENCHMARKS_ROOT / args.benchmark
    locales = args.locales or sorted(p.name for p in bench_dir.iterdir()
                                     if p.is_dir() and p.name != "en")
    backup_root = BACKUP_ROOT / f"{args.benchmark}_en_native_hints"

    for locale in locales:
        print(f"[{locale}]")
        jobs = [
            (experiment_stem(locale, "full_lang_desc"), english_rewrite(locale)),
            (experiment_stem(locale, "full_full_full"), localized_rewrite(locale)),
        ]
        for stem, (pat, repl) in jobs:
            path = bench_dir / locale / f"{stem}.jsonl"
            if not path.is_file():
                print(f"  {path.name}: not found, skipped")
                continue
            process_file(path, pat, repl, args.apply, backup_root / locale)


if __name__ == "__main__":
    main()
