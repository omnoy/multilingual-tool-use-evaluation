"""
add_language_descriptors.py — annotate a benchmark's natural-language parameters
with an explicit "Language: <langs>." descriptor.

Unlike `mbfcl descriptors` (which derives the language list per entry from each
entry's own `locale` field), this standalone script takes the languages
explicitly on the command line, so you control exactly what appears in the hint
regardless of what locale the entries claim to be in. This is the "do the same
thing" variant for building descriptor benchmarks with an arbitrary language set.

Detection is unchanged and ground-truth driven (see
translator.add_language_descriptors_to_entry):
  - a parameter is natural-language if at least one of its ground-truth values
    passes the basic is_translatable_string regex (not a number/bool/date);
  - enums are excluded;
  - the descriptor is propagated to structurally-identical params in sibling
    (distractor) functions so it never reveals which function is the answer.

The descriptor languages are the NAMES of `--locale` followed by `--locales`,
order-preserving and de-duplicated. For example:

  --locale en --locales he        ->  "Language: English, Hebrew."
  --locale en --locales he ar      ->  "Language: English, Hebrew, Arabic."
  --locale he --locales en         ->  "Language: Hebrew, English."

Usage:
  python scripts/add_language_descriptors.py INPUT.jsonl --locales he
  python scripts/add_language_descriptors.py INPUT.jsonl --locale en --locales he ar
  python scripts/add_language_descriptors.py INPUT.jsonl --locales he --output OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from multilingual_bfcl.localization.locale_config import get_locale  # noqa: E402
from multilingual_bfcl.localization.translator import (  # noqa: E402
    add_language_descriptors_to_entry,
)


def build_language_string(source_locale: str, dest_locale: str) -> str:
    """Resolve [locale, *locales] to locale NAMES, order-preserving and de-duplicated.

    e.g. ("en", ["he", "ar"]) -> "English, Hebrew, Arabic"
    """
    names: list[str] = []
    seen: set[str] = set()
    for code in [source_locale, dest_locale]:
        name = get_locale(code).name  # raises ValueError on an unknown code
        if name not in seen:
            seen.add(name)
            names.append(name)
    return ", ".join(names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append an explicit 'Language: <langs>.' descriptor to the "
        "natural-language parameters of a built benchmark .jsonl.",
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to a built benchmark .jsonl (must carry inline ground_truth).",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Benchmark Category"
    )
    parser.add_argument(
        "--source_locale", "-L",
        default="en",
        help="Base locale whose name leads the descriptor (default: en).",
    )
    parser.add_argument(
        "--dest_locale", "-l",
        required=True,
        help="Target locale code(s) to list after the base, e.g. he ar zh-CN.",
    )
    parser.add_argument(
        "--suffix",
        default="lang_desc",
        help="Suffix for the output filename (<stem>_<suffix><ext>). "
        "Ignored when --output is given. Default: lang_desc.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Explicit output path (overrides --suffix).",
    )
    args = parser.parse_args()

    try:
        languages = build_language_string(args.source_locale, args.dest_locale)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    data_root = PACKAGE_ROOT / "data" / "benchmarks" / args.category / args.dest_locale

    out = data_root / f"{args.input_file}_{args.suffix}.jsonl"

    entries = []

    with open(data_root / f"{args.input_file}.jsonl" , encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    total_modified = 0
    tagged_entries = 0
    for entry in entries:
        n = add_language_descriptors_to_entry(entry, languages)
        total_modified += n
        if n:
            tagged_entries += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Descriptor : Language: {languages}.")
    print(f"[done] {len(entries)} entries -> {out}")
    print(f"       {total_modified} parameter description(s) tagged "
          f"across {tagged_entries} entrie(s).")


if __name__ == "__main__":
    main()
