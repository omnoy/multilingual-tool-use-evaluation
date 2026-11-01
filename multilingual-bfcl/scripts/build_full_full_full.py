"""
build_full_full_full.py — build the "full_full_full" benchmark variation.

Combines three existing variations of a benchmark into the condition carrying the
maximum steering signal (no API calls — pure deterministic merge):

  - user QUERY + ground truth            <- <locale>_translatable_full.jsonl
  - function/parameter DESCRIPTIONS      <- <locale>_translatable_full_func_desc.jsonl
  - language descriptor on NL parameters <- <locale>_translatable_full_lang_desc.jsonl

The descriptor is localized into the target language: e.g. for Hebrew the English
"Languages: English, Hebrew." becomes "שפות: אנגלית, עברית.". Which parameters get a
descriptor (and which languages it lists) is read from the full_lang_desc file, so
the tagging is exactly the one used there.

All three sources must exist and share the same ids and function structure
(ignoring descriptions); any mismatch raises an error.

Output:
  data/benchmarks/<benchmark>/<locale>/<locale>_translatable_full_full_full.jsonl

Usage:
    python scripts/build_full_full_full.py --benchmark bfcl_multiple --locale he
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = PACKAGE_ROOT / "data" / "benchmarks"

LEVEL = "full_full_full"
OUTPUT_VARIANT = "full_full_full"

# Matches the English descriptor appended by translator.add_language_descriptors_to_entry.
HINT_RE = re.compile(r"\s*Languages: ([^.]+)\.$")

# Per target locale: the localized "Languages" label, the list separator, the
# label/value separator, the final punctuation, and the localized language names.
LOCALIZED_HINTS: dict[str, dict[str, Any]] = {
    "he": {
        "label": "שפות", "colon": ": ", "sep": ", ", "end": ".",
        "names": {"English": "אנגלית", "Hebrew": "עברית"},
    },
    "de": {
        "label": "Sprachen", "colon": ": ", "sep": ", ", "end": ".",
        "names": {"English": "Englisch", "German": "Deutsch"},
    },
    "ar": {
        "label": "اللغات", "colon": ": ", "sep": "، ", "end": ".",
        "names": {"English": "الإنجليزية", "Arabic": "العربية"},
    },
    "zh-CN": {
        "label": "语言", "colon": "：", "sep": "、", "end": "。",
        "names": {"English": "英语", "Chinese (Simplified)": "简体中文"},
    },
    "zh": {
        "label": "语言", "colon": "：", "sep": "、", "end": "。",
        "names": {"English": "英语", "Chinese (Simplified)": "简体中文", "Chinese": "中文"},
    },
    "fr": {
        "label": "Langues", "colon": " : ", "sep": ", ", "end": ".",
        "names": {"English": "anglais", "French": "français"},
    },
    "es": {
        "label": "Idiomas", "colon": ": ", "sep": ", ", "end": ".",
        "names": {"English": "inglés", "Spanish": "español"},
    },
}


def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required source file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["id"]: r for r in rows}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_descriptions(obj: Any) -> Any:
    """Copy of a structure with every "description" value blanked, for comparison."""
    if isinstance(obj, dict):
        return {k: ("" if k == "description" else _strip_descriptions(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_descriptions(x) for x in obj]
    return obj


def localize_hint(english_languages: str, locale: str) -> str:
    """"English, Hebrew" -> "שפות: אנגלית, עברית." for locale "he"."""
    cfg = LOCALIZED_HINTS[locale]
    names = []
    for name in (n.strip() for n in english_languages.split(",")):
        if name not in cfg["names"]:
            raise ValueError(f"no {locale!r} translation for language name {name!r}; "
                             f"add it to LOCALIZED_HINTS")
        names.append(cfg["names"][name])
    return f"{cfg['label']}{cfg['colon']}{cfg['sep'].join(names)}{cfg['end']}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full_full_full variation (translated query + translated "
                    "function descriptions + localized language descriptors).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", "-b", required=True,
                        help="Benchmark directory under data/benchmarks/, e.g. bfcl_multiple.")
    parser.add_argument("--locale", "-l", required=True,
                        help=f"Target locale, one of: {', '.join(LOCALIZED_HINTS)}.")
    args = parser.parse_args()

    if args.locale not in LOCALIZED_HINTS:
        raise ValueError(f"no localized descriptor for locale {args.locale!r}; "
                         f"supported: {', '.join(LOCALIZED_HINTS)}")

    locale_dir = BENCHMARKS_ROOT / args.benchmark / args.locale
    stem = f"{args.locale}_translatable"
    full = _load_jsonl(locale_dir / f"{stem}_full.jsonl")
    func_desc = _load_jsonl(locale_dir / f"{stem}_full_func_desc.jsonl")
    lang_desc = _load_jsonl(locale_dir / f"{stem}_full_lang_desc.jsonl")

    if not (set(full) == set(func_desc) == set(lang_desc)):
        diff = sorted((set(full) | set(func_desc) | set(lang_desc))
                      - (set(full) & set(func_desc) & set(lang_desc)))
        raise ValueError(f"{len(diff)} id(s) not present in all three sources, e.g. {diff[:5]}")

    out_entries: list[dict] = []
    tagged = 0
    for entry_id, fd_entry in func_desc.items():  # preserve func_desc order
        ld_entry = lang_desc[entry_id]
        if _strip_descriptions(fd_entry["function"]) != _strip_descriptions(ld_entry["function"]):
            raise ValueError(f"{entry_id}: function structure differs between "
                             f"full_func_desc and full_lang_desc")
        if "ground_truth" not in full[entry_id]:
            raise ValueError(f"{entry_id}: no ground_truth in the full source")

        merged = copy.deepcopy(fd_entry)  # translated function descriptions
        merged["messages"] = full[entry_id]["messages"]  # translated query
        merged["ground_truth"] = full[entry_id]["ground_truth"]
        merged["localization_level"] = LEVEL

        for fd_func, ld_func in zip(merged["function"], ld_entry["function"]):
            ld_props = ld_func.get("parameters", {}).get("properties", {})
            fd_props = fd_func.get("parameters", {}).get("properties", {})
            for param, ld_pdef in ld_props.items():
                match = HINT_RE.search(ld_pdef.get("description", ""))
                if not match:
                    continue
                hint = localize_hint(match.group(1), args.locale)
                desc = fd_props[param].get("description", "")
                fd_props[param]["description"] = f"{desc} {hint}".strip()
                tagged += 1

        out_entries.append(merged)

    out_path = locale_dir / f"{stem}_{OUTPUT_VARIANT}.jsonl"
    _write_jsonl(out_path, out_entries)
    print(f"wrote {len(out_entries)} entries -> {out_path}")
    print(f"      {tagged} parameter description(s) tagged with a localized descriptor")


if __name__ == "__main__":
    main()
