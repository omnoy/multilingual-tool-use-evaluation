"""
seed_fn_desc_cache.py — pre-fill translate_function_descriptions.py's translation cache
for a derived benchmark (e.g. bfcl_multiple_fixed_desc) from translations that already
exist for its source benchmark, so a re-run only translates functions whose English
descriptions changed.

translate_function_descriptions.py caches each translated function under
sha256([language, <English descriptions in DFS order>]). A function whose descriptions
were not edited therefore has the same key in the derived benchmark, and a seeded cache
lets the translator skip it. Edited functions get a new key and are translated.

Sources for the seed, per locale (later sources never overwrite earlier ones):
  1. the target's existing cache   data/benchmarks/<target>/<lc>/.<lc>_fn_desc_cache.json
  2. the source's existing cache   data/benchmarks/<source>/<lc>/.<lc>_fn_desc_cache.json
  3. translations reconstructed from an already-translated variant of the source,
     default <lc>_translatable_full_func_desc.jsonl: each English function in
     <source>/en/en_translatable.jsonl is paired with the same-position function of the
     same entry, and their description lists are paired in DFS order.

Output: data/benchmarks/<target>/<lc>/.<lc>_fn_desc_cache.json, plus a report of how many
distinct functions in <target>/en/en_translatable.jsonl still need translating.

Usage:
    .venv/Scripts/python scripts/seed_fn_desc_cache.py --locales de he
    .venv/Scripts/python scripts/seed_fn_desc_cache.py --locales de he --dry-run

Then translate only what is missing:
    python scripts/translate_function_descriptions.py --category bfcl_multiple_fixed_desc --locales de he --model <model>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PACKAGE_ROOT / "data" / "benchmarks"

# Reuse the translator's own key/collection functions so the keys always match.
_spec = importlib.util.spec_from_file_location(
    "translate_function_descriptions", Path(__file__).with_name("translate_function_descriptions.py"))
tfd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfd)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def source_id(entry: dict) -> str:
    return entry.get("source_id") or entry["id"].rsplit("_", 1)[0]


def reconstruct(en_entries: list[dict], translated_entries: list[dict], language: str) -> tuple[dict, list[str]]:
    """{cache key: translated descriptions} from paired English/translated entries."""
    by_id = {source_id(e): e for e in translated_entries}
    out: dict[str, list[str]] = {}
    problems: list[str] = []
    for en in en_entries:
        tr = by_id.get(en["id"])
        if tr is None:
            problems.append(f"{en['id']}: missing from translated file")
            continue
        if [f["name"] for f in en["function"]] != [f["name"] for f in tr["function"]]:
            problems.append(f"{en['id']}: function names differ")
            continue
        for f_en, f_tr in zip(en["function"], tr["function"]):
            a, b = tfd.collect_descriptions(f_en), tfd.collect_descriptions(f_tr)
            if not a:
                continue
            if len(a) != len(b):
                problems.append(f"{en['id']} / {f_en['name']}: {len(a)} English vs {len(b)} translated descriptions")
                continue
            key = tfd._cache_key(a, language)
            if key in out and out[key] != b:
                problems.append(f"{en['id']} / {f_en['name']}: conflicting translations for identical English; keeping first")
                continue
            out.setdefault(key, b)
    return out, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-benchmark", default="bfcl_multiple")
    ap.add_argument("--target-benchmark", default="bfcl_multiple_fixed_desc")
    ap.add_argument("--locales", nargs="+", required=True, help="Locale codes, e.g. de he.")
    ap.add_argument("--en-source", default="en/en_translatable.jsonl",
                    help="English file, relative to each benchmark folder (default: %(default)s).")
    ap.add_argument("--translated-variant", default="full_func_desc",
                    help="Variant whose descriptions are translated: <lc>_translatable_<variant>.jsonl (default: %(default)s).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; do not write caches.")
    args = ap.parse_args()

    src_root = BENCHMARK_DIR / args.source_benchmark
    tgt_root = BENCHMARK_DIR / args.target_benchmark
    en_src = load_jsonl(src_root / args.en_source)
    en_tgt_path = tgt_root / args.en_source
    if not en_tgt_path.exists():
        sys.exit(f"Target English file not found: {en_tgt_path}")
    en_tgt = load_jsonl(en_tgt_path)

    for lc in args.locales:
        locale = tfd.get_locale(lc)
        tgt_cache_path = tfd.cache_path(args.target_benchmark, lc)
        src_cache_path = tfd.cache_path(args.source_benchmark, lc)
        tgt_cache = tfd._load_cache(tgt_cache_path)
        src_cache = tfd._load_cache(src_cache_path)

        translated_path = src_root / lc / f"{lc}_translatable_{args.translated_variant}.jsonl"
        recon, problems = ({}, [f"{translated_path} not found"]) if not translated_path.exists() \
            else reconstruct(en_src, load_jsonl(translated_path), locale.name)

        # Consistency check: reconstructed vs. the source's own cache, where both exist.
        both = [k for k in recon if k in src_cache]
        disagree = sum(src_cache[k] != recon[k] for k in both)

        seeded = dict(tgt_cache)
        added_src = added_recon = 0
        for k, v in src_cache.items():
            if k not in seeded:
                seeded[k] = v; added_src += 1
        for k, v in recon.items():
            if k not in seeded:
                seeded[k] = v; added_recon += 1

        needed = tfd.descriptions_by_cache_key(tfd.distinct_functions(en_tgt), locale)
        missing = {k: d for k, d in needed.items() if len(seeded.get(k, [])) != len(d)}
        names = {}
        for f in tfd.distinct_functions(en_tgt).values():
            d = tfd.collect_descriptions(f)
            if d:
                names.setdefault(tfd._cache_key(d, locale.name), f["name"])

        print(f"[{lc}] {locale.name}")
        print(f"  existing target cache : {len(tgt_cache)}")
        print(f"  from source cache     : +{added_src} (of {len(src_cache)})")
        print(f"  reconstructed         : +{added_recon} (of {len(recon)} from {translated_path.name})"
              + (f"; {disagree}/{len(both)} disagree with source cache" if both else ""))
        print(f"  seeded cache size     : {len(seeded)}")
        print(f"  target functions      : {len(needed)} distinct, {len(needed) - len(missing)} already translated, "
              f"{len(missing)} to translate")
        for p in problems[:10]:
            print(f"  ! {p}")
        if len(problems) > 10:
            print(f"  ! ... and {len(problems) - 10} more")
        if not args.dry_run:
            tfd._save_cache(tgt_cache_path, seeded)
            print(f"  wrote {tgt_cache_path.relative_to(PACKAGE_ROOT)}")
    if args.dry_run:
        print("\nDry run: no caches written.")


if __name__ == "__main__":
    main()
