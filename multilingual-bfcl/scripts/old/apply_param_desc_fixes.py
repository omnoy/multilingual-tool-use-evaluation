"""
apply_param_desc_fixes.py — apply the parameter-description review (from
scripts/param_desc_review.html) to every variant of a benchmark that still carries
the English parameter descriptions, writing the result to a separate benchmark folder.

Input
  data/param_desc_reviews/<benchmark>/en_translatable.param_desc.csv
    One row per (function, param, original_description). Rows with
    description_changed=true replace the description; rows with remove_default=true
    delete the parameter's "default" key.
  data/benchmarks/<benchmark>/<lang>/*.jsonl
    Every variant except those whose name ends in an excluded suffix (default:
    _full_func_desc, _full_full_full — their descriptions are translated, so the
    English fixes cannot be matched).

Matching
  A parameter is fixed when its function name, parameter path and current description
  equal a reviewed row's function, param and original_description. Parameter paths use
  the review page's syntax: nested properties as "outer.inner", array-item properties
  as "outer[].inner". In *_lang_desc files the description carries an appended
  " Languages: English, Hebrew." descriptor; it is stripped for matching and re-appended
  after the new description.

Output
  data/benchmarks/<benchmark>_fixed_desc/<lang>/<same file name>.jsonl
  data/benchmarks/<benchmark>_fixed_desc/fix_manifest.json  (sources, CSV, per-file counts)
  Only lines that changed are re-serialized; all other lines are copied byte-for-byte.
  The output category name still contains "multiple", so run_robustness_eval.py scores
  it exactly like the original:  --category bfcl_multiple_fixed_desc

Usage
  python scripts/apply_param_desc_fixes.py                      # bfcl_multiple
  python scripts/apply_param_desc_fixes.py --dry-run            # report only
  python scripts/apply_param_desc_fixes.py --force              # overwrite an existing output folder
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PACKAGE_ROOT / "data"
BENCHMARKS_ROOT = DATA_ROOT / "benchmarks"

DEFAULT_EXCLUDE = ("_full_func_desc", "_full_full_full")
# Appended by add_language_descriptors.py, e.g. " Languages: English, Hebrew."
LANG_SUFFIX_RE = re.compile(r"( Languages?: [^.]*\.)$")


def load_fixes(csv_path: Path) -> dict[tuple[str, str, str], dict]:
    """(function, param, original_description) -> {"new": str | None, "remove_default": bool}."""
    fixes = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            changed = row["description_changed"] == "true"
            remove = row["remove_default"] == "true"
            if not (changed or remove):
                continue
            key = (row["function"], row["param"], row["original_description"])
            fixes[key] = {"new": row["edited_description"] if changed else None, "remove_default": remove}
    return fixes


def iter_params(props: dict | None, prefix: str = ""):
    """Yield (param_path, schema) for every parameter, including nested ones."""
    for name, schema in (props or {}).items():
        if not isinstance(schema, dict):
            continue
        path = prefix + name
        yield path, schema
        yield from iter_params(schema.get("properties"), path + ".")
        items = schema.get("items")
        if isinstance(items, dict):
            yield from iter_params(items.get("properties"), path + "[].")


def fix_entry(entry: dict, fixes: dict, stats: Counter, seen: set) -> bool:
    changed = False
    for fn in entry.get("function") or []:
        params = (fn.get("parameters") or {}).get("properties")
        for path, schema in iter_params(params):
            desc = schema.get("description", "")
            key, suffix = (fn["name"], path, desc), ""
            if key not in fixes:
                m = LANG_SUFFIX_RE.search(desc)
                if m:
                    key, suffix = (fn["name"], path, desc[: m.start()]), m.group(1)
            fix = fixes.get(key)
            if fix is None:
                continue
            seen.add(key)
            if fix["new"] is not None:
                new_desc = fix["new"] + suffix
                if new_desc != desc:
                    schema["description"] = new_desc
                    stats["descriptions_changed"] += 1
                    changed = True
            if fix["remove_default"]:
                if "default" in schema:
                    del schema["default"]
                    stats["defaults_removed"] += 1
                    changed = True
                else:
                    stats["default_already_absent"] += 1
    return changed


def process_file(src: Path, dst: Path, fixes: dict, dry_run: bool) -> dict:
    stats: Counter = Counter()
    seen: set = set()
    out_lines = []
    raw = src.read_text(encoding="utf-8")
    for line in raw.split("\n"):
        if not line.strip():
            out_lines.append(line)
            continue
        entry = json.loads(line)
        stats["entries"] += 1
        if fix_entry(entry, fixes, stats, seen):
            stats["entries_changed"] += 1
            out_lines.append(json.dumps(entry, ensure_ascii=False))
        else:
            out_lines.append(line)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(out_lines), encoding="utf-8")
    return {"stats": dict(stats), "matched_fix_keys": len(seen), "unmatched_fix_keys": sorted(set(fixes) - seen)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="bfcl_multiple", help="Benchmark folder under data/benchmarks/ (default: bfcl_multiple).")
    ap.add_argument("--reviews", type=Path, help="Review CSV (default: data/param_desc_reviews/<benchmark>/en_translatable.param_desc.csv).")
    ap.add_argument("--output-benchmark", help="Output folder name under data/benchmarks/ (default: <benchmark>_fixed_desc).")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="File-stem suffixes to skip (default: %(default)s).")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything.")
    ap.add_argument("--force", action="store_true", help="Overwrite files in an existing output folder.")
    args = ap.parse_args()

    src_root = BENCHMARKS_ROOT / args.benchmark
    csv_path = args.reviews or DATA_ROOT / "param_desc_reviews" / args.benchmark / "en_translatable.param_desc.csv"
    out_name = args.output_benchmark or f"{args.benchmark}_fixed_desc"
    out_root = BENCHMARKS_ROOT / out_name
    if not src_root.is_dir():
        sys.exit(f"Benchmark folder not found: {src_root}")
    if not csv_path.is_file():
        sys.exit(f"Review CSV not found: {csv_path}")
    if out_root.resolve() == src_root.resolve():
        sys.exit("Output folder must differ from the source benchmark folder.")
    if out_root.exists() and any(out_root.rglob("*.jsonl")) and not args.force and not args.dry_run:
        sys.exit(f"{out_root} already has .jsonl files; pass --force to overwrite.")

    fixes = load_fixes(csv_path)
    n_desc = sum(f["new"] is not None for f in fixes.values())
    n_def = sum(f["remove_default"] for f in fixes.values())
    print(f"Review CSV: {csv_path.relative_to(PACKAGE_ROOT)}")
    print(f"  {len(fixes)} parameter fixes: {n_desc} description edits, {n_def} default removals")

    files = sorted(p for p in src_root.glob("*/*.jsonl") if not p.stem.endswith(tuple(args.exclude)))
    skipped = sorted(p for p in src_root.glob("*/*.jsonl") if p.stem.endswith(tuple(args.exclude)))
    report = {}
    problems = False
    for src in files:
        rel = src.relative_to(src_root)
        res = process_file(src, out_root / rel, fixes, args.dry_run)
        report[rel.as_posix()] = res
        s = res["stats"]
        print(f"{rel.as_posix():45s} entries {s.get('entries', 0):4d} changed {s.get('entries_changed', 0):4d} | "
              f"descriptions {s.get('descriptions_changed', 0):4d} | defaults removed {s.get('defaults_removed', 0):3d} | "
              f"fixes matched {res['matched_fix_keys']}/{len(fixes)}")
        if res["unmatched_fix_keys"]:
            problems = True
            for fn, param, orig in res["unmatched_fix_keys"][:10]:
                print(f"    ! not found: {fn} / {param}: {orig[:80]!r}")
            if len(res["unmatched_fix_keys"]) > 10:
                print(f"    ! ... and {len(res['unmatched_fix_keys']) - 10} more")
    for p in skipped:
        print(f"{p.relative_to(src_root).as_posix():45s} skipped (excluded suffix)")

    if not args.dry_run:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_benchmark": args.benchmark,
            "review_csv": csv_path.relative_to(PACKAGE_ROOT).as_posix(),
            "excluded_suffixes": args.exclude,
            "skipped_files": [p.relative_to(src_root).as_posix() for p in skipped],
            "fix_counts": {"total": len(fixes), "description_edits": n_desc, "default_removals": n_def},
            "files": {k: {"stats": v["stats"], "matched_fix_keys": v["matched_fix_keys"],
                          "unmatched_fix_keys": [list(x) for x in v["unmatched_fix_keys"]]} for k, v in report.items()},
        }
        (out_root / "fix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {len(files)} file(s) to {out_root.relative_to(PACKAGE_ROOT)}")
    else:
        print("\nDry run: nothing written.")
    if problems:
        print("WARNING: some fixes were not found in some files (see '!' lines above).")


if __name__ == "__main__":
    main()
