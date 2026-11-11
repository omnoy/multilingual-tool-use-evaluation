"""
apply_gt_reviews.py — apply the ground-truth review (from scripts/gt_review.html) to a
benchmark and to the variants derived from the reviewed file, writing the result to a
separate benchmark folder.

Input
  data/gt_reviews/<benchmark>/<stem>.review.csv     one row per entry id
  data/benchmarks/<benchmark>/<lang>/*.jsonl

Which files a review CSV is applied to
  <stem>.review.csv always applies to <lang>/<stem>.jsonl. When <stem> ends in "_full"
  (e.g. he_translatable_full) it is also applied to the variants built from it:

    <stem>_lang_desc   everything; edited descriptions keep the appended
                       " Languages: English, Hebrew." descriptor
    <stem>_func_desc   everything EXCEPT function/parameter description text (those
                       are translated there); defaults and enums ARE changed, because
                       build_full_full_full.py requires func_desc and lang_desc to have
                       identical function structure
    <stem>_full_full   (= *_full_full_full) is NOT written: rebuild it afterwards with
                       build_full_full_full.py --benchmark <output-benchmark>

  Every other file (other locales, en/, _query, manifests) is copied unchanged.

What is applied, per reviewed entry
  entry_removed          the entry is dropped
  query_changes          message content replaced (only if it still equals "old")
  removed/added_params   ground-truth parameters deleted / created
  removed/added_values   acceptable values removed / appended
  function_desc_changes  function description replaced   (not in _func_desc)
  param_desc_changes     parameter description replaced  (not in _func_desc)
  removed_defaults       "default" key deleted
  enum_changes           enum (or items.enum) replaced, or deleted when removed
  For the reviewed file itself, the resulting ground truth is checked against the CSV's
  final_ground_truth column.

Output
  data/benchmarks/<output-benchmark>/<lang>/<same file name>.jsonl
  data/benchmarks/<output-benchmark>/gt_fix_manifest.json
  Unchanged lines are copied byte-for-byte.

Usage
  python scripts/apply_gt_reviews.py --dry-run                        # report only
  python scripts/apply_gt_reviews.py                                  # bfcl_multiple_fixed_desc -> bfcl_multiple_fixed_desc_gt
  python scripts/apply_gt_reviews.py --benchmark X --output-benchmark Y --force
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PACKAGE_ROOT / "data"
BENCHMARKS_ROOT = DATA_ROOT / "benchmarks"
REVIEWS_ROOT = DATA_ROOT / "gt_reviews"

# Appended by add_language_descriptors.py, e.g. " Languages: English, Hebrew."
LANG_SUFFIX_RE = re.compile(r"( Languages?: [^.]*\.)$")

# variant suffix (appended to a *_full stem) -> mode
DERIVED = {"_lang_desc": "lang_desc", "_func_desc": "func_desc"}
REBUILD_SUFFIX = "_full_full"  # <stem>_full_full == *_full_full_full


def jkey(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def jload(s: str, default):
    return json.loads(s) if s and s.strip() else default


# ---------------------------------------------------------------- review CSV
def load_review(csv_path: Path) -> dict[str, dict]:
    """id -> parsed change set (only rows that change something)."""
    out = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("entry_removed") != "true" and row.get("has_changes") != "true":
                continue
            out[row["id"]] = {
                "removed": row.get("entry_removed") == "true",
                "query": jload(row.get("query_changes", ""), []),
                "fn_desc": jload(row.get("function_desc_changes", ""), []),
                "param_desc": jload(row.get("param_desc_changes", ""), []),
                "defaults": jload(row.get("removed_defaults", ""), []),
                "enums": jload(row.get("enum_changes", ""), []),
                "rm_vals": jload(row.get("removed_values", ""), {}),
                "add_vals": jload(row.get("added_values", ""), {}),
                "rm_params": jload(row.get("removed_params", ""), []),
                "add_params": jload(row.get("added_params", ""), []),
                "final_gt": jload(row.get("final_ground_truth", ""), None),
            }
    return out


# ---------------------------------------------------------------- appliers
def split_key(k: str) -> tuple[str, str]:
    fn, p = k.rsplit(".", 1)  # function names may contain dots, params don't
    return fn, p


def gt_call(gt: list, fn: str) -> dict | None:
    for call in gt:
        if fn in call:
            return call[fn]
    return None


def apply_gt(entry: dict, ch: dict, stats: Counter, warn) -> None:
    gt = entry.get("ground_truth") or []
    for k in ch["rm_params"]:
        fn, p = split_key(k)
        call = gt_call(gt, fn)
        if call is not None and p in call:
            del call[p]
            stats["gt_params_removed"] += 1
    for k in ch["add_params"]:
        fn, p = split_key(k)
        call = gt_call(gt, fn)
        if call is None:
            warn(f"ground truth has no call to {fn} (adding param {p})")
        elif p not in call:
            call[p] = []
            stats["gt_params_added"] += 1
    for k, vals in ch["rm_vals"].items():
        fn, p = split_key(k)
        call = gt_call(gt, fn)
        if call is None or p not in call:
            continue  # parameter removed as a whole
        drop = {jkey(v) for v in vals}
        before = len(call[p])
        call[p] = [v for v in call[p] if jkey(v) not in drop]
        stats["gt_values_removed"] += before - len(call[p])
    for k, vals in ch["add_vals"].items():
        fn, p = split_key(k)
        call = gt_call(gt, fn)
        if call is None or p not in call:
            warn(f"cannot add values to missing ground-truth param {k}")
            continue
        have = {jkey(v) for v in call[p]}
        for v in vals:
            if jkey(v) not in have:
                call[p].append(v)
                have.add(jkey(v))
                stats["gt_values_added"] += 1


def apply_query(entry: dict, ch: dict, stats: Counter, warn) -> None:
    msgs = (entry.get("messages") or [[]])[0]
    for q in ch["query"]:
        i = q["index"]
        if i >= len(msgs):
            warn(f"query turn {i + 1} does not exist")
        elif msgs[i].get("content") == q["new"]:
            stats["query_already_applied"] += 1
        elif msgs[i].get("content") == q["old"]:
            msgs[i]["content"] = q["new"]
            stats["query_edited"] += 1
        else:
            warn(f"query turn {i + 1} differs from the reviewed original; not changed")


def replace_desc(holder: dict, old: str, new: str) -> str:
    """Returns 'changed' | 'already' | 'mismatch'. Keeps a trailing Languages: descriptor."""
    cur = holder.get("description", "")
    suffix = ""
    m = LANG_SUFFIX_RE.search(cur)
    if cur not in (old, new) and m:
        cur, suffix = cur[: m.start()], m.group(1)
    if cur == new:
        return "already"
    if cur != old:
        return "mismatch"
    holder["description"] = new + suffix
    return "changed"


def find_fn(entry: dict, name: str) -> dict | None:
    return next((f for f in entry.get("function") or [] if f.get("name") == name), None)


def props_of(fn: dict | None) -> dict:
    return ((fn or {}).get("parameters") or {}).get("properties") or {}


def apply_schema(entry: dict, ch: dict, mode: str, stats: Counter, warn) -> None:
    if mode != "func_desc":
        for c in ch["fn_desc"]:
            fn = find_fn(entry, c["name"])
            if fn is None:
                warn(f"function {c['name']} not found (description edit)")
                continue
            r = replace_desc(fn, c["old"], c["new"])
            stats[f"fn_desc_{r}"] += 1
            if r == "mismatch":
                warn(f"description of {c['name']} differs from the reviewed original; not changed")
        for c in ch["param_desc"]:
            sp = props_of(find_fn(entry, c["function"])).get(c["param"])
            if sp is None:
                warn(f"parameter {c['function']}.{c['param']} not found (description edit)")
                continue
            r = replace_desc(sp, c["old"], c["new"])
            stats[f"param_desc_{r}"] += 1
            if r == "mismatch":
                warn(f"description of {c['function']}.{c['param']} differs from the reviewed original; not changed")
    for c in ch["defaults"]:
        sp = props_of(find_fn(entry, c["function"])).get(c["param"])
        if sp is None:
            warn(f"parameter {c['function']}.{c['param']} not found (default removal)")
        elif "default" in sp:
            del sp["default"]
            stats["defaults_removed"] += 1
        else:
            stats["default_already_absent"] += 1
    for c in ch["enums"]:
        sp = props_of(find_fn(entry, c["function"])).get(c["param"])
        if sp is not None and c["scope"] == "items.enum":
            sp = sp.get("items")
        if not isinstance(sp, dict):
            warn(f"{c['function']}.{c['param']} ({c['scope']}) not found (enum edit)")
            continue
        if "enum" in sp and jkey(sp["enum"]) not in (jkey(c["old"]), jkey(c["new"])):
            warn(f"enum of {c['function']}.{c['param']} differs from the reviewed original; overwritten anyway")
        if c["removed"]:
            if "enum" in sp:
                del sp["enum"]
                stats["enums_removed"] += 1
        elif sp.get("enum") != c["new"]:
            sp["enum"] = c["new"]
            stats["enums_edited"] += 1


# ---------------------------------------------------------------- per file
def process_file(src: Path, dst: Path, review: dict, mode: str, check_final: bool, dry_run: bool) -> dict:
    stats: Counter = Counter()
    warnings: list[str] = []
    seen = set()
    out_lines = []
    for line in src.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            out_lines.append(line)
            continue
        entry = json.loads(line)
        stats["entries"] += 1
        ch = review.get(entry["id"])
        if ch is None:
            out_lines.append(line)
            continue
        seen.add(entry["id"])
        if ch["removed"]:
            stats["entries_dropped"] += 1
            continue
        warn = lambda msg, _id=entry["id"]: warnings.append(f"{_id}: {msg}")
        before = json.dumps(entry, ensure_ascii=False)
        apply_query(entry, ch, stats, warn)
        apply_gt(entry, ch, stats, warn)
        apply_schema(entry, ch, mode, stats, warn)
        if ch["final_gt"] is not None and entry.get("ground_truth") != ch["final_gt"]:
            if check_final:
                warn("resulting ground truth differs from the CSV's final_ground_truth")
            else:
                stats["gt_differs_from_reviewed_final"] += 1
        after = json.dumps(entry, ensure_ascii=False)
        if after != before:
            stats["entries_changed"] += 1
            out_lines.append(after)
        else:
            out_lines.append(line)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(out_lines), encoding="utf-8")
    return {"mode": mode, "stats": dict(stats), "warnings": warnings,
            "reviewed_ids_not_in_file": sorted(set(review) - seen)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="bfcl_multiple_fixed_desc", help="Source folder under data/benchmarks/ (default: %(default)s).")
    ap.add_argument("--reviews-dir", type=Path, help="Folder with *.review.csv (default: data/gt_reviews/<benchmark>).")
    ap.add_argument("--output-benchmark", help="Output folder under data/benchmarks/ (default: <benchmark>_gt).")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything.")
    ap.add_argument("--force", action="store_true", help="Overwrite files in an existing output folder.")
    args = ap.parse_args()

    src_root = BENCHMARKS_ROOT / args.benchmark
    rev_dir = args.reviews_dir or REVIEWS_ROOT / args.benchmark
    out_name = args.output_benchmark or f"{args.benchmark}_gt"
    out_root = BENCHMARKS_ROOT / out_name
    if not src_root.is_dir():
        sys.exit(f"Benchmark folder not found: {src_root}")
    csvs = sorted(rev_dir.glob("*.review.csv"))
    if not csvs:
        sys.exit(f"No *.review.csv files in {rev_dir}")
    if out_root.resolve() == src_root.resolve():
        sys.exit("Output folder must differ from the source benchmark folder.")
    if out_root.exists() and any(out_root.rglob("*.jsonl")) and not args.force and not args.dry_run:
        sys.exit(f"{out_root} already has .jsonl files; pass --force to overwrite.")

    # plan: target file -> (review, mode, check_final, review csv)
    plan: dict[Path, tuple] = {}
    no_rebuild: list[Path] = []
    for csv_path in csvs:
        stem = csv_path.name[: -len(".review.csv")]
        lang = stem.split("_")[0]
        review = load_review(csv_path)
        base = src_root / lang / f"{stem}.jsonl"
        if not base.is_file():
            sys.exit(f"{csv_path.name}: matching data file not found: {base}")
        n_rm = sum(c["removed"] for c in review.values())
        print(f"{csv_path.relative_to(PACKAGE_ROOT).as_posix()}: {len(review)} entries with changes ({n_rm} removed)")
        plan[base] = (review, "base", True, csv_path)
        if stem.endswith("_full"):
            for suffix, mode in DERIVED.items():
                p = src_root / lang / f"{stem}{suffix}.jsonl"
                if p.is_file():
                    plan[p] = (review, mode, False, csv_path)
            fff = src_root / lang / f"{stem}{REBUILD_SUFFIX}.jsonl"
            if fff.is_file() and review:
                no_rebuild.append(fff)
    print()

    report, problems = {}, False
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = out_root / rel
        if src in no_rebuild:
            print(f"{rel.as_posix():45s} NOT written: rebuild with build_full_full_full.py")
            continue
        if src not in plan:
            if src.name in ("gt_fix_manifest.json",):
                continue
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            if src.suffix == ".jsonl":
                print(f"{rel.as_posix():45s} copied unchanged")
            continue
        review, mode, check_final, csv_path = plan[src]
        res = process_file(src, dst, review, mode, check_final, args.dry_run)
        res["review_csv"] = csv_path.relative_to(PACKAGE_ROOT).as_posix()
        report[rel.as_posix()] = res
        s = res["stats"]
        print(f"{rel.as_posix():45s} [{mode}] entries {s.get('entries', 0)} -> changed {s.get('entries_changed', 0)}, "
              f"dropped {s.get('entries_dropped', 0)}")
        detail = {k: v for k, v in s.items() if k not in ("entries", "entries_changed", "entries_dropped") and v}
        if detail:
            print("    " + ", ".join(f"{k}={v}" for k, v in sorted(detail.items())))
        if res["reviewed_ids_not_in_file"]:
            print(f"    reviewed ids not in this file (skipped): {', '.join(res['reviewed_ids_not_in_file'])}")
        for w in res["warnings"]:
            problems = True
            print(f"    ! {w}")

    if not args.dry_run:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_benchmark": args.benchmark,
            "review_dir": rev_dir.relative_to(PACKAGE_ROOT).as_posix() if rev_dir.is_relative_to(PACKAGE_ROOT) else str(rev_dir),
            "needs_rebuild": [p.relative_to(src_root).as_posix() for p in no_rebuild],
            "files": report,
        }
        (out_root / "gt_fix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {out_root.relative_to(PACKAGE_ROOT)}")
    else:
        print("\nDry run: nothing written.")
    for p in no_rebuild:
        lang = p.parent.name
        print(f"Next: python scripts/build_full_full_full.py --benchmark {out_name} --locale {lang}")
    if problems:
        print("WARNING: some changes could not be applied cleanly (see '!' lines above).")


if __name__ == "__main__":
    main()
