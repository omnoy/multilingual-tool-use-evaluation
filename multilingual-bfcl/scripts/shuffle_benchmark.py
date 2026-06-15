"""
shuffle_benchmark.py — Build "shuffled" benchmarks that mix two localization
levels of the same translated benchmark at the parameter level.

Given two versions of a translated benchmark that differ only in how much was
localized — typically:

    query-level : question translated, input parameters left in English
    full-level  : question translated AND input parameters translated

this writes two new benchmarks, each keeping one version's question text but
randomly mixing the ground-truth input parameters between the two versions:

    <locale>_<prefix>_full_shuffled.json
        question  : taken verbatim from the *full* benchmark
        parameters: each top-level parameter randomly drawn from full or query,
                    guaranteed to contain at least one parameter from *query*

    <locale>_<prefix>_query_shuffled.json
        question  : taken verbatim from the *query* benchmark
        parameters: each top-level parameter randomly drawn from full or query,
                    guaranteed to contain at least one parameter from *full*

So a single ground-truth function call may end up with some parameters in
English and others in the target language. The "at least one from the other
version" guarantee prefers a parameter whose value actually differs between the
two versions, so the mix is observable; if an entry has no differing parameters
(e.g. all-numeric), it is unavoidably identical and the guarantee is trivially
met.

Outputs preserve the newline-delimited JSON format and original ordering:

    data/benchmarks/<category>/<locale>_<prefix>_query.json   (source)
    data/benchmarks/<category>/<locale>_<prefix>_full.json    (source)
        -> data/benchmarks/<category>/<locale>_<prefix>_full_shuffled.json
        -> data/benchmarks/<category>/<locale>_<prefix>_query_shuffled.json
    (and the matching possible_answer/ files)

The two source benchmarks must describe the same test cases (same ids, function
defs and ground-truth structure); only the localized text should differ. ids are
matched after stripping a trailing "_<locale>" suffix, so the two sources may use
either the bare ("multiple_2") or suffixed ("multiple_2_he") id convention.

Usage (defaults: he_translatable_query/full -> he_translatable_{full,query}_shuffled):
    python scripts/shuffle_benchmark.py
    python scripts/shuffle_benchmark.py --locale ar --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent           # multilingual-bfcl/
DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a file where each line is a JSON object."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as newline-delimited JSON, matching the source format."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def base_id(record_id: str, locale: str) -> str:
    """Strip a trailing "_<locale>" so the two sources can be matched regardless
    of whether they use the bare or locale-suffixed id convention."""
    return re.sub(rf"_{re.escape(locale)}$", "", record_id)


def index_by_base(records: list[dict[str, Any]], locale: str) -> dict[str, dict]:
    return {base_id(r["id"], locale): r for r in records}


def write_question_file(src: Path, dst: Path, out_level: str) -> int:
    """Write a benchmark whose questions/function defs come verbatim from `src`,
    relabeling localization_level. (The two source benchmarks share identical
    function defs; only the question text differs between them.)"""
    records = load_jsonl(src)
    for rec in records:
        rec["localization_level"] = out_level
    write_jsonl(records, dst)
    return len(records)


def _value(gt: list[dict], ci: int, fn: str, key: str, fallback: Any) -> Any:
    """Look up a parameter's value list in a ground-truth structure, defensively."""
    try:
        return gt[ci][fn][key]
    except (KeyError, IndexError):
        return fallback


def shuffle_answers(
    pa_a: Path,
    pa_b: Path,
    dst: Path,
    locale: str,
    guaranteed: str,
    rng: random.Random,
) -> tuple[int, int]:
    """Write a possible_answer file mixing parameter values from versions A and B.

    `pa_a`/`pa_b` are the two possible_answer files; `guaranteed` is "a" or "b"
    and names the version that must contribute at least one parameter per entry
    (preferring a parameter whose value actually differs between the versions).

    Returns (entries_written, entries_with_observable_mix).
    """
    recs_a = load_jsonl(pa_a)
    idx_b = index_by_base(load_jsonl(pa_b), locale)

    out: list[dict[str, Any]] = []
    n_mixed = 0
    for ans_a in recs_a:
        bid = base_id(ans_a["id"], locale)
        ans_b = idx_b.get(bid)
        if ans_b is None:
            print(f"[WARN] {ans_a['id']}: missing from {pa_b.name}, skipping.",
                  file=sys.stderr)
            continue

        gt_a = ans_a.get("ground_truth", [])
        gt_b = ans_b.get("ground_truth", [])

        # Enumerate every top-level parameter slot, using A's structure as the
        # reference (the two versions share the same call/param structure).
        slots: list[tuple[int, str, str]] = []
        for ci, call_a in enumerate(gt_a):
            for fn, params_a in call_a.items():
                for key in params_a:
                    slots.append((ci, fn, key))

        # Random per-slot source, then enforce the "at least one from `guaranteed`"
        # guarantee on a differing slot when one exists (so the mix is observable).
        source = {slot: ("a" if rng.random() < 0.5 else "b") for slot in slots}

        def differs(slot: tuple[int, str, str]) -> bool:
            ci, fn, key = slot
            va = _value(gt_a, ci, fn, key, None)
            vb = _value(gt_b, ci, fn, key, va)
            return json.dumps(va, ensure_ascii=False) != json.dumps(vb, ensure_ascii=False)

        differing = [s for s in slots if differs(s)]
        if differing:
            if not any(source[s] == guaranteed for s in differing):
                source[rng.choice(differing)] = guaranteed
            n_mixed += 1

        # Build the merged ground truth from the chosen sources.
        merged_gt: list[dict[str, Any]] = []
        for ci, call_a in enumerate(gt_a):
            merged_call: dict[str, Any] = {}
            for fn, params_a in call_a.items():
                merged_params: dict[str, Any] = {}
                for key in params_a:
                    gt = gt_a if source[(ci, fn, key)] == "a" else gt_b
                    chosen = _value(gt, ci, fn, key, params_a[key])
                    merged_params[key] = copy.deepcopy(chosen)
                merged_call[fn] = merged_params
            merged_gt.append(merged_call)

        merged = copy.deepcopy(ans_a)
        merged["id"] = f"{bid}_{locale}"
        merged["ground_truth"] = merged_gt
        out.append(merged)

    write_jsonl(out, dst)
    return len(out), n_mixed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full_shuffled and query_shuffled benchmarks that mix "
                    "ground-truth parameters between two localization levels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category", default="multiple",
                        help="Benchmark category folder under data/benchmarks/.")
    parser.add_argument("--locale", default="he",
                        help="Locale code used in filenames and id suffixes.")
    parser.add_argument("--prefix", default="translatable",
                        help="Filename middle segment: <locale>_<prefix>_<level>.json.")
    parser.add_argument("--level-query", default="query",
                        help="Level name of the query-translated source.")
    parser.add_argument("--level-full", default="full",
                        help="Level name of the fully-translated source.")
    parser.add_argument("--subdir", default=None,
                        help="Locale subfolder under the category (e.g. 'heb'). "
                             "Default: auto-detect; pass '' to force the flat layout.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffling.")
    args = parser.parse_args()

    bench = DATA_ROOT / args.category
    if not bench.is_dir():
        sys.exit(f"ERROR: {bench} not found.")
    answers_dir = bench / "possible_answer"

    def fname(level: str) -> str:
        return f"{args.locale}_{args.prefix}_{level}.json"

    q_level, f_level = args.level_query, args.level_full

    # Resolve the (optional) locale subfolder. Files may live either directly
    # under the category or under a locale subfolder like 'heb/'.
    subdir = args.subdir
    if subdir is None:
        if (bench / fname(q_level)).exists():
            subdir = ""
        else:
            cands = [d.name for d in sorted(bench.iterdir())
                     if d.is_dir() and d.name != "possible_answer"
                     and (d / fname(q_level)).exists()]
            if not cands:
                sys.exit(
                    f"ERROR: could not find {fname(q_level)} under {bench} or any "
                    f"subfolder. Use --subdir to point at the locale folder."
                )
            subdir = cands[0]
            print(f"Auto-detected locale subfolder: {subdir!r}\n")

    bench_dir = bench / subdir if subdir else bench
    answers_subdir = answers_dir / subdir if subdir else answers_dir

    src_q_bench, src_f_bench = bench_dir / fname(q_level), bench_dir / fname(f_level)
    src_q_ans, src_f_ans = answers_subdir / fname(q_level), answers_subdir / fname(f_level)

    rng = random.Random(args.seed)

    # Use a single RNG so the whole run is reproducible from --seed. Generate the
    # full_shuffled variant first, then query_shuffled.
    variants = [
        # out_level,            question_src,  answer "a"=query, "b"=full, guaranteed
        (f"{f_level}_shuffled", src_f_bench, "a"),   # full question, guarantee a (query)
        (f"{q_level}_shuffled", src_q_bench, "b"),   # query question, guarantee b (full)
    ]

    print(f"Seed: {args.seed}\n")
    for out_level, question_src, guaranteed in variants:
        out_bench = bench_dir / fname(out_level)
        out_ans = answers_subdir / fname(out_level)

        n_q = write_question_file(question_src, out_bench, out_level)
        n_a, n_mixed = shuffle_answers(
            src_q_ans, src_f_ans, out_ans, args.locale, guaranteed, rng,
        )
        guaranteed_level = q_level if guaranteed == "a" else f_level
        print(f"[{out_level}]")
        print(f"  question from : {question_src.name}")
        print(f"  guarantee     : >=1 param from '{guaranteed_level}'")
        print(f"  {n_q} entries -> {out_bench}")
        print(f"  {n_a} answers -> {out_ans}  ({n_mixed} with an observable mix)")
        if n_q != n_a:
            print(f"  [WARN] entry/answer count mismatch: {n_q} vs {n_a}",
                  file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
