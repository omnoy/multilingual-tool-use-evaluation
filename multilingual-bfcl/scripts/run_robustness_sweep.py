"""
run_robustness_sweep.py — Run run_robustness_eval.py over many models x benchmark variants.

Replaces one IDE run configuration per (model, benchmark) pair: the model roster (name ->
--model-type) is the model_type column of model_data.csv, and the benchmark variants are discovered from
data/benchmarks/<category>/<lang-dir>/*.jsonl. Each model is a separate
run_robustness_eval.py subprocess that gets all of its benchmarks at once (--benchmark is
nargs="+"), so one model failing does not stop the others.

Any argument not recognised here is forwarded verbatim to run_robustness_eval.py
(e.g. --resume, --concurrency 8, --max-api-calls 12).

Usage:
    # Every model in MODELS on every German variant, full files:
    python scripts/run_robustness_sweep.py --lang-dir de

    # A subset of models and variants (variant = file stem minus the "<lang>_" prefix):
    python scripts/run_robustness_sweep.py --lang-dir de \
        --models gpt-6-astra Kimi-K2.6 --variants native_query_eng_target native_query_native_target

    # Finish runs that stopped early (forwarded to the harness):
    python scripts/run_robustness_sweep.py --lang-dir de --resume

    # Rebuild statistics.csv + summary.json from saved transcripts, no API calls. Pairs
    # with no transcripts are skipped:
    python scripts/run_robustness_sweep.py --lang-dir de --recompute-stats

    # Just print the commands:
    python scripts/run_robustness_sweep.py --lang-dir de --dry-run

    # Several languages and categories at once (missing combinations are skipped):
    python scripts/run_robustness_sweep.py --lang-dir en he de tr \
        --category bfcl_multiple_translatable wildtoolbench
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PACKAGE_ROOT / "scripts" / "run_robustness_eval.py"

MODEL_DATA_CSV = PACKAGE_ROOT / "model_data.csv"


def load_model_types(path: Path = MODEL_DATA_CSV) -> dict[str, str]:
    """Azure deployment name -> --model-type (see azure_client.ModelType), read from the
    `model_type` column of model_data.csv. Rows with a blank model_type are not in the
    sweep roster."""
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            row["model"].strip(): row["model_type"].strip()
            for row in csv.DictReader(fh)
            if (row.get("model_type") or "").strip()
        }


MODELS: dict[str, str] = load_model_types()


def discover_benchmarks(category: str, lang_dir: str, variants: list[str] | None) -> tuple[list[str], set[str]]:
    """Return (selected benchmark stems, the requested variants they matched). An empty or
    missing folder yields ([], set()) — with several categories/languages not every
    combination exists (e.g. en only has en_baseline)."""
    data_dir = PACKAGE_ROOT / "data" / "benchmarks" / category / lang_dir
    stems = sorted(p.stem for p in data_dir.glob("*.jsonl"))
    if variants is None:
        return stems, set()
    prefix = f"{lang_dir}_"
    selected = [s for s in stems if s.removeprefix(prefix) in variants or s in variants]
    matched = {v for v in variants for s in selected if v in (s, s.removeprefix(prefix))}
    return selected, matched


def has_transcripts(output_dir: str, model: str, category: str, lang_dir: str, benchmark: str) -> bool:
    transcripts = PACKAGE_ROOT / output_dir / model / category / lang_dir / benchmark / "transcripts"
    return any(transcripts.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run_robustness_eval.py over many models x benchmark variants. "
                    "Unrecognised args are forwarded to run_robustness_eval.py.",
    )
    parser.add_argument("--lang-dir", nargs="+", required=True,
                        help="Language subfolder(s), e.g. --lang-dir en he de tr. Missing "
                             "category/lang combinations are skipped.")
    parser.add_argument("--category", nargs="+", required=True,
                        help="Benchmark category folder(s) under data/benchmarks, e.g. "
                             "--category bfcl_multiple_translatable wildtoolbench.")
    parser.add_argument("--models", nargs="+", required=True,
                        help=f"Subset of MODELS to run (default: all). Known: {list(MODELS)}")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Benchmark variants (file stem with or without the '<lang>_' "
                             "prefix). Default: every .jsonl in the language folder.")
    parser.add_argument("--limit", default="0",
                        help="Forwarded --limit (default 0 = full file).")
    parser.add_argument("--recompute-stats", action="store_true",
                        help="No API calls: rebuild statistics.csv and summary.json from "
                             "existing transcripts. Pairs with no transcripts are skipped.")
    parser.add_argument("--output-dir", default="results_robustness",
                        help="Forwarded --output-dir (relative to the package root).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args, passthrough = parser.parse_known_args()

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"Unknown model(s) {unknown}; set their model_type in model_data.csv. Known: {list(MODELS)}")

    # (category, lang_dir) -> benchmark stems, over every requested combination.
    combo_benchmarks: dict[tuple[str, str], list[str]] = {}
    matched_variants: set[str] = set()
    for category in args.category:
        for lang_dir in args.lang_dir:
            benchmarks, matched = discover_benchmarks(category, lang_dir, args.variants)
            matched_variants |= matched
            if not benchmarks:
                print(f"Skipping {category}/{lang_dir}: no matching benchmark files.")
                continue
            combo_benchmarks[(category, lang_dir)] = benchmarks
            print(f"{category}/{lang_dir} ({len(benchmarks)}): {benchmarks}")
    if args.variants is not None:
        unmatched = sorted(set(args.variants) - matched_variants)
        if unmatched:
            parser.error(f"Variant(s) {unmatched} not found in any of "
                         f"categories {args.category} x lang dirs {args.lang_dir}")
    if not combo_benchmarks:
        sys.exit("No benchmark files found for any category/lang-dir combination.")

    # (model, category, lang_dir, benchmark), one process each.
    jobs = [(m, c, l, b)
            for m in args.models
            for (c, l), bms in combo_benchmarks.items()
            for b in bms]
    if args.recompute_stats:
        # The harness raises on a benchmark with no transcripts, so skip those up front.
        skipped = [j for j in jobs if not has_transcripts(args.output_dir, *j)]
        for m, c, l, b in skipped:
            print(f"Skipping {m} / {c}/{l}/{b} (no transcripts)")
        jobs = [j for j in jobs if j not in skipped]
        passthrough.append("--recompute-stats")

    commands = {
        f"{model} / {category}/{lang_dir}/{benchmark}": [
            sys.executable, str(EVAL_SCRIPT),
            "--category", category,
            "--lang-dir", lang_dir,
            "--benchmark", benchmark,
            "--model", model,
            "--model-type", MODELS[model],
            "--limit", args.limit,
            "--output-dir", args.output_dir,
            *passthrough,
        ]
        for model, category, lang_dir, benchmark in jobs
    }

    if args.dry_run:
        for cmd in commands.values():
            print(subprocess.list2cmdline(cmd))
        return

    failed: list[str] = []
    print(f"Launching {len(commands)} processes in parallel.", flush=True)
    procs = {job: subprocess.Popen(cmd, cwd=PACKAGE_ROOT) for job, cmd in commands.items()}
    failed = [job for job, p in procs.items() if p.wait() != 0]

    print(f"\nDone. {len(commands) - len(failed)}/{len(commands)} jobs succeeded.")
    if failed:
        hint = "" if args.recompute_stats else " — rerun those with --resume"
        print(f"Failed: {failed}{hint}")
        sys.exit(1)


if __name__ == "__main__":
    main()
