"""
run_robustness_sweep.py — Run run_robustness_eval.py over many models x benchmark variants.

Replaces one IDE run configuration per (model, benchmark) pair: the model roster (name ->
--model-type) lives in MODELS below, and the benchmark variants are discovered from
data/benchmarks/<category>/<lang-dir>/*.jsonl. Each model is a separate
run_robustness_eval.py subprocess that gets all of its benchmarks at once (--benchmark is
nargs="+"), so one model failing does not stop the others.

Any argument not recognised here is forwarded verbatim to run_robustness_eval.py
(e.g. --resume, --concurrency 8, --max-api-calls 12, --recompute-stats).

Usage:
    # Every model in MODELS on every German variant, full files:
    python scripts/run_robustness_sweep.py --lang-dir de

    # A subset of models and variants (variant = file stem minus the "<lang>_" prefix):
    python scripts/run_robustness_sweep.py --lang-dir de \
        --models gpt-6-astra Kimi-K2.6 --variants translatable_query translatable_full

    # Finish runs that stopped early (forwarded to the harness):
    python scripts/run_robustness_sweep.py --lang-dir de --resume

    # Run every (model, benchmark) pair as its own concurrent process:
    python scripts/run_robustness_sweep.py --lang-dir de --parallel

    # Just print the commands:
    python scripts/run_robustness_sweep.py --lang-dir de --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PACKAGE_ROOT / "scripts" / "run_robustness_eval.py"

# Azure deployment name -> --model-type (see azure_client.ModelType).
MODELS: dict[str, str] = {
    "gpt-6-astra": "responses",
    "gpt-5.6-sol": "reasoning",
    "gpt-5.6-terra": "reasoning",
    "claude-opus-5": "claude",
    "claude-opus-5-5": "claude",
    "Kimi-K2.6": "reasoning",
    "grok-4.6": "reasoning",
}


def discover_benchmarks(category: str, lang_dir: str, variants: list[str] | None) -> list[str]:
    data_dir = PACKAGE_ROOT / "data" / "benchmarks" / category / lang_dir
    stems = sorted(p.stem for p in data_dir.glob("*.jsonl"))
    if not stems:
        sys.exit(f"No benchmark files found in {data_dir}")
    if variants is None:
        return stems
    prefix = f"{lang_dir}_translatable_"
    selected = [s for s in stems if s.removeprefix(prefix) in variants or s in variants]
    missing = set(variants) - {s.removeprefix(prefix) for s in selected} - set(selected)
    if missing:
        sys.exit(f"Unknown variant(s) {sorted(missing)}; available: {stems}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run_robustness_eval.py over many models x benchmark variants. "
                    "Unrecognised args are forwarded to run_robustness_eval.py.",
    )
    parser.add_argument("--lang-dir", required=True, help="Language subfolder, e.g. de, he.")
    parser.add_argument("--category", default="bfcl_multiple")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help=f"Subset of MODELS to run (default: all). Known: {list(MODELS)}")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Benchmark variants (file stem with or without the '<lang>_' "
                             "prefix). Default: every .jsonl in the language folder.")
    parser.add_argument("--limit", default="0",
                        help="Forwarded --limit (default 0 = full file).")
    parser.add_argument("--parallel", action="store_true",
                        help="Run every (model, benchmark) pair as its own concurrent process.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args, passthrough = parser.parse_known_args()

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"Unknown model(s) {unknown}; add them to MODELS. Known: {list(MODELS)}")

    benchmarks = discover_benchmarks(args.category, args.lang_dir, args.variants)
    print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")

    # Sequential: one process per model running all its benchmarks back to back.
    # Parallel: one process per (model, benchmark) so every pair runs concurrently.
    jobs = (
        [(m, [b]) for m in args.models for b in benchmarks] if args.parallel
        else [(m, benchmarks) for m in args.models]
    )
    commands = {
        f"{model} / {' '.join(bms) if args.parallel else 'all'}": [
            sys.executable, str(EVAL_SCRIPT),
            "--category", args.category,
            "--lang-dir", args.lang_dir,
            "--benchmark", *bms,
            "--model", model,
            "--model-type", MODELS[model],
            "--limit", args.limit,
            *passthrough,
        ]
        for model, bms in jobs
    }

    if args.dry_run:
        for cmd in commands.values():
            print(subprocess.list2cmdline(cmd))
        return

    failed: list[str] = []
    if args.parallel:
        print(f"Launching {len(commands)} processes in parallel.", flush=True)
        procs = {job: subprocess.Popen(cmd, cwd=PACKAGE_ROOT) for job, cmd in commands.items()}
        failed = [job for job, p in procs.items() if p.wait() != 0]
    else:
        for job, cmd in commands.items():
            print(f"\n=== {job} ===", flush=True)
            if subprocess.run(cmd, cwd=PACKAGE_ROOT).returncode != 0:
                failed.append(job)

    print(f"\nDone. {len(commands) - len(failed)}/{len(commands)} jobs succeeded.")
    if failed:
        print(f"Failed: {failed} — rerun those with --resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
