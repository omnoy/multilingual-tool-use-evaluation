"""
summarize_robustness_results.py — collect every robustness-run summary into one CSV.

Walks results_robustness/<model>/<category>/<lang>/<benchmark>/summary.json (skipping
the "old" directory) and writes one row per summary:

    model_family, model_name, language, experiment,
    max_retries_acc, first_attempt_acc, average_attempts, cost

Rows are ordered model_family -> model_name -> language (English first, then
alphabetical) -> experiment (en_translatable, query, full, full_func_desc,
full_lang_desc, full_full_full; unknown variations go last, alphabetically).

Usage:
    python scripts/summarize_robustness_results.py
    python scripts/summarize_robustness_results.py --results-dir results_robustness --output results_robustness/summary_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_ORDER = [
    "en_translatable",
    "query",
    "full",
    "full_func_desc",
    "full_lang_desc",
    "full_full_full",
]

FAMILY_PREFIXES = {
    "claude": "Claude",
    "gpt": "OpenAI",
    "grok": "Grok",
    "kimi": "Kimi",
}

COLUMNS = [
    "model_family",
    "model_name",
    "language",
    "experiment",
    "max_retries_acc",
    "first_attempt_acc",
    "average_attempts",
    "cost",
]


def model_family(model: str) -> str:
    lowered = model.lower()
    for prefix, family in FAMILY_PREFIXES.items():
        if lowered.startswith(prefix):
            return family
    return model


def experiment_name(benchmark: str, lang: str) -> str:
    """'he_translatable_full_func_desc' -> 'full_func_desc'; English keeps 'en_translatable'."""
    prefix = f"{lang}_translatable_"
    if benchmark.startswith(prefix):
        return benchmark[len(prefix):]
    return benchmark


def sort_key(row: dict) -> tuple:
    lang = row["language"]
    exp = row["experiment"]
    exp_rank = EXPERIMENT_ORDER.index(exp) if exp in EXPERIMENT_ORDER else len(EXPERIMENT_ORDER)
    return (
        row["model_family"].lower(),
        row["model_name"].lower(),
        (lang != "en", lang),
        (exp_rank, exp),
    )


def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in results_dir.rglob("summary.json"):
        rel = path.relative_to(results_dir)
        if "old" in rel.parts:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary.get("model", rel.parts[0])
        lang = summary.get("lang_dir", rel.parts[2])
        benchmark = summary.get("benchmark", rel.parts[3])
        rows.append({
            "model_family": model_family(model),
            "model_name": model,
            "language": lang,
            "experiment": experiment_name(benchmark, lang),
            "max_retries_acc": f'{summary.get("success_rate"):.4f}',
            "first_attempt_acc": f'{summary.get("first_attempt_success_rate"):.4f}',
            "average_attempts": f'{summary.get("avg_attempts"):.2f}',
            "cost": f'{summary.get("total_estimated_cost_usd"):.3f}',
        })
    rows.sort(key=sort_key)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results_robustness")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default: <results-dir>/summary_table.csv)")
    args = parser.parse_args()

    output = args.output or args.results_dir / "summary_table.csv"
    rows = collect_rows(args.results_dir)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
