"""
summarize_robustness_results.py — collect every robustness-run summary into one CSV.

Walks results_robustness/<model>/<category>/<lang>/<error-lvl>/<benchmark>/summary.json
(skipping the "old" directory) and writes one row per summary:

    model_family, model_name, benchmark, language, experiment, error_lvl,
    max_retries_acc, first_attempt_acc, average_attempts, cost

Rows are ordered model_family -> model_name -> benchmark (the category folder, e.g.
bfcl_multiple_translatable; alphabetical) -> language (English first, then
alphabetical) -> experiment (en_baseline, native_query_eng_target,
native_query_native_target, native_qt_func_descs, native_qt_lang_hints,
native_qt_all_cues; unknown variations go last, alphabetically) -> error_lvl.

Usage:
    python scripts/summarize_robustness_results.py
    python scripts/summarize_robustness_results.py --results-dir results_robustness --output results_robustness/summary_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from multilingual_bfcl.experiment_names import EXPERIMENT_ORDER, split_stem  # noqa: E402

FAMILY_PREFIXES = {
    "claude": "Claude",
    "gpt": "GPT",
    "grok": "Grok",
    "kimi": "Kimi",
    "deepseek": "DeepSeek",
}

LANGAUGE_ORDER: list[str] = [
    "en",
    "de",
    "zh",
    "tr",
    "he"
]

MODEL_FAMILY_ORDER: list[str] = [
    "Claude",
    "GPT",
    "Grok",
    "DeepSeek",
    "Kimi"
]

COLUMNS = [
    "model_family",
    "model_name",
    "benchmark",
    "language",
    "experiment",
    "error_lvl",
    "max_retries_acc",
    "first_attempt_acc",
    "avg_attempts_per_success",
    "cost",
]


def model_family(model: str) -> str:
    lowered = model.lower()
    for prefix, family in FAMILY_PREFIXES.items():
        if lowered.startswith(prefix):
            return family
    return model


def experiment_name(benchmark: str, lang: str) -> str:
    """'he_native_qt_func_descs' -> 'native_qt_func_descs'; English keeps 'en_baseline'."""
    return split_stem(benchmark, lang)


def sort_key(row: dict) -> tuple:
    model_family_name = row["model_family"]
    lang = row["language"]
    exp = row["experiment"]

    model_family_rank = MODEL_FAMILY_ORDER.index(model_family_name) if model_family_name in MODEL_FAMILY_ORDER else len(MODEL_FAMILY_ORDER)
    lang_rank = LANGAUGE_ORDER.index(lang) if lang in LANGAUGE_ORDER else len(LANGAUGE_ORDER)
    exp_rank = EXPERIMENT_ORDER.index(exp) if exp in EXPERIMENT_ORDER else len(EXPERIMENT_ORDER)

    return (
        (model_family_rank, model_family_name),
        row["model_name"].lower(),
        row["benchmark"],
        (lang_rank, lang),
        (exp_rank, exp),
        row["error_lvl"],
    )


def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in results_dir.rglob("summary.json"):
        rel = path.relative_to(results_dir)
        if "old" in rel.parts:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary.get("model", rel.parts[0])
        category = summary.get("category", rel.parts[1])
        lang = summary.get("lang_dir", rel.parts[2])
        error_lvl = summary.get("error_lvl", rel.parts[3])
        benchmark = summary.get("benchmark", rel.parts[4])
        rows.append({
            "model_family": model_family(model),
            "model_name": model,
            "benchmark": category,
            "language": lang,
            "experiment": experiment_name(benchmark, lang),
            "error_lvl": error_lvl,
            "max_retries_acc": f'{summary.get("success_rate"):.4f}',
            "first_attempt_acc": f'{summary.get("first_attempt_success_rate"):.4f}',
            "avg_attempts_per_success": f'{summary.get("avg_attempts_per_success"):.2f}',
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
