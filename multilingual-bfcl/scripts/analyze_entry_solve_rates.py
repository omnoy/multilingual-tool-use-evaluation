"""Per-entry solve-rate analysis across all robustness-eval results.

Walks results_robustness/ (excluding the deprecated results_robustness/old/
subtree), reads every statistics.csv, and aggregates each benchmark entry's
solve rate across models/locales/formats. Also pulls a representative query
and ground_truth for each entry from its transcripts.

Usage:
    python scripts/analyze_entry_solve_rates.py [--results-dir DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_robustness"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results_robustness" / "entry_solve_rates.tsv"

# Priority order (lower = preferred) for picking which transcript to pull the
# representative query / ground_truth from for a given entry.
BENCHMARK_PRIORITY = {
    "en_translatable": 0,
    "he_translatable_full": 1,
    "he_translatable_query": 2,
    "he_translatable_full_func_desc": 3,
    "he_translatable_full_lang_desc": 4,
}


def pct(successes: int, total: int) -> str:
    if total == 0:
        return ""
    return f"{100.0 * successes / total:.2f}"


def find_statistics_files(results_dir: Path):
    for path in sorted(results_dir.rglob("statistics.csv")):
        rel_parts = path.relative_to(results_dir).parts
        if rel_parts and rel_parts[0] == "old":
            continue
        yield path


def load_transcript_query_and_gt(transcript_path: Path):
    with transcript_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    query_text = ""
    for msg in data.get("final_messages", []):
        if msg.get("role") == "user":
            parts = [
                block.get("text", "")
                for block in msg.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            query_text = "\n".join(p for p in parts if p)
            break

    ground_truth = json.dumps(data.get("ground_truth"), ensure_ascii=False)
    return query_text, ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results_dir: Path = args.results_dir

    # Per-entry (canonical source id) accumulators.
    all_results = defaultdict(list)
    en_results = defaultdict(list)
    query_results = defaultdict(list)
    full_results = defaultdict(list)
    full_lang_desc_results = defaultdict(list)
    full_func_desc_results = defaultdict(list)
    error_counts = defaultdict(Counter)

    # Best transcript candidate per entry: (priority, path)
    best_transcript = {}

    stats_files = list(find_statistics_files(results_dir))
    if not stats_files:
        raise SystemExit(f"No statistics.csv files found under {results_dir}")

    for stats_path in stats_files:
        transcripts_dir = stats_path.parent / "transcripts"
        with stats_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_id = row["id"]
                source_id = row.get("source_id") or ""
                canonical_id = source_id if source_id else row_id
                benchmark = row["benchmark"]
                success = row["success"] == "True"

                all_results[canonical_id].append(success)

                if benchmark == "en_translatable":
                    en_results[canonical_id].append(success)
                if benchmark.endswith("query"):
                    query_results[canonical_id].append(success)
                if "full" in benchmark:
                    full_results[canonical_id].append(success)
                if benchmark == "he_translatable_full_lang_desc":
                    full_lang_desc_results[canonical_id].append(success)
                if benchmark == "he_translatable_full_func_desc":
                    full_func_desc_results[canonical_id].append(success)

                if not success:
                    error_type = row.get("final_error_type") or "unknown"
                    error_counts[canonical_id][error_type] += 1

                priority = BENCHMARK_PRIORITY.get(benchmark, 99)
                current = best_transcript.get(canonical_id)
                if current is None or priority < current[0]:
                    best_transcript[canonical_id] = (priority, transcripts_dir / f"{row_id}.json")

    rows = []
    for canonical_id in sorted(all_results.keys(), key=lambda s: (len(s), s)):
        n_total = len(all_results[canonical_id])
        n_en = len(en_results[canonical_id])
        n_query = len(query_results[canonical_id])
        n_full = len(full_results[canonical_id])
        n_full_lang_desc = len(full_lang_desc_results[canonical_id])
        n_full_func_desc = len(full_func_desc_results[canonical_id])

        top_error = ""
        counter = error_counts.get(canonical_id)
        if counter:
            top_error, _ = counter.most_common(1)[0]

        query_text, ground_truth = "", ""
        candidate = best_transcript.get(canonical_id)
        if candidate is not None:
            _, transcript_path = candidate
            if transcript_path.exists():
                query_text, ground_truth = load_transcript_query_and_gt(transcript_path)

        rows.append(
            {
                "id": canonical_id,
                "solve_rate": pct(sum(all_results[canonical_id]), n_total),
                "solve_rate_en": pct(sum(en_results[canonical_id]), n_en),
                "solve_rate_query": pct(sum(query_results[canonical_id]), n_query),
                "solve_rate_full": pct(sum(full_results[canonical_id]), n_full),
                "solve_rate_full_lang_desc": pct(sum(full_lang_desc_results[canonical_id]), n_full_lang_desc),
                "solve_rate_full_func_desc": pct(sum(full_func_desc_results[canonical_id]), n_full_func_desc),
                "n_total": n_total,
                "n_en": n_en,
                "n_query": n_query,
                "n_full": n_full,
                "top_error_type": top_error,
                "query": query_text,
                "ground_truth": ground_truth,
            }
        )

    fieldnames = [
        "id",
        "solve_rate",
        "solve_rate_en",
        "solve_rate_query",
        "solve_rate_full",
        "solve_rate_full_lang_desc",
        "solve_rate_full_func_desc",
        "n_total",
        "n_en",
        "n_query",
        "n_full",
        "top_error_type",
        "query",
        "ground_truth",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {args.out}")


if __name__ == "__main__":
    main()
