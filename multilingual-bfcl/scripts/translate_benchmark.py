"""
translate_benchmark.py — Batch translator for multilingual-bfcl benchmarks.

Translates a local benchmark (data/benchmarks/<category>/<source>, default
en/en_translatable.jsonl) into one or more locales using the Azure OpenAI Batch API,
one batch item per (test case, locale). Each entry carries its own `ground_truth`,
which is written (translated or not, per level) into the output entry.

Levels:
  --level query   translate only the user query; ground truth unchanged.
  --level full    translate query + textual ground-truth parameter values, including
                  strings nested in array/object values (e.g. [["Sam", "Alice"]])
                  (function names, descriptions, enums, numbers stay in English).

Outputs (under data/benchmarks/<category>/):
  <locale>/<locale>_translatable_<level>.jsonl   e.g. he/he_translatable_full.jsonl

Usage:
    # Submit and wait for results:
    python scripts/translate_benchmark.py --category bfcl_multiple --locales he zh-CN --level full --model gpt-5.6-terra

    # Translate the translatable-only subset, query level:
    python scripts/translate_benchmark.py --category bfcl_multiple --locales he \
        --source en/en_translatable.jsonl --level query --model gpt-5.6-terra

    # Submit only, then retrieve later:
    python scripts/translate_benchmark.py --category bfcl_multiple --locales he --level full --model gpt-5.6-terra --submit-only
    python scripts/translate_benchmark.py --category bfcl_multiple --retrieve batch_01AbCd...

    # No Batch-SKU deployment available? Send plain concurrent requests (full price):
    python scripts/translate_benchmark.py --category bfcl_multiple --locales he --level full --model gpt-5.6-terra --no-batch

    # Preview the first prompt and the cost estimate without calling the API:
    python scripts/translate_benchmark.py --category bfcl_multiple --locales he --level full --model gpt-5.6-terra --dry-run

Cost: an estimate (tiktoken-based) is printed before submitting, and the actual cost
(from the API's reported token usage) after retrieval. Both use model_data.csv list
prices, with the Batch API discount applied unless --no-batch.

Requirements:
    pip install openai langchain-openai
Environment (multilingual-bfcl/.env):
    AZURE_OPENAI_ENDPOINT=...      # …/openai/v1 (or the resource root)
    AZURE_OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the package importable when run as a plain script.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
load_dotenv(PACKAGE_ROOT / ".env")

from multilingual_bfcl.azure_client import OPENAI_MODEL_TYPES, ModelType  # noqa: E402
from multilingual_bfcl.benchmark_builder import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SOURCE,
    retrieve_translation,
    translate_benchmark,
)
from multilingual_bfcl.localization.translator import LocalizationLevel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-translate a multilingual-bfcl benchmark into target locales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category", required=True,
                        help="Benchmark category folder under data/benchmarks/.")
    parser.add_argument("--locales", nargs="+", default=None,
                        help="Target locale codes, e.g. he zh-CN. Required unless --retrieve.")
    parser.add_argument("--level", choices=[lv.value for lv in LocalizationLevel],
                        default=LocalizationLevel.QUERY.value,
                        help="Translation scope: 'query' or 'full'.")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Input .jsonl path relative to data/benchmarks/<category>/.")
    parser.add_argument("--model", required=True,
                        help="Azure deployment name to translate with (no default, "
                             "to prevent accidental runs).")
    parser.add_argument("--model-type", choices=[mt.value for mt in OPENAI_MODEL_TYPES],
                        default=ModelType.STANDARD.value,
                        help="Deployment kind: 'standard' (temperature + max_tokens) or "
                             "'reasoning' (no temperature, max_completion_tokens).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Translate only the first N source entries (per locale).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the first prompt and exit without calling the API.")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit the batch, print the batch ID, then exit. Use --retrieve later.")
    parser.add_argument("--retrieve", metavar="BATCH_ID", default=None,
                        help="Retrieve a previously submitted Azure batch and write outputs.")
    parser.add_argument("--no-batch", action="store_true",
                        help="Skip the Batch API: send plain concurrent chat.completions "
                             "requests (for deployments without a Global/Data Zone Batch "
                             "SKU). Full price, results written immediately.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="--no-batch only: max requests in flight.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="--no-batch only: SDK retries per request (backoff on 429/5xx).")
    args = parser.parse_args()

    if args.submit_only and args.retrieve:
        parser.error("--submit-only and --retrieve are mutually exclusive.")
    if args.retrieve and args.dry_run:
        parser.error("--dry-run has no effect with --retrieve.")
    if args.no_batch and (args.submit_only or args.retrieve):
        parser.error("--no-batch cannot be combined with --submit-only or --retrieve.")

    # Retrieve-only path — no asyncio, selection read from the manifest.
    if args.retrieve:
        retrieve_translation(args.retrieve, args.category)
        return

    if not args.locales:
        parser.error("--locales is required unless using --retrieve.")

    asyncio.run(translate_benchmark(
        category=args.category,
        locales=args.locales,
        level=LocalizationLevel(args.level),
        source=args.source,
        model_name=args.model,
        model_type=ModelType(args.model_type),
        limit=args.limit,
        dry_run=args.dry_run,
        submit_only=args.submit_only,
        use_batch=not args.no_batch,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    ))


if __name__ == "__main__":
    main()
