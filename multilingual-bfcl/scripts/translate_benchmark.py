"""
translate_benchmark.py — Batch translator for multilingual-bfcl benchmarks.

Translates a local benchmark (data/benchmarks/<category>/<source>) into one or more
locales using the provider's native Batch API (via langasync), one batch item per
(test case, locale). Mirrors scripts/classify_benchmark.py.

Levels:
  --level query   translate only the user query; ground truth unchanged.
  --level full    translate query + textual ground-truth parameter values
                  (function names, descriptions, enums, numbers stay in English).
                  Requires possible_answer/<source> to exist.

Outputs (under data/benchmarks/<category>/):
  <locale>_<level>.json                  e.g. he_full.json
  possible_answer/<locale>_<level>.json
(For non-default sources the stem is preserved, e.g.
 eng_base_translatable.json -> he_translatable_full.json.)

Usage:
    # Submit and wait for results:
    python scripts/translate_benchmark.py --category multiple --locales he zh-CN --level full

    # Translate the translatable-only subset, query level:
    python scripts/translate_benchmark.py --category multiple --locales he \
        --source eng_base_translatable.json --level query

    # Submit only, then retrieve later (Anthropic):
    python scripts/translate_benchmark.py --category multiple --locales he --level full --submit-only
    python scripts/translate_benchmark.py --category multiple --retrieve msgbatch_01AbCd...

    # Preview the first prompt without calling the API:
    python scripts/translate_benchmark.py --category multiple --locales he --level full --dry-run

Requirements:
    pip install langasync langchain-anthropic langchain-openai
Environment (multilingual-bfcl/.env):
    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...   # only for --provider openai
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

from multilingual_bfcl.benchmark_builder import (  # noqa: E402
    retrieve_translation,
    translate_benchmark,
)
from multilingual_bfcl.localization.translator import (  # noqa: E402
    DEFAULT_TRANSLATION_MODEL,
    LocalizationLevel,
)


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
    parser.add_argument("--source", default="eng_base.json",
                        help="Input filename under data/benchmarks/<category>/.")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider.")
    parser.add_argument("--model", default=DEFAULT_TRANSLATION_MODEL,
                        help="Model name passed to the provider SDK.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Translate only the first N source entries (per locale).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the first prompt and exit without calling the API.")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit the batch, print the batch ID, then exit. Use --retrieve later.")
    parser.add_argument("--retrieve", metavar="BATCH_ID", default=None,
                        help="Retrieve a previously submitted Anthropic batch and write outputs.")
    args = parser.parse_args()

    if args.submit_only and args.retrieve:
        parser.error("--submit-only and --retrieve are mutually exclusive.")
    if args.retrieve and args.dry_run:
        parser.error("--dry-run has no effect with --retrieve.")

    # Retrieve-only path — no asyncio, selection read from the manifest.
    if args.retrieve:
        if args.provider != "anthropic":
            parser.error("--retrieve only supports --provider anthropic.")
        retrieve_translation(args.retrieve, args.category)
        return

    if not args.locales:
        parser.error("--locales is required unless using --retrieve.")

    asyncio.run(translate_benchmark(
        category=args.category,
        locales=args.locales,
        level=LocalizationLevel(args.level),
        source=args.source,
        provider=args.provider,
        model_name=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
        submit_only=args.submit_only,
    ))


if __name__ == "__main__":
    main()
