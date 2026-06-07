"""
Benchmark builder: translates BFCL datasets into target locales.

Usage (programmatic):
    from multilingual_bfcl.benchmark_builder import build_benchmark
    build_benchmark("simple_python", ["he", "zh-CN"], level=LocalizationLevel.QUERY_ONLY)

The builder reads source data from the BFCL package's data directory and writes
translated files to multilingual-bfcl/data/benchmarks/<category>/<locale>.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic
from tqdm import tqdm

from multilingual_bfcl.localization.locale_config import SUPPORTED_LOCALES, get_locale
from multilingual_bfcl.localization.translator import LocalizationLevel, translate_test_case

# Root of the multilingual-bfcl package (two levels up from this file)
_PACKAGE_ROOT = Path(__file__).parent.parent.parent
_BENCHMARK_DIR = _PACKAGE_ROOT / "data" / "benchmarks"

# BFCL data directory (resolved from the installed package)
def _bfcl_data_dir() -> Path:
    import bfcl_eval
    return Path(bfcl_eval.__file__).parent / "data"


def _source_path(category: str) -> Path:
    """Return the path to the BFCL source JSON for a category."""
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    data_dir = _bfcl_data_dir()
    filename = f"{VERSION_PREFIX}_{category}.json"
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"BFCL source file not found: {path}\n"
            f"Make sure bfcl_eval is installed and the category name is correct."
        )
    return path


def load_source(category: str) -> list[dict[str, Any]]:
    """Load and return all test cases for a BFCL category."""
    path = _source_path(category)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def output_path(category: str, locale_code: str) -> Path:
    return _BENCHMARK_DIR / category / f"{locale_code}.json"


def build_benchmark(
    category: str,
    locales: list[str],
    level: LocalizationLevel = LocalizationLevel.QUERY_ONLY,
    limit: int | None = None,
    force: bool = False,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Path]:
    """
    Translate a BFCL category into one or more target locales.

    Args:
        category:  BFCL category name, e.g. "simple_python".
        locales:   List of BCP-47 locale codes, e.g. ["he", "zh-CN"].
        level:     LocalizationLevel — QUERY_ONLY or FULL.
        limit:     If set, only translate the first N test cases (useful for testing).
        force:     Re-translate even if the output file already exists.
        client:    Optional pre-constructed Anthropic client.

    Returns:
        A dict mapping locale_code → output Path for each locale processed.
    """
    if client is None:
        client = anthropic.Anthropic()

    source = load_source(category)
    if limit is not None:
        source = source[:limit]

    outputs: dict[str, Path] = {}

    for locale_code in locales:
        locale = get_locale(locale_code)
        out = output_path(category, locale_code)

        if out.exists() and not force:
            print(f"[skip] {out} already exists (use --force to overwrite)")
            outputs[locale_code] = out
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        translated: list[dict[str, Any]] = []

        for test_case in tqdm(source, desc=f"{category} → {locale.name}"):
            translated.append(
                translate_test_case(test_case, locale, level, client=client)
            )

        with open(out, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        print(f"[done] {len(translated)} cases written to {out}")
        outputs[locale_code] = out

    return outputs


def list_available_categories() -> list[str]:
    """Return all BFCL category names for which a source file exists."""
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    data_dir = _bfcl_data_dir()
    prefix = f"{VERSION_PREFIX}_"
    return sorted(
        p.stem[len(prefix):]
        for p in data_dir.glob(f"{prefix}*.json")
    )


def list_built_benchmarks() -> dict[str, list[str]]:
    """
    Return a dict of {category: [locale_codes]} for already-built benchmarks.
    """
    result: dict[str, list[str]] = {}
    if not _BENCHMARK_DIR.exists():
        return result
    for cat_dir in sorted(_BENCHMARK_DIR.iterdir()):
        if cat_dir.is_dir():
            locales = sorted(p.stem for p in cat_dir.glob("*.json"))
            if locales:
                result[cat_dir.name] = locales
    return result
