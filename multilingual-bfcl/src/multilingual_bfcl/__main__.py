"""
CLI entry point for multilingual-bfcl.

Commands:
  mbfcl build   -- translate BFCL datasets into target locales
  mbfcl locales -- list supported locales
  mbfcl status  -- show which benchmarks have been built

Usage examples:
  mbfcl build bfcl_multiple --locales he zh-CN --level query
  mbfcl build bfcl_multiple --locales he --level full --source eng_translatable.json
  mbfcl build bfcl_multiple --retrieve batch_01AbCd...
  mbfcl locales
  mbfcl status
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

app = typer.Typer(
    name="mbfcl",
    help="Multilingual Berkeley Function Calling Leaderboard toolkit",
    no_args_is_help=True,
)


class Level(str, Enum):
    query = "query"
    full = "full"


@app.command()
def build(
    category: str = typer.Argument(..., help="Benchmark category folder under data/benchmarks/"),
    locales: list[str] = typer.Option(None, "--locales", "-l", help="Space-separated locale codes, e.g. he zh-CN"),
    level: Level = typer.Option(Level.query, "--level", help="Translation scope: query or full"),
    source: str = typer.Option("eng_base.json", "--source", help="Input filename under data/benchmarks/<category>/"),
    model: Optional[str] = typer.Option(None, "--model", help="Azure deployment name (defaults to AZURE_OPENAI_DEPLOYMENT)"),
    model_type: str = typer.Option("standard", "--model-type", help="Deployment kind: 'standard' or 'reasoning'"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Translate only the first N source entries"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the first prompt and exit"),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit the batch and exit; retrieve later"),
    retrieve: Optional[str] = typer.Option(None, "--retrieve", help="Retrieve a previously submitted batch id"),
) -> None:
    """Batch-translate a benchmark category into one or more target locales."""
    import asyncio

    from multilingual_bfcl.azure_client import ModelType
    from multilingual_bfcl.benchmark_builder import retrieve_translation, translate_benchmark
    from multilingual_bfcl.localization.translator import (
        DEFAULT_TRANSLATION_MODEL,
        LocalizationLevel,
    )

    if retrieve:
        retrieve_translation(retrieve, category)
        return
    if not locales:
        raise typer.BadParameter("--locales is required unless using --retrieve.")

    asyncio.run(translate_benchmark(
        category=category,
        locales=locales,
        level=LocalizationLevel(level.value),
        source=source,
        model_name=model or DEFAULT_TRANSLATION_MODEL,
        model_type=ModelType(model_type),
        limit=limit,
        dry_run=dry_run,
        submit_only=submit_only,
    ))


@app.command()
def descriptors(
    source: str = typer.Argument(..., help="Path to a built benchmark .jsonl (e.g. data/benchmarks/bfcl_multiple/he/he_translatable_full.jsonl)"),
    suffix: str = typer.Option("langdesc", "--suffix", help="Suffix for the output filename (<stem>_<suffix>.jsonl)"),
) -> None:
    """Append a 'Language: <langs>.' descriptor to natural-language parameters.

    Detects natural-language parameters from each entry's ground truth (numbers,
    dates, booleans and enums are excluded) and annotates their descriptions with
    the languages their value may appear in (English + the entry's locale). Writes
    a new benchmark file alongside the source.
    """
    from pathlib import Path

    from multilingual_bfcl.benchmark_builder import add_language_descriptors_file

    add_language_descriptors_file(Path(source), suffix=suffix)


@app.command()
def locales() -> None:
    """List all supported locales."""
    from multilingual_bfcl.localization.locale_config import SUPPORTED_LOCALES

    rows = [(code, loc.name, "RTL" if loc.rtl else "LTR")
            for code, loc in SUPPORTED_LOCALES.items()]
    max_code = max(len(r[0]) for r in rows)
    max_name = max(len(r[1]) for r in rows)
    typer.echo(f"{'Code':<{max_code + 2}}{'Name':<{max_name + 2}}Direction")
    typer.echo("-" * (max_code + max_name + 14))
    for code, name, direction in rows:
        typer.echo(f"{code:<{max_code + 2}}{name:<{max_name + 2}}{direction}")


@app.command()
def status() -> None:
    """Show which benchmark/locale combinations have been built."""
    from multilingual_bfcl.benchmark_builder import list_built_benchmarks

    built = list_built_benchmarks()
    if not built:
        typer.echo("No benchmarks built yet. Run `mbfcl build` to create some.")
        return
    for category, locale_codes in built.items():
        typer.echo(f"  {category}: {', '.join(locale_codes)}")


@app.command(name="categories")
def list_categories() -> None:
    """List BFCL categories available for translation."""
    from multilingual_bfcl.benchmark_builder import list_available_categories

    for cat in list_available_categories():
        typer.echo(f"  {cat}")


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
