"""
CLI entry point for multilingual-bfcl.

Commands:
  mbfcl build   -- translate BFCL datasets into target locales
  mbfcl locales -- list supported locales
  mbfcl status  -- show which benchmarks have been built

Usage examples:
  mbfcl build multiple --locales he zh-CN --level query
  mbfcl build multiple --locales he --level full --source eng_base_translatable.json
  mbfcl build multiple --retrieve msgbatch_01AbCd...
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
    provider: str = typer.Option("anthropic", "--provider", help="LLM provider: anthropic or openai"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name (defaults to the translator's default)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Translate only the first N source entries"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the first prompt and exit"),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit the batch and exit; retrieve later"),
    retrieve: Optional[str] = typer.Option(None, "--retrieve", help="Retrieve a previously submitted batch id"),
) -> None:
    """Batch-translate a benchmark category into one or more target locales."""
    import asyncio

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
        provider=provider,
        model_name=model or DEFAULT_TRANSLATION_MODEL,
        limit=limit,
        dry_run=dry_run,
        submit_only=submit_only,
    ))


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
