"""
CLI entry point for multilingual-bfcl.

Commands:
  mbfcl build   -- translate BFCL datasets into target locales
  mbfcl locales -- list supported locales
  mbfcl status  -- show which benchmarks have been built

Usage examples:
  mbfcl build simple_python --locales he zh-CN --level query_only
  mbfcl build simple_python --locales he --limit 10 --force
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
    query_only = "query_only"
    full = "full"


@app.command()
def build(
    category: str = typer.Argument(..., help="BFCL category name, e.g. 'simple_python'"),
    locales: list[str] = typer.Option(..., "--locales", "-l", help="Space-separated locale codes, e.g. he zh-CN"),
    level: Level = typer.Option(Level.query_only, "--level", help="Localization level"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Translate only the first N test cases"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output files"),
) -> None:
    """Translate a BFCL category into one or more target locales."""
    from multilingual_bfcl.benchmark_builder import build_benchmark
    from multilingual_bfcl.localization.translator import LocalizationLevel

    lv = LocalizationLevel(level.value)
    outputs = build_benchmark(category, locales, level=lv, limit=limit, force=force)
    for locale_code, path in outputs.items():
        typer.echo(f"  {locale_code}: {path}")


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
