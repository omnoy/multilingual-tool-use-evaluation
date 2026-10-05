"""
Token pricing helpers for cost reporting.

Prices are USD per 1M tokens, read from model_prices.csv (rows `model,input,output`)
at the package root. The CSV holds list (synchronous) prices; the Batch API discount
is applied here, at estimate time, via `batch=True`.
"""

from __future__ import annotations

import csv
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parent.parent.parent
DEFAULT_PRICES_CSV = _PACKAGE_ROOT / "model_prices.csv"

# Azure OpenAI Global Batch is billed at 50% of the standard per-token price.
BATCH_DISCOUNT = 0.5


def load_price_table(path: Path = DEFAULT_PRICES_CSV) -> dict[str, tuple[float, float]]:
    """Parse a prices CSV into {model: (input_per_mtok, output_per_mtok)}.

    Blank lines, a `model,...` header, and lines starting with '#' are ignored.
    Missing file -> empty table (everything unpriced).
    """
    table: dict[str, tuple[float, float]] = {}
    if not path.exists():
        return table
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 3 or row[0].strip().lower() == "model":
                continue
            try:
                table[row[0].strip()] = (float(row[1]), float(row[2]))
            except ValueError:
                continue
    return table


def lookup_price(
    model: str, table: dict[str, tuple[float, float]] | None = None
) -> tuple[float, float] | None:
    """Exact match on the deployment name, else the longest listed substring of it."""
    table = load_price_table() if table is None else table
    if model in table:
        return table[model]
    matches = [(k, v) for k, v in table.items() if k in model]
    if not matches:
        return None
    return max(matches, key=lambda kv: len(kv[0]))[1]


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int, *, batch: bool = False
) -> float | None:
    """USD cost for the given token counts, or None if the model isn't priced."""
    prices = lookup_price(model)
    if prices is None:
        return None
    in_price, out_price = prices
    cost = input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
    return cost * BATCH_DISCOUNT if batch else cost


# Lazily loaded tiktoken encoder (False once loading has failed).
_ENCODER = None


def count_tokens(text: str) -> int:
    """Approximate token count (tiktoken o200k_base; ~4 chars/token if unavailable)."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENCODER = False
    if _ENCODER:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


def format_cost(model: str, input_tokens: int, output_tokens: int, *, batch: bool = False) -> str:
    """One-line token + cost summary, e.g. for console output."""
    cost = estimate_cost_usd(model, input_tokens, output_tokens, batch=batch)
    tokens = f"{input_tokens:,} input + {output_tokens:,} output tokens"
    if cost is None:
        return f"{tokens} (no price for {model!r} in {DEFAULT_PRICES_CSV.name})"
    note = f", {int(BATCH_DISCOUNT * 100)}% batch discount applied" if batch else ""
    return f"{tokens} -> ${cost:.4f}{note}"
