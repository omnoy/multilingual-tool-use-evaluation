"""
run_robustness_eval.py — Interactive tool-calling robustness harness for multilingual-bfcl.

Tests whether an LLM can recover from API-side rejections when the user query is in
one language but the API (tool definitions + error messages) is in English. For each
benchmark entry we run a short *conversation*, not a single shot:

  1. Send the (translated) query + the English tool definitions to the model.
  2. Decode the model's function call and check it against the ground-truth answer
     using BFCL's own AST checker (so scoring matches standard BFCL exactly).
  3. If the call is wrong, return a generic English "400 Bad Request" tool result
     and let the model try again — up to --max-attempts times.
  4. The entry succeeds the moment any attempt produces a correct call; otherwise it
     fails after the attempt budget is exhausted.

Scoring is BFCL's own AST checker, with one relaxation enabled by default: a response
that contains the correct call AND extra (superfluous) calls still counts as a pass
(each expected call must be matched by a distinct model call; extras are ignored).
Pass --strict-call-count for standard BFCL behaviour (exact call count required).
The statistics record success_with_extra_calls so the two can be compared.

Why plain async and not the Batch API: each retry depends on the error returned for the
previous attempt, so the turns within an entry are inherently sequential and cannot be
pre-packed into a batch. Entries, however, are independent, so we run them concurrently
with a semaphore cap and rely on ClaudeHandler's built-in rate-limit backoff.

Inputs (under data/benchmarks/<category>/):
  <lang-dir>/<benchmark>.json                  e.g. heb/he_translatable_query.json
  possible_answer/<lang-dir>/<benchmark>.json  ground-truth function calls

Outputs (under <output-dir>/<model>/<category>/<lang-dir>/<benchmark>/):
  transcripts/<entry_id>.json   full reasoning + tool calls + injected errors per entry
  statistics.csv                one row per entry (success, #attempts, #failed calls,
                                tokens, estimated_cost_usd, ...)
  summary.json                  aggregate counts + total_estimated_cost_usd for the run

The cost column is derived from the recorded token counts (Anthropic per-token
pricing). Pass --recompute-stats to rebuild statistics.csv + summary.json (including
cost) from existing transcripts WITHOUT calling the API — useful after a pricing
change or to add the cost column to runs done before it existed.

Usage:
    # Small sample (default 10 entries) against the default model:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query

    # Full file, more retries, higher concurrency:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query \
        --limit 0 --max-attempts 8 --concurrency 8

    # Several benchmark files in one run:
    python scripts/run_robustness_eval.py \
        --benchmark he_translatable_query he_translatable_full

    # Rebuild stats + cost from existing transcripts, no API calls:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query --recompute-stats

Environment (multilingual-bfcl/.env):
    ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Make the package importable when run as a plain script.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
load_dotenv(PACKAGE_ROOT / ".env")

import dataclasses  # noqa: E402

from anthropic.types import TextBlock, ToolUseBlock  # noqa: E402
from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import (  # noqa: E402
    ast_checker,
    find_description,
    simple_function_checker,
)
from bfcl_eval.model_handler.api_inference.claude import ClaudeHandler  # noqa: E402

from multilingual_bfcl.localization.locale_config import get_locale  # noqa: E402

# BFCL's checker looks the model up in MODEL_CONFIG_MAPPING (keyed by registry name)
# to learn whether '.' in function names was rewritten to '_' for the API. The bundled
# release predates claude-opus-4-8, so we clone the registered Claude FC entry (which
# correctly has underscore_to_dot=True for Anthropic) under the requested name.
_TEMPLATE_CLAUDE_FC_KEY = "claude-opus-4-5-20251101-FC"


def ensure_model_config(registry_name: str, api_model_name: str) -> None:
    """Register a ModelConfig for registry_name if BFCL doesn't already know it."""
    if registry_name in MODEL_CONFIG_MAPPING:
        return
    template = MODEL_CONFIG_MAPPING[_TEMPLATE_CLAUDE_FC_KEY]
    MODEL_CONFIG_MAPPING[registry_name] = dataclasses.replace(
        template, model_name=api_model_name, display_name=f"{registry_name} (cloned)"
    )


# USD per 1M tokens (input, output), keyed by model-id substring. Source: Anthropic
# pricing as of 2026-06. Update here if prices change.
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimated_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost estimate from recorded token counts. None if the model isn't priced.

    Note: input_tokens already sums the (growing) context resent on each retry,
    so this reflects the real billed input across the conversation. It does not
    account for prompt-cache discounts (caching is off for the 'multiple' category).
    """
    for key, (in_price, out_price) in _PRICING_PER_MTOK.items():
        if key in model:
            return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
    return None


def evaluate_call(
    func_descriptions: list,
    decoded: list,
    ground_truth: list,
    registry_name: str,
    category: str,
    allow_extra_calls: bool,
) -> dict:
    """Score a decoded model call list against ground truth.

    Runs BFCL's strict `ast_checker` first. When that fails *and* allow_extra_calls
    is set, falls back to a subset match: every expected ground-truth call must be
    satisfied by some distinct model call, reusing BFCL's own `simple_function_checker`
    for param/type/value scoring so only the count constraint is relaxed. This lets a
    model that emits the correct call plus extra calls count as a success.

    Returns: {valid, error_type, error, num_calls, num_expected, relaxed}
    where `relaxed` is True when the pass came only from the lenient fallback.
    """
    strict = ast_checker(
        func_descriptions, decoded, ground_truth,
        Language.PYTHON, category, registry_name,
    )
    base = {
        "num_calls": len(decoded),
        "num_expected": len(ground_truth),
        "relaxed": False,
    }
    if strict.get("valid") or not allow_extra_calls:
        return {
            "valid": bool(strict.get("valid")),
            "error_type": strict.get("error_type"),
            "error": strict.get("error"),
            **base,
        }

    # Lenient fallback: each expected call must match a distinct model call;
    # any extra model calls are ignored.
    if len(decoded) < len(ground_truth):
        return {"valid": False, "error_type": strict.get("error_type"),
                "error": strict.get("error"), **base}

    matched: set[int] = set()
    for expected in ground_truth:
        func_name_expected = list(expected.keys())[0]
        description = find_description(func_descriptions, func_name_expected)
        found = False
        for idx, call in enumerate(decoded):
            if idx in matched:
                continue
            res = simple_function_checker(
                description, call, expected, Language.PYTHON, registry_name
            )
            if res.get("valid"):
                matched.add(idx)
                found = True
                break
        if not found:
            return {"valid": False, "error_type": strict.get("error_type"),
                    "error": strict.get("error"), **base}

    return {"valid": True, "error_type": None, "error": None,
            **{**base, "relaxed": True}}


DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks"

# Generic, English, non-leaking rejection. It signals bad parameters without naming
# which parameter is wrong or what the expected value is, so it cannot leak the answer.
ERROR_TEMPLATE = (
    "400 Bad Request: invalid parameters for function '{name}'. "
    "Please review the function definition and try again."
)
NO_CALL_FEEDBACK = (
    "400 Bad Request: no function call was made. "
    "You must answer by calling one of the available functions."
)

# Injected at the start of the system prompt so the model knows to expect a
# non-English query while the tools stay in English. Mirrors MultilingualHandler.
LANG_PREFIX_TEMPLATE = (
    "The user will write in {language}. "
    "Understand the request in {language} and respond with the correct function call "
    "exactly as specified in the tool definitions (which remain in English)."
)


class PatchedClaudeHandler(ClaudeHandler):
    """ClaudeHandler whose _get_max_tokens knows about models the bundled BFCL
    release does not (e.g. claude-opus-4-8), instead of raising ValueError."""

    _MAX_TOKENS = {
        "claude-opus-4-8": 64000,
        "claude-opus-4-5": 64000,
        "claude-sonnet-4-6": 64000,
        "claude-sonnet-4-5": 64000,
        "claude-haiku-4-5": 64000,
    }

    # Models that reject the `temperature` parameter (it is deprecated for them).
    _NO_TEMPERATURE = ("claude-opus-4-8", "claude-sonnet-4-6")

    def _get_max_tokens(self) -> int:
        for key, value in self._MAX_TOKENS.items():
            if key in self.model_name:
                return value
        return 8192

    def generate_with_backoff(self, **kwargs):
        if any(m in self.model_name for m in self._NO_TEMPERATURE):
            kwargs.pop("temperature", None)
        return super().generate_with_backoff(**kwargs)


@dataclass
class Attempt:
    attempt: int
    text: list[str]
    tool_calls: list[dict[str, Any]]
    decoded: Any
    valid: bool
    error_type: str | None
    checker_error: Any
    feedback_sent: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    num_calls: int = 0
    # True when this attempt passed only because extra calls were tolerated.
    relaxed: bool = False


@dataclass
class EntryResult:
    id: str
    source_id: str | None
    locale: str | None
    localization_level: str | None
    benchmark: str
    model: str
    success: bool
    first_attempt_success: bool
    # Success that required tolerating extra (superfluous) function calls.
    success_with_extra_calls: bool
    num_attempts: int
    num_failed_calls: int
    final_error_type: str | None
    total_input_tokens: int
    total_output_tokens: int
    total_latency_s: float
    estimated_cost_usd: float | None = None
    error: str | None = None
    ground_truth: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    final_messages: Any = None


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _build_ground_truth_lookup(entries: list[dict]) -> dict[str, Any]:
    """Index possible-answer entries by their declared id. Different files in this
    project key answers differently (source id like 'multiple_2' vs locale-suffixed
    'multiple_2_he'), so we expose every key verbatim and let the caller try both."""
    return {e["id"]: e["ground_truth"] for e in entries}


def _resolve_ground_truth(entry: dict, lookup: dict[str, Any]) -> Any:
    for key in (entry.get("id"), entry.get("source_id")):
        if key is not None and key in lookup:
            return lookup[key]
    return None


def _json_default(obj: Any) -> Any:
    # Anthropic SDK content blocks are pydantic models.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


async def run_entry(
    handler: ClaudeHandler,
    entry: dict,
    ground_truth: Any,
    benchmark: str,
    category: str,
    registry_name: str,
    display_model: str,
    max_attempts: int,
    add_lang_prefix: bool,
    allow_extra_calls: bool,
    semaphore: asyncio.Semaphore,
) -> EntryResult:
    """Drive one entry's conversation loop: query -> check -> 400 -> retry."""
    result = EntryResult(
        id=entry["id"],
        source_id=entry.get("source_id"),
        locale=entry.get("locale"),
        localization_level=entry.get("localization_level"),
        benchmark=benchmark,
        model=display_model,
        success=False,
        first_attempt_success=False,
        success_with_extra_calls=False,
        num_attempts=0,
        num_failed_calls=0,
        final_error_type=None,
        total_input_tokens=0,
        total_output_tokens=0,
        total_latency_s=0.0,
        ground_truth=ground_truth,
    )

    async with semaphore:
        try:
            test_entry = copy.deepcopy(entry)

            # Initialise the BFCL inference state (message list, system prompt, tools).
            inference_data: dict = {}
            inference_data = handler._pre_query_processing_FC(inference_data, test_entry)
            inference_data = handler._compile_tools(inference_data, test_entry)

            if add_lang_prefix and entry.get("locale"):
                language = get_locale(entry["locale"]).name
                prefix = {"type": "text", "text": LANG_PREFIX_TEMPLATE.format(language=language)}
                existing = inference_data.get("system_prompt", [])
                inference_data["system_prompt"] = [prefix] + existing

            first_turn = copy.deepcopy(test_entry["question"][0])
            handler.add_first_turn_message_FC(inference_data, first_turn)

            for attempt_no in range(1, max_attempts + 1):
                api_response, latency = await asyncio.to_thread(
                    handler._query_FC, inference_data
                )
                parsed = handler._parse_query_response_FC(api_response)
                handler._add_assistant_message_FC(inference_data, parsed)

                texts = [b.text for b in api_response.content if isinstance(b, TextBlock)]
                tool_calls = [
                    {"name": b.name, "arguments": b.input, "id": b.id}
                    for b in api_response.content
                    if isinstance(b, ToolUseBlock)
                ]

                result.num_attempts = attempt_no
                result.total_input_tokens += parsed["input_token"]
                result.total_output_tokens += parsed["output_token"]
                result.total_latency_s += latency

                # Case 1: the model produced no function call at all.
                if not parsed["tool_call_ids"]:
                    result.num_failed_calls += 1
                    feedback = NO_CALL_FEEDBACK
                    result.final_error_type = "no_function_call"
                    result.attempts.append(Attempt(
                        attempt=attempt_no, text=texts, tool_calls=tool_calls,
                        decoded=None, valid=False, error_type="no_function_call",
                        checker_error=None,
                        feedback_sent=feedback if attempt_no < max_attempts else None,
                        input_tokens=parsed["input_token"],
                        output_tokens=parsed["output_token"], latency_s=latency,
                    ))
                    if attempt_no < max_attempts:
                        inference_data["message"].append(
                            {"role": "user", "content": [{"type": "text", "text": feedback}]}
                        )
                    continue

                # Case 2: decode the call(s) and check against ground truth.
                # When allow_extra_calls is set, a correct call accompanied by extra
                # (superfluous) calls still counts as a pass.
                relaxed = False
                num_calls = len(tool_calls)
                try:
                    decoded = handler.decode_ast(
                        parsed["model_responses"], ReturnFormat.PYTHON, False
                    )
                    checker = evaluate_call(
                        test_entry["function"], decoded, ground_truth,
                        registry_name, category, allow_extra_calls,
                    )
                    valid = bool(checker.get("valid"))
                    error_type = checker.get("error_type")
                    checker_error = checker.get("error")
                    relaxed = bool(checker.get("relaxed"))
                except Exception as exc:  # malformed call the decoder can't parse
                    decoded = None
                    valid = False
                    error_type = "decode_failed"
                    checker_error = [str(exc)]

                if valid:
                    result.success = True
                    result.first_attempt_success = attempt_no == 1
                    result.success_with_extra_calls = relaxed
                    result.final_error_type = None
                    result.attempts.append(Attempt(
                        attempt=attempt_no, text=texts, tool_calls=tool_calls,
                        decoded=decoded, valid=True, error_type=None,
                        checker_error=None, feedback_sent=None,
                        input_tokens=parsed["input_token"],
                        output_tokens=parsed["output_token"], latency_s=latency,
                        num_calls=num_calls, relaxed=relaxed,
                    ))
                    break

                # Wrong call: tally, record, and inject a generic 400 per tool call.
                result.num_failed_calls += 1
                result.final_error_type = error_type
                execution_results = [
                    ERROR_TEMPLATE.format(name=tc["name"]) for tc in tool_calls
                ]
                feedback = " | ".join(execution_results)
                result.attempts.append(Attempt(
                    attempt=attempt_no, text=texts, tool_calls=tool_calls,
                    decoded=decoded, valid=False, error_type=error_type,
                    checker_error=checker_error,
                    feedback_sent=feedback if attempt_no < max_attempts else None,
                    input_tokens=parsed["input_token"],
                    output_tokens=parsed["output_token"], latency_s=latency,
                    num_calls=num_calls,
                ))
                if attempt_no < max_attempts:
                    handler._add_execution_results_FC(
                        inference_data, execution_results, parsed
                    )

            result.final_messages = inference_data.get("message")
        except Exception as exc:  # keep one bad entry from killing the whole run
            result.error = f"{type(exc).__name__}: {exc}"

    result.estimated_cost_usd = estimated_cost_usd(
        display_model, result.total_input_tokens, result.total_output_tokens
    )
    return result


def _write_transcript(out_dir: Path, result: EntryResult) -> None:
    transcripts = out_dir / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    with (transcripts / f"{result.id}.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)


STATS_COLUMNS = [
    "id", "source_id", "locale", "localization_level", "benchmark", "model",
    "success", "first_attempt_success", "success_with_extra_calls",
    "num_attempts", "num_failed_calls", "final_error_type",
    "total_input_tokens", "total_output_tokens", "estimated_cost_usd",
    "total_latency_s", "error",
]


def _write_statistics(out_dir: Path, results: list[EntryResult]) -> None:
    with (out_dir / "statistics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        results.sort(key=lambda result: int(result.id.split("_")[1]))
        for r in results:
            writer.writerow({k: getattr(r, k) for k in STATS_COLUMNS})


def _write_summary(out_dir: Path, results: list[EntryResult], meta: dict) -> dict:
    total = len(results)
    errored = [r for r in results if r.error]
    scored = [r for r in results if not r.error]
    succeeded = [r for r in scored if r.success]
    summary = {
        **meta,
        "total_entries": total,
        "errored_entries": len(errored),
        "scored_entries": len(scored),
        "success_count": len(succeeded),
        "success_rate": (len(succeeded) / len(scored)) if scored else None,
        "first_attempt_success_count": sum(r.first_attempt_success for r in scored),
        "first_attempt_success_rate": (
            sum(r.first_attempt_success for r in scored) / len(scored) if scored else None
        ),
        "success_with_extra_calls_count": sum(r.success_with_extra_calls for r in scored),
        "total_failed_calls": sum(r.num_failed_calls for r in scored),
        "avg_attempts": (
            sum(r.num_attempts for r in scored) / len(scored) if scored else None
        ),
        "total_input_tokens": sum(r.total_input_tokens for r in results),
        "total_output_tokens": sum(r.total_output_tokens for r in results),
        "total_estimated_cost_usd": round(
            sum(r.estimated_cost_usd or 0.0 for r in results), 4
        ),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


_ENTRY_RESULT_FIELDS = {f.name for f in dataclasses.fields(EntryResult)}


def recompute_stats(benchmark: str, args: argparse.Namespace) -> None:
    """Regenerate statistics.csv + summary.json from existing transcripts, without
    calling the API. Re-derives estimated_cost_usd from the recorded token counts
    so a run done before the cost column existed (or under stale pricing) is updated."""
    out_dir = (
        PACKAGE_ROOT / args.output_dir / args.model / args.category
        / args.lang_dir / benchmark
    )
    transcripts = out_dir / "transcripts"
    if not transcripts.is_dir():
        raise FileNotFoundError(f"No transcripts to recompute under: {transcripts}")

    results: list[EntryResult] = []
    for path in sorted(transcripts.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        result = EntryResult(**{k: v for k, v in data.items() if k in _ENTRY_RESULT_FIELDS})
        result.estimated_cost_usd = estimated_cost_usd(
            args.model, result.total_input_tokens, result.total_output_tokens
        )
        results.append(result)

    _write_statistics(out_dir, results)
    summary = _write_summary(out_dir, results, {
        "benchmark": benchmark,
        "category": args.category,
        "lang_dir": args.lang_dir,
        "model": args.model,
        "max_attempts": args.max_attempts,
        "lang_prefix": not args.no_lang_prefix,
        "allow_extra_calls": not args.strict_call_count,
        "recomputed_from_transcripts": True,
    })
    print(f"=== {benchmark} | recomputed {len(results)} transcripts (no API calls) ===")
    print(f"  -> total estimated cost ${summary['total_estimated_cost_usd']}")
    print(f"  -> wrote {out_dir}")


async def run_benchmark(
    handler: ClaudeHandler,
    benchmark: str,
    args: argparse.Namespace,
) -> None:
    bench_path = DATA_ROOT / args.category / args.lang_dir / f"{benchmark}.json"
    gt_path = DATA_ROOT / args.category / "possible_answer" / args.lang_dir / f"{benchmark}.json"
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {bench_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    entries = _load_jsonl(bench_path)
    gt_lookup = _build_ground_truth_lookup(_load_jsonl(gt_path))
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    runnable: list[tuple[dict, Any]] = []
    skipped = 0
    for entry in entries:
        gt = _resolve_ground_truth(entry, gt_lookup)
        if gt is None:
            skipped += 1
            print(f"  ! no ground truth for {entry['id']}; skipping", file=sys.stderr)
            continue
        runnable.append((entry, gt))

    out_dir = (
        PACKAGE_ROOT / args.output_dir / args.model / args.category
        / args.lang_dir / benchmark
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {benchmark} | model={args.model} | {len(runnable)} entries "
          f"(skipped {skipped}) | max_attempts={args.max_attempts} ===")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(run_entry(
            handler, entry, gt, benchmark, args.category,
            f"{args.model}-FC", args.model,
            args.max_attempts, not args.no_lang_prefix,
            not args.strict_call_count, semaphore,
        ))
        for entry, gt in runnable
    ]

    results: list[EntryResult] = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        _write_transcript(out_dir, result)
        results.append(result)
        done += 1
        flag = "OK " if result.success else ("ERR" if result.error else "FAIL")
        print(f"  [{done}/{len(tasks)}] {flag} {result.id} "
              f"(attempts={result.num_attempts}, failed_calls={result.num_failed_calls})")

    # Keep statistics ordered by entry id for stable diffs.
    results.sort(key=lambda r: r.id)
    _write_statistics(out_dir, results)
    summary = _write_summary(out_dir, results, {
        "benchmark": benchmark,
        "category": args.category,
        "lang_dir": args.lang_dir,
        "model": args.model,
        "max_attempts": args.max_attempts,
        "lang_prefix": not args.no_lang_prefix,
        "allow_extra_calls": not args.strict_call_count,
    })

    print(f"  -> success {summary['success_count']}/{summary['scored_entries']} "
          f"({(summary['success_rate'] or 0):.1%}), "
          f"first-attempt {summary['first_attempt_success_count']}, "
          f"with-extra-calls {summary['success_with_extra_calls_count']}, "
          f"failed calls {summary['total_failed_calls']}, "
          f"cost ${summary['total_estimated_cost_usd']}")
    print(f"  -> wrote {out_dir}")


async def main_async(args: argparse.Namespace) -> None:
    if args.recompute_stats:
        # Pure local recompute from saved transcripts — no model, no API key needed.
        for benchmark in args.benchmark:
            recompute_stats(benchmark, args)
        return

    registry_name = f"{args.model}-FC"
    ensure_model_config(registry_name, args.model)
    handler = PatchedClaudeHandler(
        model_name=registry_name,
        temperature=args.temperature,
        registry_name=registry_name,
        is_fc_model=True,
    )
    for benchmark in args.benchmark:
        await run_benchmark(handler, benchmark, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive tool-calling robustness harness for multilingual-bfcl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", nargs="+", required=True,
                        help="Benchmark file stem(s) under data/benchmarks/<category>/<lang-dir>/, "
                             "e.g. he_translatable_query.")
    parser.add_argument("--category", default="multiple",
                        help="Benchmark category folder under data/benchmarks/.")
    parser.add_argument("--lang-dir", default="heb",
                        help="Language subfolder under the category (directory name, e.g. heb).")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="Model id (the handler appends/normalises the -FC suffix).")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max function-call attempts per entry before it is marked failed.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Only run the first N entries per benchmark (0 or less = all).")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max entries evaluated in parallel.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--no-lang-prefix", action="store_true",
                        help="Do not inject the 'user writes in <language>' system prompt.")
    parser.add_argument("--strict-call-count", action="store_true",
                        help="Require the exact expected number of calls (standard BFCL). "
                             "By default, a correct call plus extra calls still counts as a pass.")
    parser.add_argument("--output-dir", default="results_robustness",
                        help="Output root (relative to the package root).")
    parser.add_argument("--recompute-stats", action="store_true",
                        help="Do not call the API: rebuild statistics.csv and summary.json "
                             "(including the cost column) from existing transcripts for the "
                             "given --benchmark/--model/--category/--lang-dir.")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
