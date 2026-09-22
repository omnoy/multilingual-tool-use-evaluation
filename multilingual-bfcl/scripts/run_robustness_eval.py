"""
run_robustness_eval.py — Interactive tool-calling robustness harness for multilingual-bfcl.

Tests whether an LLM can recover from API-side rejections when the user query is in
one language but the API (tool definitions + error messages) is in English. For each
benchmark entry we run a short *conversation*, not a single shot:

  1. Send the (translated) query + the English tool definitions to the model.
  2. Decode the model's function call and check it against the ground-truth answer
     using BFCL's own AST checker (so scoring matches standard BFCL exactly).
  3. If the call is wrong, return a generic English "400 Bad Request" tool result
     and let the model try again — up to --max-attempts *tool-call* attempts.
  4. The entry succeeds the moment any attempt produces a correct call; otherwise it
     fails after the tool-call budget (or the total API-call ceiling) is exhausted.

Attempt budgeting (two independent limits):
  --max-attempts   counts only turns in which the model actually emitted a function
                   call and we checked it. A turn where the model instead asks for
                   clarification (no function call) does NOT consume this budget — it
                   is re-prompted so the retries are spent on real tool-call attempts,
                   not on clarification chatter.
  --max-api-calls  a hard ceiling on the TOTAL number of model turns (API calls) for an
                   entry, so a model that only ever asks for clarification still stops.

Scoring is BFCL's own AST checker, with one relaxation enabled by default: an attempt
counts as a pass if ANY one of the model's calls matches a ground-truth call, whatever
the number of calls (so a correct call alongside extra or wrong calls still passes, and
the strict checker's "wrong_count" never fails an attempt). Pass --strict-call-count for
standard BFCL behaviour (exact call count required). The statistics record
success_with_extra_calls so the two can be compared.

Why plain async and not the Batch API: each retry depends on the error returned for the
previous attempt, so the turns within an entry are inherently sequential and cannot be
pre-packed into a batch. Entries, however, are independent, so we run them concurrently
with a semaphore cap and rely on the handler's built-in rate-limit backoff.

Data layout (under data/benchmarks/<category>/<lang-dir>/):
  <benchmark>.jsonl   one JSON object per line, each carrying its own ground truth.
                      Keys: id, messages (the BFCL "question" turns), function (the
                      English tool defs), ground_truth (the expected calls), and — for
                      localized files — source_id, locale, localization_level.
  Categories seen in this project: bfcl_multiple, wildtoolbench.
  Lang dirs: en (source, no locale), he, zh, ...

Outputs (under <output-dir>/<model>/<category>/<lang-dir>/<benchmark>/):
  transcripts/<entry_id>.json   full reasoning + tool calls + injected errors per entry
  statistics.csv                one row per entry (success, #attempts, #api calls,
                                #failed calls, tokens, estimated_cost_usd, ...)
  summary.json                  aggregate counts + total_estimated_cost_usd for the run

The cost column is derived from the recorded token counts using per-1M-token prices
from a CSV (model_prices.csv by default; override with --prices, or override the price
for all models with --price-input/--price-output). A model absent from the CSV is
unpriced (cost 0) and a warning is printed. Pass --recompute-stats to rebuild
statistics.csv + summary.json (including cost) from existing transcripts WITHOUT calling
the API — useful after editing prices or to add the cost column to older runs.

A "model" here is an Azure deployment name. --model-type selects the route + how request
parameters are shaped: 'standard'/'reasoning' use the OpenAI-compatible v1 route, and
'claude' uses the Foundry /anthropic route (AnthropicFoundry client). See azure_client.ModelType.

Usage:
    # Small sample (default 10 entries) against an Azure deployment:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query \
        --category bfcl_multiple --lang-dir he --model gpt-4o

    # Full file, more retries, higher concurrency:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query \
        --model gpt-4o --limit 0 --max-attempts 8 --concurrency 8

    # A reasoning deployment (no temperature; uses max_completion_tokens):
    python scripts/run_robustness_eval.py --benchmark he_translatable_query \
        --model o3 --model-type reasoning

    # A Claude deployment via the Foundry /anthropic route:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query \
        --category bfcl_multiple --lang-dir he --model claude-opus-5 --model-type claude

    # Finish a run that stopped early (only runs entries missing a transcript), then
    # rebuild stats over the whole benchmark:
    python scripts/run_robustness_eval.py --benchmark he_translatable_full \
        --category bfcl_multiple --lang-dir he --model gpt-4o --limit 0 --resume

    # Rebuild stats + cost from existing transcripts, no API calls:
    python scripts/run_robustness_eval.py --benchmark he_translatable_query --recompute-stats

Environment (multilingual-bfcl/.env):
    AZURE_OPENAI_ENDPOINT=...      # …/openai/v1 (or the resource root)
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_DEPLOYMENT=...    # optional default for --model
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import re
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

from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import (  # noqa: E402
    ast_checker,
    find_description,
    simple_function_checker,
)

from multilingual_bfcl.azure_client import ModelType  # noqa: E402
from multilingual_bfcl.model_handler.anthropic_foundry_handler import (  # noqa: E402
    AnthropicFoundryFCHandler,
)
from multilingual_bfcl.model_handler.azure_openai_handler import (  # noqa: E402
    AzureOpenAIFCHandler,
)
from multilingual_bfcl.localization.locale_config import get_locale  # noqa: E402

# Any handler the harness drives (both expose the same BFCL FC methods plus the
# extract_turn / apply_language_prefix format adapters run_entry relies on).
FCHandler = AzureOpenAIFCHandler | AnthropicFoundryFCHandler

# BFCL's checker looks the model up in MODEL_CONFIG_MAPPING (keyed by registry name)
# to learn whether '.' in function names was rewritten to '_' for the API. Azure
# deployments are not registered, so we clone a bundled FC entry (both have
# underscore_to_dot=True) under the requested name. Only the checker-relevant fields
# matter; the handler is built separately. Use the OpenAI template for v1 deployments
# and the Claude template for the /anthropic route.
_TEMPLATE_OPENAI_FC_KEY = "gpt-4.1-2025-04-14-FC"
_TEMPLATE_CLAUDE_FC_KEY = "claude-opus-4-5-20251101-FC"


def ensure_model_config(
    registry_name: str, api_model_name: str, template_key: str = _TEMPLATE_OPENAI_FC_KEY
) -> None:
    """Register a ModelConfig for registry_name if BFCL doesn't already know it."""
    if registry_name in MODEL_CONFIG_MAPPING:
        return
    template = MODEL_CONFIG_MAPPING[template_key]
    MODEL_CONFIG_MAPPING[registry_name] = dataclasses.replace(
        template, model_name=api_model_name, display_name=f"{registry_name} (cloned)"
    )


# USD per 1M tokens (input, output), keyed by deployment name. Loaded from a prices
# CSV (model_prices.csv by default) at startup; see load_price_table / --prices.
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {}

# Populated from --price-input / --price-output when both are given; applies to any
# deployment, taking precedence over the table above.
_PRICE_OVERRIDE: tuple[float, float] | None = None

# Default prices file (per-1M-token USD), alongside the package root.
DEFAULT_PRICES_CSV = PACKAGE_ROOT / "model_prices.csv"


def load_price_table(path: Path) -> dict[str, tuple[float, float]]:
    """Parse a prices CSV into {model: (input_per_mtok, output_per_mtok)}.

    Rows are `model,input,output`. Blank lines, a `model,...` header, and lines
    starting with '#' are ignored. Missing file -> empty table (cost stays unpriced).
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


def _lookup_price(model: str) -> tuple[float, float] | None:
    """Exact match on the deployment name, else the longest listed substring of it."""
    if model in _PRICING_PER_MTOK:
        return _PRICING_PER_MTOK[model]
    matches = [(k, v) for k, v in _PRICING_PER_MTOK.items() if k in model]
    if not matches:
        return None
    return max(matches, key=lambda kv: len(kv[0]))[1]


def estimated_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost estimate from recorded token counts. None if the model isn't priced.

    Note: input_tokens already sums the (growing) context resent on each retry,
    so this reflects the real billed input across the conversation.
    """
    prices = _PRICE_OVERRIDE if _PRICE_OVERRIDE is not None else _lookup_price(model)
    if prices is not None:
        in_price, out_price = prices
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
    is set, falls back to an "any-match" check: the attempt is valid if ANY one of the
    model's calls matches ANY expected ground-truth call (via BFCL's own
    `simple_function_checker` for name/param/type/value scoring). The number of calls
    is not constrained, so a model that emits the correct call — alone or alongside
    extra or wrong calls — passes, and the strict checker's `wrong_count` is never
    surfaced.

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

    # Lenient fallback (default): valid if ANY model call matches ANY expected
    # ground-truth call. Call count is ignored (so `wrong_count` never fails an attempt).
    for expected in ground_truth:
        func_name_expected = list(expected.keys())[0]
        description = find_description(func_descriptions, func_name_expected)
        for call in decoded:
            res = simple_function_checker(
                description, call, expected, Language.PYTHON, registry_name
            )
            if res.get("valid"):
                return {"valid": True, "error_type": None, "error": None,
                        **{**base, "relaxed": True}}

    return {"valid": False, "error_type": "no_matching_call",
            "error": ["No model call matched a ground-truth call."], **base}


DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks"

# Generic, English, non-leaking rejection. It signals bad parameters without naming
# which parameter is wrong or what the expected value is, so it cannot leak the answer.
MULTIPLE_TOOL_CALL_ERROR_TEMPLATE = (
    "400 Bad Request: w. "
    "Please review the function definition and try again."
)
PARAMETER_VALUE_ERROR_TEMPLATE = (
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
    "exactly as specified in the tool definitions."
)


@dataclass
class Attempt:
    # 1-based index of this model turn / API call within the entry's conversation.
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
    # True when the model emitted at least one function call this turn (i.e. this turn
    # consumed one of the --max-attempts tool-call retries). False for clarification
    # turns, which are re-prompted for free and only count against --max-api-calls.
    is_tool_call: bool = False
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
    # Number of tool-call attempts made (turns that emitted a function call). This is the
    # budget bounded by --max-attempts; clarification turns do not increment it.
    num_attempts: int
    # Total model turns / API calls for this entry (bounded by --max-api-calls). Equals
    # num_attempts plus the number of clarification (no-function-call) turns.
    num_api_calls: int
    # Turns where the model asked for clarification instead of calling a function.
    num_clarifications: int
    num_failed_calls: int
    final_error_type: str | None
    total_input_tokens: int
    total_output_tokens: int
    total_latency_s: float
    estimated_cost_usd: float | None = None
    error: str | None = None
    query: Any = None
    ground_truth: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    final_messages: Any = None
    # The system prompt the model actually received (the language prefix, plus any
    # entry-provided system content). None when no system prompt was set. Captured
    # provider-agnostically: OpenAI keeps it as a system-role message, Claude in a
    # separate `system` field, so it is not always visible in final_messages.
    system_prompt: Any = None


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# Locale codes as they appear in the data files may be shorter than the registry keys
# (e.g. the data uses "zh" while the registry lists "zh-CN"). Map the common short forms
# so the language-prefix lookup still resolves.
_LOCALE_ALIASES = {"zh": "zh-CN"}


def _locale_name(code: str) -> str:
    """Human-readable English language name for a locale code, defensively.

    Falls back through a small alias map (zh -> zh-CN) and finally to the raw code so an
    unregistered locale never crashes a run — the prefix just names the code verbatim.
    """
    for candidate in (code, _LOCALE_ALIASES.get(code)):
        if candidate is None:
            continue
        try:
            return get_locale(candidate).name
        except ValueError:
            continue
    return code


def _extract_system_prompt(inference_data: dict) -> Any:
    """The effective system prompt the model received, whatever the provider stored it as.

    Claude keeps it in a separate `system_prompt` field; OpenAI keeps it as a
    system-role message in the message list. Returns None when neither is present.
    """
    system_prompt = inference_data.get("system_prompt")
    if system_prompt:
        return system_prompt
    for msg in inference_data.get("message", []) or []:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return msg.get("content")
    return None


def _json_default(obj: Any) -> Any:
    # OpenAI / Anthropic SDK message/response objects are pydantic models.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


async def run_entry(
    handler: FCHandler,
    entry: dict,
    ground_truth: Any,
    benchmark: str,
    category: str,
    registry_name: str,
    display_model: str,
    max_tool_call_attempts: int,
    max_api_calls: int,
    add_lang_prefix: bool,
    allow_extra_calls: bool,
    semaphore: asyncio.Semaphore,
) -> EntryResult:
    """Drive one entry's conversation loop: query -> check -> 400 -> retry.

    Two limits bound the loop. --max-attempts caps *tool-call* attempts (turns that
    emit a function call); a clarification turn (no function call) is re-prompted but
    does not consume that budget. --max-api-calls caps the total number of turns so a
    model that only ever clarifies still terminates.
    """
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
        num_api_calls=0,
        num_clarifications=0,
        num_failed_calls=0,
        final_error_type=None,
        total_input_tokens=0,
        total_output_tokens=0,
        total_latency_s=0.0,
        query=entry["messages"],
        ground_truth=ground_truth,
    )

    async with semaphore:
        try:
            test_entry = copy.deepcopy(entry)
            # BFCL's handler methods expect the turns under the "question" key; this
            # project's files now store them under "messages", so alias them.
            test_entry["question"] = test_entry["messages"]

            # Initialise the BFCL inference state (message list, system prompt, tools).
            inference_data: dict = {}
            inference_data = handler._pre_query_processing_FC(inference_data, test_entry)
            inference_data = handler._compile_tools(inference_data, test_entry)

            # Inject the language-awareness prefix. Where it lands is provider-specific
            # (OpenAI: a system-role message; Claude: the separate `system` field), so
            # each handler implements apply_language_prefix. This runs before the first
            # user turn is added just below.
            if add_lang_prefix and entry.get("locale"):
                language = _locale_name(entry["locale"])
                handler.apply_language_prefix(
                    inference_data, LANG_PREFIX_TEMPLATE.format(language=language)
                )

            first_turn = copy.deepcopy(test_entry["question"][0])
            handler.add_first_turn_message_FC(inference_data, first_turn)

            api_calls = 0
            tool_call_attempts = 0
            # Loop until the model succeeds, exhausts its tool-call retries, or hits the
            # hard total-API-call ceiling.
            while tool_call_attempts < max_tool_call_attempts and api_calls < max_api_calls:
                api_response, latency = await asyncio.to_thread(
                    handler._query_FC, inference_data
                )
                api_calls += 1
                parsed = handler._parse_query_response_FC(api_response)
                handler._add_assistant_message_FC(inference_data, parsed)

                texts, tool_calls = handler.extract_turn(api_response)

                result.num_api_calls = api_calls
                result.total_input_tokens += parsed["input_token"]
                result.total_output_tokens += parsed["output_token"]
                result.total_latency_s += latency

                # Case 1: the model produced no function call at all (a clarification).
                # This does NOT consume the tool-call budget — we re-prompt it to call a
                # function, bounded only by the total API-call ceiling.
                if not parsed["tool_call_ids"]:
                    result.num_clarifications += 1
                    result.final_error_type = "no_function_call"
                    # We will re-prompt as long as tool-call retries remain AND there is
                    # room under the total API-call ceiling for another turn.
                    reprompt = tool_call_attempts < max_tool_call_attempts and api_calls < max_api_calls
                    feedback = NO_CALL_FEEDBACK
                    result.attempts.append(Attempt(
                        attempt=api_calls, text=texts, tool_calls=tool_calls,
                        decoded=None, valid=False, error_type="no_function_call",
                        checker_error=None,
                        feedback_sent=feedback if reprompt else None,
                        input_tokens=parsed["input_token"],
                        output_tokens=parsed["output_token"], latency_s=latency,
                        is_tool_call=False,
                    ))
                    if reprompt:
                        inference_data["message"].append(
                            {"role": "user", "content": feedback}
                        )
                    continue

                # Case 2: a function call was made — this consumes one tool-call attempt.
                # Decode the call(s) and check against ground truth. When allow_extra_calls
                # is set, a correct call accompanied by extra (superfluous) calls still
                # counts as a pass.
                tool_call_attempts += 1
                result.num_attempts = tool_call_attempts
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
                    result.first_attempt_success = tool_call_attempts == 1
                    result.success_with_extra_calls = relaxed
                    result.final_error_type = None
                    result.attempts.append(Attempt(
                        attempt=api_calls, text=texts, tool_calls=tool_calls,
                        decoded=decoded, valid=True, error_type=None,
                        checker_error=None, feedback_sent=None,
                        input_tokens=parsed["input_token"],
                        output_tokens=parsed["output_token"], latency_s=latency,
                        num_calls=num_calls, is_tool_call=True, relaxed=relaxed,
                    ))
                    break

                # Wrong call: tally, record, and inject a generic 400 per tool call.
                result.num_failed_calls += 1
                result.final_error_type = error_type
                execution_results = [
                    PARAMETER_VALUE_ERROR_TEMPLATE.format(name=tc["name"]) for tc in tool_calls
                ]
                feedback = " | ".join(execution_results)
                retry = tool_call_attempts < max_tool_call_attempts and api_calls < max_api_calls
                result.attempts.append(Attempt(
                    attempt=api_calls, text=texts, tool_calls=tool_calls,
                    decoded=decoded, valid=False, error_type=error_type,
                    checker_error=checker_error,
                    feedback_sent=feedback if retry else None,
                    input_tokens=parsed["input_token"],
                    output_tokens=parsed["output_token"], latency_s=latency,
                    num_calls=num_calls, is_tool_call=True,
                ))
                if retry:
                    handler._add_execution_results_FC(
                        inference_data, execution_results, parsed
                    )

            result.final_messages = inference_data.get("message")
            result.system_prompt = _extract_system_prompt(inference_data)
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
    "num_attempts", "num_api_calls", "num_clarifications",
    "num_failed_calls", "final_error_type",
    "total_input_tokens", "total_output_tokens", "estimated_cost_usd",
    "total_latency_s", "error",
]


def _entry_sort_key(result: EntryResult) -> tuple:
    """Stable ordering by the numeric index embedded in the id, robust across id shapes
    like 'multiple_2_he' and 'wild_tool_bench_0_zh'. Uses the last integer token found,
    falling back to the raw id when there is none."""
    numbers = re.findall(r"\d+", result.id)
    return (int(numbers[-1]) if numbers else float("inf"), result.id)


def _write_statistics(out_dir: Path, results: list[EntryResult]) -> None:
    with (out_dir / "statistics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        results.sort(key=_entry_sort_key)
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
        "total_clarifications": sum(r.num_clarifications for r in scored),
        "avg_attempts": (
            sum(r.num_attempts for r in scored) / len(scored) if scored else None
        ),
        "avg_api_calls": (
            sum(r.num_api_calls for r in scored) / len(scored) if scored else None
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


def _out_dir_for(benchmark: str, args: argparse.Namespace) -> Path:
    return (
        PACKAGE_ROOT / args.output_dir / args.model / args.category
        / args.lang_dir / benchmark
    )


def _existing_transcript_ids(out_dir: Path) -> set[str]:
    """Ids of entries that already have a transcript written under out_dir."""
    transcripts = out_dir / "transcripts"
    if not transcripts.is_dir():
        return set()
    return {p.stem for p in transcripts.glob("*.json")}


def _load_results_from_transcripts(out_dir: Path, model: str) -> list[EntryResult]:
    """Reconstruct EntryResult objects from every transcript under out_dir, re-deriving
    estimated_cost_usd from the recorded token counts (so cost tracks current pricing).
    The transcripts are the source of truth, so this is what lets a resumed run report
    stats over the whole benchmark, not just the entries run this time."""
    transcripts = out_dir / "transcripts"
    if not transcripts.is_dir():
        raise FileNotFoundError(f"No transcripts under: {transcripts}")
    results: list[EntryResult] = []
    for path in sorted(transcripts.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        result = EntryResult(**{k: v for k, v in data.items() if k in _ENTRY_RESULT_FIELDS})
        result.estimated_cost_usd = estimated_cost_usd(
            model, result.total_input_tokens, result.total_output_tokens
        )
        results.append(result)
    return results


def _summary_meta(benchmark: str, args: argparse.Namespace, **extra: Any) -> dict:
    return {
        "benchmark": benchmark,
        "category": args.category,
        "lang_dir": args.lang_dir,
        "model": args.model,
        "max_tool_call_attempts": args.max_tool_call_attempts,
        "max_api_calls": args.max_api_calls,
        "lang_prefix": not args.no_lang_prefix,
        "allow_extra_calls": not args.strict_call_count,
        **extra,
    }


def recompute_stats(benchmark: str, args: argparse.Namespace) -> None:
    """Regenerate statistics.csv + summary.json from existing transcripts, without
    calling the API. Re-derives estimated_cost_usd from the recorded token counts
    so a run done before the cost column existed (or under stale pricing) is updated."""
    out_dir = _out_dir_for(benchmark, args)
    results = _load_results_from_transcripts(out_dir, args.model)

    _write_statistics(out_dir, results)
    summary = _write_summary(out_dir, results, _summary_meta(
        benchmark, args, recomputed_from_transcripts=True,
    ))
    print(f"=== {benchmark} | recomputed {len(results)} transcripts (no API calls) ===")
    print(f"  -> total estimated cost ${summary['total_estimated_cost_usd']}")
    print(f"  -> wrote {out_dir}")


async def run_benchmark(
    handler: FCHandler,
    benchmark: str,
    args: argparse.Namespace,
) -> None:
    bench_path = DATA_ROOT / args.category / args.lang_dir / f"{benchmark}.jsonl"
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {bench_path}")

    entries = _load_jsonl(bench_path)
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    out_dir = _out_dir_for(benchmark, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --resume: skip entries that already have a transcript so a run that stopped early
    # (e.g. it ran out of credits) can be finished without re-running completed entries.
    already_done = _existing_transcript_ids(out_dir) if args.resume else set()

    # Ground truth now travels with each entry (the separate possible_answer/ files are
    # gone), so we read it straight off the record.
    runnable: list[tuple[dict, Any]] = []
    skipped = 0
    resumed = 0
    for entry in entries:
        if entry["id"] in already_done:
            resumed += 1
            continue
        gt = entry.get("ground_truth")
        if gt is None:
            skipped += 1
            print(f"  ! no ground truth for {entry['id']}; skipping", file=sys.stderr)
            continue
        runnable.append((entry, gt))

    resume_note = f" | resuming, {resumed} already done" if args.resume else ""
    print(f"\n=== {benchmark} | model={args.model} | {len(runnable)} entries to run "
          f"(skipped {skipped}){resume_note} "
          f"| max_tool_call_attempts={args.max_tool_call_attempts} "
          f"| max_api_calls={args.max_api_calls} ===")

    if not runnable and args.resume:
        # Nothing left to run — just (re)build the combined stats from what's on disk.
        results = _load_results_from_transcripts(out_dir, args.model)
        _write_statistics(out_dir, results)
        summary = _write_summary(out_dir, results, _summary_meta(benchmark, args))
        print(f"  -> nothing to run; rebuilt stats over {len(results)} transcripts, "
              f"success {summary['success_count']}/{summary['scored_entries']}, "
              f"cost ${summary['total_estimated_cost_usd']}")
        print(f"  -> wrote {out_dir}")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(run_entry(
            handler, entry, gt, benchmark, args.category,
            f"{args.model}-FC", args.model,
            args.max_tool_call_attempts, args.max_api_calls, not args.no_lang_prefix,
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
              f"(attempts={result.num_attempts}, api_calls={result.num_api_calls}, "
              f"clarifications={result.num_clarifications}, "
              f"failed_calls={result.num_failed_calls})")

    # On a --resume run, this invocation only produced the missing entries. Rebuild the
    # stats over EVERY transcript on disk so statistics.csv / summary.json describe the
    # whole benchmark (the entries done earlier + the ones just filled in).
    if args.resume:
        results = _load_results_from_transcripts(out_dir, args.model)
    # Keep statistics ordered by entry index for stable diffs.
    results.sort(key=_entry_sort_key)
    _write_statistics(out_dir, results)
    summary = _write_summary(out_dir, results, _summary_meta(benchmark, args))

    print(f"  -> success {summary['success_count']}/{summary['scored_entries']} "
          f"({(summary['success_rate'] or 0):.1%}), "
          f"first-attempt {summary['first_attempt_success_count']}, "
          f"with-extra-calls {summary['success_with_extra_calls_count']}, "
          f"failed calls {summary['total_failed_calls']}, "
          f"clarifications {summary['total_clarifications']}, "
          f"cost ${summary['total_estimated_cost_usd']}")
    print(f"  -> wrote {out_dir}")


async def main_async(args: argparse.Namespace) -> None:
    # Load the prices CSV for the cost column, then apply the ad-hoc override (if both
    # flags are given, it wins over the table for every model).
    global _PRICING_PER_MTOK, _PRICE_OVERRIDE
    _PRICING_PER_MTOK = load_price_table(Path(args.prices))
    if args.price_input is not None and args.price_output is not None:
        _PRICE_OVERRIDE = (args.price_input, args.price_output)
    elif _PRICE_OVERRIDE is None and _lookup_price(args.model) is None:
        print(
            f"  ! no price for '{args.model}' in {args.prices} and no --price-input/"
            "--price-output given; the cost column will be 0. Add a row to the CSV or "
            "pass the price flags.",
            file=sys.stderr,
        )

    if args.recompute_stats:
        # Pure local recompute from saved transcripts — no model, no API key needed.
        for benchmark in args.benchmark:
            recompute_stats(benchmark, args)
        return

    model_type = ModelType(args.model_type)
    registry_name = f"{args.model}-FC"
    handler: FCHandler
    if model_type == ModelType.CLAUDE:
        # Foundry /anthropic route (native Anthropic message format).
        ensure_model_config(registry_name, args.model, _TEMPLATE_CLAUDE_FC_KEY)
        handler = AnthropicFoundryFCHandler(
            model_name=registry_name,
            temperature=args.temperature,
            registry_name=registry_name,
            is_fc_model=True,
            max_tokens=args.max_tokens,
        )
    else:
        # Foundry OpenAI-compatible v1 route (standard / reasoning param shaping).
        ensure_model_config(registry_name, args.model, _TEMPLATE_OPENAI_FC_KEY)
        handler = AzureOpenAIFCHandler(
            model_name=registry_name,
            temperature=args.temperature,
            registry_name=registry_name,
            is_fc_model=True,
            model_type=model_type,
            max_tokens=args.max_tokens,
        )
    for benchmark in args.benchmark:
        await run_benchmark(handler, benchmark, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive tool-calling robustness harness for multilingual-bfcl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category", required=True,
                        help="Benchmark category folder under data/benchmarks/ "
                             "(e.g. bfcl_multiple, wildtoolbench).")
    parser.add_argument("--lang-dir", required=True,
                        help="Language subfolder under the category (directory name, e.g. en, he, zh).")
    parser.add_argument("--benchmark", nargs="+", required=True,
                        help="Benchmark file stem(s) (without the .jsonl extension) under "
                             "data/benchmarks/<category>/<lang-dir>/, e.g. he_translatable_query.")
    parser.add_argument("--model", required=True,
                        help="Azure deployment name (the harness appends the -FC registry suffix).")
    parser.add_argument("--model-type", choices=[mt.value for mt in ModelType],
                        default=ModelType.STANDARD.value,
                        help="Deployment kind / route: 'standard' (v1 route, temperature + "
                             "max_tokens), 'reasoning' (v1 route, no temperature, "
                             "max_completion_tokens), or 'claude' (Foundry /anthropic route, "
                             "AnthropicFoundry client, native Anthropic message format).")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens generated per model turn.")
    parser.add_argument("--prices", default=str(DEFAULT_PRICES_CSV),
                        help="CSV of per-1M-token prices (model,input_per_mtok,output_per_mtok) "
                             "for the cost column.")
    parser.add_argument("--price-input", type=float, default=None,
                        help="USD per 1M input tokens, for the cost column. Overrides the "
                             "prices CSV for all models. Set together with --price-output.")
    parser.add_argument("--price-output", type=float, default=None,
                        help="USD per 1M output tokens, for the cost column. Overrides the "
                             "prices CSV for all models. Set together with --price-input.")
    parser.add_argument("--max-tool-call-attempts", type=int, default=5,
                        help="Max *tool-call* attempts per entry (turns that emit a function "
                             "call) before it is marked failed. Clarification turns do not "
                             "count against this budget.")
    parser.add_argument("--max-api-calls", type=int, default=10,
                        help="Hard ceiling on the TOTAL model turns (API calls) per entry, so "
                             "an entry that only ever asks for clarification still terminates.")
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
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries that already have a transcript (finish a run that "
                             "stopped early, e.g. out of credits), then rebuild "
                             "statistics.csv/summary.json over ALL transcripts on disk so they "
                             "cover the whole benchmark. Use with --limit 0.")
    parser.add_argument("--recompute-stats", action="store_true",
                        help="Do not call the API: rebuild statistics.csv and summary.json "
                             "(including the cost column) from existing transcripts for the "
                             "given --benchmark/--model/--category/--lang-dir.")
    args = parser.parse_args()

    # Route guard: Claude deployments are served on the Foundry /anthropic route
    # (--model-type claude), NOT the OpenAI-compatible /openai/v1 route. Sending a
    # Claude model to the v1 route returns a cryptic 404 "api_not_supported", so
    # catch the obvious mismatch here with an actionable message.
    model_is_claude = args.model.lower().startswith("claude")
    type_is_claude = args.model_type == ModelType.CLAUDE.value
    if model_is_claude and not type_is_claude:
        parser.error(
            f"'{args.model}' looks like a Claude deployment but --model-type is "
            f"'{args.model_type}'. Claude models are served on the Foundry /anthropic "
            "route; pass --model-type claude."
        )
    if type_is_claude and not model_is_claude:
        print(
            f"  ! --model-type claude with non-Claude deployment '{args.model}'; "
            "this will call the /anthropic route with that model name.",
            file=sys.stderr,
        )

    # The total API-call ceiling must leave room for every tool-call attempt; otherwise
    # the ceiling would cut the run short before the tool-call budget is even spent.
    if args.max_api_calls < args.max_tool_call_attempts:
        print(f"  ! --max-api-calls ({args.max_api_calls}) < --max-tool-call-attempts "
              f"({args.max_tool_call_attempts}); raising it to {args.max_tool_call_attempts}.", file=sys.stderr)
        args.max_api_calls = args.max_tool_call_attempts

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
