"""
classify_benchmark.py — Localization classifier for multilingual-bfcl benchmark entries.

For each entry in data/benchmarks/<category>/eng_base.json, classifies:
  - parameter_type    : "universal" | "textual" | "mixed"
  - convertible_units : "true" | "false"
  - localizable_query : "true" | "false"
  - localizable_parameters : "true" | "false"

Uses langasync to dispatch a single batch job to the provider's native Batch API
(Anthropic Message Batches, OpenAI Batch API, etc.), then writes results to
data/benchmarks/<category>/base_classifications.csv.

Usage:
    # Submit and wait for results (default):
    python scripts/classify_benchmark.py --category multiple

    # Submit only — print the batch ID and exit immediately:
    python scripts/classify_benchmark.py --category multiple --submit-only

    # Retrieve results for a previously submitted batch (Anthropic only):
    python scripts/classify_benchmark.py --category multiple --retrieve msgbatch_01AbCdEf...

    # Other options:
    python scripts/classify_benchmark.py --category multiple --model claude-opus-4-8 --provider anthropic
    python scripts/classify_benchmark.py --category multiple --model gpt-4o-mini --provider openai
    python scripts/classify_benchmark.py --category multiple --dry-run

Requirements:
    pip install langasync langchain-anthropic langchain-openai

Environment variables (put in multilingual-bfcl/.env):
    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...      # only needed when --provider openai
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — resolve package root regardless of where the script is run from
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent           # multilingual-bfcl/
DATA_ROOT = PACKAGE_ROOT / "data" / "benchmarks"

load_dotenv(PACKAGE_ROOT / ".env")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant classifying function-calling benchmark entries for \
multilingual localization research. You will be given a benchmark entry consisting \
of a user query, available function definitions, and the ground-truth function call \
with argument values. Classify the entry on exactly four dimensions and return ONLY \
a JSON object — no explanation, no markdown fences."""

CLASSIFICATION_TEMPLATE = """\
Classify this benchmark entry on the four dimensions described below.

=== QUERY ===
{query}

=== AVAILABLE FUNCTIONS ===
{functions}

=== GROUND TRUTH (correct function call + argument values) ===
{ground_truth}

=== CLASSIFICATION DIMENSIONS ===

1. parameter_type
   Examine ONLY the ground-truth argument values (not the query text).
   - "universal"  : Every argument value that carries semantic content is \
non-textual — numbers, booleans, arrays of numbers, coordinates, or empty/default. \
No string argument carries a human-readable name, place, or concept.
   - "textual"    : Every semantically meaningful argument value is a string \
(names, places, search terms, natural-language labels, etc.).
   - "mixed"      : Ground truth contains BOTH non-textual values (numbers, \
booleans, numeric arrays) AND textual strings that carry semantic meaning.

2. convertible_units
   "true"  if the query text or ground-truth argument values mention physical \
measurement units that can differ between measurement systems \
(e.g. meters/feet, km/miles, kg/lb, °C/°F, liters/gallons, m/s, sq ft, etc.).
   "false" otherwise.

3. localizable_query
   "true"  if the query text contains culturally-anchored references that could \
be replaced with culturally equivalent references from another country or culture: \
place names, personal names, local institutions, local sports leagues/teams, \
local currencies, local public figures, etc.
   "false" if the query is purely abstract, mathematical, or technical — no \
cultural anchors that would need substitution.

4. localizable_parameters
   HARD RULE: If parameter_type is "universal", this MUST be "false".
   Otherwise:
   "true"  if ground-truth argument string values contain culturally-anchored \
content (place names, person names, local entities) that would need to change \
when localizing to a different culture.
   "false" if the string values are universal technical identifiers, abstract \
labels, programming constructs, or culture-neutral terms.

Return ONLY this JSON (no other text):
{{"parameter_type": "...", "convertible_units": "...", "localizable_query": "...", "localizable_parameters": "..."}}"""


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a file where each line is a JSON object (JSONL or newline-delimited JSON)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_ground_truth(ground_truth: list[dict]) -> str:
    """
    Render the ground-truth structure as readable text for the prompt.

    Ground truth format:
        [{"function_name": {"param": [acceptable_value, ...]}}]

    We show the first acceptable value for each param to keep the prompt concise.
    """
    lines = []
    for call in ground_truth:
        for func_name, params in call.items():
            lines.append(f"Function: {func_name}")
            for param, values in params.items():
                # values is a list of acceptable values; show the first non-empty one
                display = next(
                    (v for v in values if v != "" and v is not None),
                    values[0] if values else ""
                )
                lines.append(f"  {param} = {json.dumps(display, ensure_ascii=False)}")
    return "\n".join(lines)


def format_functions(functions: list[dict]) -> str:
    """Render function signatures concisely (name + param names + types)."""
    lines = []
    for func in functions:
        name = func.get("name", "?")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])
        param_strs = []
        for p_name, p_def in params.items():
            p_type = p_def.get("type", "any")
            req_mark = "*" if p_name in required else ""
            param_strs.append(f"{p_name}{req_mark}: {p_type}")
        lines.append(f"{name}({', '.join(param_strs)}) — {desc}")
    return "\n".join(lines)


def build_prompt_input(entry: dict, answer: dict) -> dict[str, str]:
    """Build the dict of template variables for one entry."""
    # Query: flatten conversation turns → user messages only
    query_parts = []
    for turn in entry.get("question", []):
        for msg in turn:
            if msg.get("role") == "user":
                query_parts.append(msg["content"])
    query = "\n".join(query_parts)

    functions_text = format_functions(entry.get("function", []))
    gt_text = format_ground_truth(answer.get("ground_truth", []))

    return {
        "query": query,
        "functions": functions_text,
        "ground_truth": gt_text,
    }


# ---------------------------------------------------------------------------
# Model factory — swap provider here
# ---------------------------------------------------------------------------

def make_chain(provider: str, model_name: str):
    """
    Build a LangChain chain: SystemMessage + HumanMessage template → model → str.
    langasync wraps this chain, so switching the model is just changing this factory.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", CLASSIFICATION_TEMPLATE),
    ])

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=model_name, max_tokens=256)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=model_name, temperature=0, max_tokens=256)
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Choose 'anthropic' or 'openai'.")

    return prompt | model | StrOutputParser()


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

VALID_PARAMETER_TYPES = {"universal", "textual", "mixed"}
VALID_BOOL = {"true", "false"}

def parse_classification(raw: str, entry_id: str) -> dict[str, str] | None:
    """
    Parse the JSON object the model returned.
    Returns None and prints a warning on failure.
    """
    # Strip accidental markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON substring
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start:end])
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse JSON for {entry_id}: {raw!r}", file=sys.stderr)
                return None
        else:
            print(f"[WARN] No JSON found for {entry_id}: {raw!r}", file=sys.stderr)
            return None

    # Normalise and validate
    result = {
        "parameter_type": str(obj.get("parameter_type", "")).lower().strip(),
        "convertible_units": str(obj.get("convertible_units", "")).lower().strip(),
        "localizable_query": str(obj.get("localizable_query", "")).lower().strip(),
        "localizable_parameters": str(obj.get("localizable_parameters", "")).lower().strip(),
    }

    # Enforce hard rule: universal → localizable_parameters must be false
    if result["parameter_type"] == "universal":
        if result["localizable_parameters"] != "false":
            print(
                f"[FIX] {entry_id}: parameter_type=universal but localizable_parameters="
                f"{result['localizable_parameters']!r} → forcing to 'false'",
                file=sys.stderr,
            )
            result["localizable_parameters"] = "false"

    # Soft validation warnings
    if result["parameter_type"] not in VALID_PARAMETER_TYPES:
        print(f"[WARN] {entry_id}: unexpected parameter_type={result['parameter_type']!r}", file=sys.stderr)
    for key in ("convertible_units", "localizable_query", "localizable_parameters"):
        if result[key] not in VALID_BOOL:
            print(f"[WARN] {entry_id}: unexpected {key}={result[key]!r}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["id", "parameter_type", "convertible_units", "localizable_query", "localizable_parameters"]


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write classification rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    error_count = sum(1 for r in rows if "ERROR" in r.get("parameter_type", ""))
    print(f"\nDone. {len(rows)} rows written to {output_path}")
    if error_count:
        print(f"  {error_count} rows had errors — check stderr output above.")


def retrieve_and_write(batch_id: str, output_path: Path) -> None:
    """
    Retrieve results for a completed Anthropic Message Batch and write the CSV.
    Uses the Anthropic SDK directly (not langasync) — provider must be Anthropic.
    """
    import anthropic as anthropic_sdk

    client = anthropic_sdk.Anthropic()

    # Check status first so we can give a clear error if not ended yet
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        rc = batch.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.expired + rc.canceled
        done  = rc.succeeded + rc.errored + rc.expired + rc.canceled
        sys.exit(
            f"ERROR: Batch {batch_id!r} is still '{batch.processing_status}' "
            f"({done}/{total} done). Wait for it to finish, then retry."
        )

    print(f"Retrieving results for batch {batch_id!r}...")
    rows: list[dict[str, str]] = []

    for item in client.messages.batches.results(batch_id):
        cid = item.custom_id

        if item.result.type == "succeeded":
            content_blocks = item.result.message.content
            raw = next((b.text for b in content_blocks if hasattr(b, "text")), "")
            parsed = parse_classification(raw, cid)
            if parsed is None:
                rows.append({"id": cid, "parameter_type": "PARSE_ERROR",
                             "convertible_units": "PARSE_ERROR",
                             "localizable_query": "PARSE_ERROR",
                             "localizable_parameters": "PARSE_ERROR"})
            else:
                rows.append({"id": cid, **parsed})

        else:
            err_type = item.result.type  # "errored", "expired", "canceled"
            if err_type == "errored":
                err_obj  = item.result.error
                err_kind = getattr(err_obj, "type", "unknown")
                err_msg  = getattr(err_obj, "message", "")
                # If message is empty, dump the full object so we can see what's there
                if not err_msg:
                    try:
                        err_detail = err_obj.model_dump() if hasattr(err_obj, "model_dump") else vars(err_obj)
                    except Exception:
                        err_detail = repr(err_obj)
                    print(f"[ERR] {cid}: {err_kind} — (no message) raw={err_detail}", file=sys.stderr)
                else:
                    print(f"[ERR] {cid}: {err_kind} — {err_msg}", file=sys.stderr)
            else:
                print(f"[{err_type.upper()}] {cid}", file=sys.stderr)
            rows.append({"id": cid, "parameter_type": err_type.upper(),
                         "convertible_units": err_type.upper(),
                         "localizable_query": err_type.upper(),
                         "localizable_parameters": err_type.upper()})

    rows.sort(key=lambda row: int(row["id"]))
    write_csv(rows, output_path)


async def classify(
    category: str,
    provider: str,
    model_name: str,
    dry_run: bool,
    submit_only: bool,
) -> None:
    bench_dir = DATA_ROOT / category
    source_path = bench_dir / "eng_base.json"
    answer_path = bench_dir / "possible_answer" / "eng_base.json"
    output_path = bench_dir / "base_classifications.csv"

    if not source_path.exists():
        sys.exit(f"ERROR: {source_path} not found.")
    if not answer_path.exists():
        sys.exit(f"ERROR: {answer_path} not found.")

    print(f"Loading benchmark: {source_path}")
    entries = load_jsonl(source_path)
    print(f"Loading ground truth: {answer_path}")
    answers = load_jsonl(answer_path)

    # Index answers by id for O(1) lookup
    answer_index = {a["id"]: a for a in answers}

    # Pair entries with their ground truth
    pairs: list[tuple[str, dict[str, str]]] = []
    missing = []
    for entry in entries:
        eid = entry["id"]
        if eid not in answer_index:
            missing.append(eid)
            continue
        prompt_vars = build_prompt_input(entry, answer_index[eid])
        pairs.append((eid, prompt_vars))

    if missing:
        print(f"[WARN] {len(missing)} entries have no ground truth and will be skipped: {missing[:5]}...", file=sys.stderr)

    print(f"Entries to classify: {len(pairs)}")

    if dry_run:
        print("\n--- DRY RUN: showing first prompt ---")
        eid, vars_ = pairs[0]
        from langchain_core.prompts import ChatPromptTemplate
        p = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", CLASSIFICATION_TEMPLATE),
        ])
        print(p.format(**vars_))
        print(f"\n(Would submit {len(pairs)} items to {provider}/{model_name})")
        return

    # Build chain and wrap with langasync
    from langasync import batch_chain

    chain = make_chain(provider, model_name)
    batch_wrapper = batch_chain(chain)

    print(f"Submitting batch of {len(pairs)} items to {provider}/{model_name}...")
    ids = [p[0] for p in pairs]
    inputs = [p[1] for p in pairs]

    job = await batch_wrapper.submit(inputs)
    job_id = getattr(job, "job_id", "?")
    print(f"Batch job submitted.")
    print(f"  Batch ID : {job_id}")
    print(f"  Category : {category}")
    print(f"  Output   : {output_path}")

    if submit_only:
        print(
            "\nRun with --retrieve to fetch results when the batch finishes:\n"
            f"  python scripts/classify_benchmark.py --category {category} --retrieve {job_id}"
        )
        return

    print("Waiting for results (this can take up to 24 hours for large batches)...")

    # get_results() polls until the batch is complete
    batch_result = await job.get_results()

    # Parse results and build rows
    rows: list[dict[str, str]] = []

    results_list = batch_result.results if hasattr(batch_result, "results") else list(batch_result)
    if len(results_list) != len(ids):
        print(
            f"[WARN] Expected {len(ids)} results, got {len(results_list)}. "
            "IDs may be misaligned.",
            file=sys.stderr,
        )

    for eid, result_item in zip(ids, results_list):
        if hasattr(result_item, "success") and not result_item.success:
            print(f"[WARN] {eid}: batch item failed — {getattr(result_item, 'error', '?')}", file=sys.stderr)
            rows.append({"id": eid, "parameter_type": "ERROR", "convertible_units": "ERROR",
                         "localizable_query": "ERROR", "localizable_parameters": "ERROR"})
            continue

        raw = result_item.content if hasattr(result_item, "content") else str(result_item)
        parsed = parse_classification(raw, eid)
        if parsed is None:
            rows.append({"id": eid, "parameter_type": "PARSE_ERROR", "convertible_units": "PARSE_ERROR",
                         "localizable_query": "PARSE_ERROR", "localizable_parameters": "PARSE_ERROR"})
        else:
            rows.append({"id": eid, **parsed})

    write_csv(rows, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify multilingual-bfcl benchmark entries via LLM batch API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Benchmark category name, e.g. 'multiple'. "
             "Must have eng_base.json and possible_answer/eng_base.json under data/benchmarks/<category>/.",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider to use for classification.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-8",
        help="Model name passed to the provider SDK.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt and exit without making any API calls.",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit the batch and print the batch ID, then exit immediately "
             "without waiting for results. Use --retrieve later to fetch them.",
    )
    parser.add_argument(
        "--retrieve",
        metavar="BATCH_ID",
        default=None,
        help="Skip submission entirely. Retrieve results for a previously submitted "
             "Anthropic Message Batch and write the CSV. "
             "Requires --provider anthropic (the default). "
             "Example: --retrieve msgbatch_01AbCdEf...",
    )
    args = parser.parse_args()

    if args.submit_only and args.retrieve:
        parser.error("--submit-only and --retrieve are mutually exclusive.")
    if args.retrieve and args.dry_run:
        parser.error("--dry-run has no effect with --retrieve.")

    # Retrieve-only path — no asyncio needed
    if args.retrieve:
        if args.provider != "anthropic":
            parser.error("--retrieve only supports --provider anthropic.")
        output_path = DATA_ROOT / args.category / "base_classifications.csv"
        retrieve_and_write(args.retrieve, output_path)
        return

    asyncio.run(classify(
        category=args.category,
        provider=args.provider,
        model_name=args.model,
        dry_run=args.dry_run,
        submit_only=args.submit_only,
    ))


if __name__ == "__main__":
    main()
