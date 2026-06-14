"""
classify_benchmark.py — Localization classifier for multilingual-bfcl benchmark entries.

For each entry in data/benchmarks/<category>/eng_base.json, classifies:
  - translatable_params          : "true" | "false"
  - localizable_query       : "true" | "false"   (only classified when translatable_params is "true")
  - localizable_parameters  : "true" | "false"   (only classified when translatable_params is "true")

A regex pre-filter checks the ground-truth argument values: if every value is
numeric, boolean, a date, or otherwise clearly non-translatable, the entry is
classified "non-translatable" locally and no API call is made for it. All
remaining entries are sent to the LLM (via langasync → the provider's native
Batch API), which classifies translatable_params itself — string values can still be
non-translatable (code, math expressions like "2x**2", identifiers) — plus the
two localizable_* dimensions. Non-translatable entries get empty localizable_*
columns.

Which classifications run is selectable on the command line:
  --translatable_params   classify only translatable_params
  --localizable      classify localizable_query + localizable_parameters
                     (auto-includes translatable_params, since localizable_parameters
                     depends on it)
With no classification flag, all classifications run.

Output file depends on the selection:
  - translatable_params only        -> base_classifications_params.csv (id, translatable_params)
  - everything else            -> base_classifications.csv (all columns)

Usage:
    # Submit and wait for results — all classifications (default):
    python scripts/classify_benchmark.py --category multiple

    # Only classify translatable_params:
    python scripts/classify_benchmark.py --category multiple --translatable_params

    # Only classify localizability (translatable_params is auto-included):
    python scripts/classify_benchmark.py --category multiple --localizable

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
import re
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
# Prompt — entries whose ground truth the regex pre-filter could not rule out
# reach the LLM, which makes the final translatable_params call (string values may
# still be code/math/identifiers) and classifies the two localizable_* dims.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant classifying function-calling benchmark entries for \
multilingual localization research. You will be given a benchmark entry consisting \
of a user query, available function definitions, and the ground-truth function call \
with argument values. Classify the entry on the given dimensions and return ONLY \
a JSON object — no explanation, no markdown fences."""

TRANSLATABILITY_PARAMETERS = """\
-- translatable_parameters
   Examine ONLY the ground-truth argument values (not the query text).
   - "true"     : At least one argument value contains natural-language \
text that would be written differently in another language — names, places, \
search terms, descriptions, human-readable labels, etc.
   - "false" : Every argument value is language-invariant. This \
includes numbers, booleans, numerical dates, AND strings that are not natural language: \
code, mathematical expressions (e.g. "2x**2", "lambda x: x+1"), programming \
identifiers, variable names (e.g. "x"), file paths and URLs. A value being a string does NOT make \
it translatable — what matters is whether it would change when written in a \
different language.
"""
TRANSLATABILITY_OUTPUT = '"translatable_params": "..."'

LOCALIZABILITY_PARAMETERS = """\
-- localizable_query
   "true"  if the query text contains culturally-anchored references that could \
be replaced with culturally equivalent references from another country or culture: \
place names, personal names, local institutions, local sports leagues/teams, \
local currencies, local public figures, etc.
   "false" if the query is purely abstract, mathematical, or technical — no \
cultural anchors that would need substitution.

-- localizable_parameters
   HARD RULE: If translatable_params is "false", this MUST be "false".
   Otherwise:
   "true"  if ground-truth argument string values contain culturally-anchored \
content (place names, person names, local entities) that would need to change \
when localizing to a different culture.
   "false" if the string values are universal technical identifiers, abstract \
labels, programming constructs, or culture-neutral terms.
"""
LOCALIZABILITY_OUTPUT = '"localizable_query": "...", "localizable_parameters": "..."'

CLASSIFICATION_TEMPLATE = """\
Classify this benchmark entry on the dimensions described below.

=== QUERY ===
{query}

=== GROUND TRUTH FUNCTIONS ===
{gt_functions_desc}

=== GROUND TRUTH PARAMETERS (correct function call + argument values) ===
{ground_truth}

=== CLASSIFICATION DIMENSIONS ===
{classification_parameters}

Return ONLY this JSON (no other text):
{{ {classification_output} }}"""


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


def format_gt_functions_desc(functions: list[dict], ground_truth_functions: list[dict]) -> str:
    """
    Render each function with its description and one line per parameter, including
    the parameter's own description and any enum constraint.

    The per-parameter detail matters for the translatability call: it lets the model
    tell a free-text value (e.g. material = "The material used for the sculpture")
    from a fixed token, instead of guessing from the bare value alone.
    """
    ground_truth_func_names = [list(func.keys())[0] for func in ground_truth_functions]
    lines = []
    for func in functions:
        func_name = func.get("name", "?")
        if func_name in ground_truth_func_names:
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            header = f"{func_name} — {desc}" if desc else func_name
            lines.append(header)
            for p_name, p_def in params.items():
                p_type = p_def.get("type", "any")
                req_mark = "*" if p_name in required else ""
                attrs = [p_type]
                enum = p_def.get("enum")
                if enum:
                    attrs.append("enum: " + " | ".join(str(v) for v in enum))
                p_desc = p_def.get("description", "")
                detail = f": {p_desc}" if p_desc else ""
                lines.append(f"  {p_name}{req_mark} ({', '.join(attrs)}){detail}")

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
    ground_truth = answer.get("ground_truth", [])
    gt_functions_desc = format_gt_functions_desc(entry.get("function", []), ground_truth)
    gt_text = format_ground_truth(ground_truth)

    return {
        "query": query,
        "gt_functions_desc": gt_functions_desc,
        "ground_truth": gt_text,
    }


# ---------------------------------------------------------------------------
# translatable_params pre-filter — local regex check, no API call needed
# ---------------------------------------------------------------------------

# Matches strings that are clearly language-invariant:
#   - booleans   : "true", "False"
#   - numbers    : "42", "-3.5", "1e10"
#   - dates      : "2024-01-05", "01/05/2024", "2024.01.05"
#   - datetimes  : "2024-01-05 14:30", "2024-01-05T14:30:00"
#   - times      : "14:30", "14:30:00"
NON_TRANSLATABLE_STRING_RE = re.compile(
    r"""^\s*(
        true|false
        |[-+]?\d+(\.\d+)?([eE][-+]?\d+)?                        # number
        |\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?  # ISO-ish date(/time)
        |\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?  # day-first/US date(/time)
        |\d{1,2}:\d{2}(:\d{2})?                                  # time
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_non_translatable_value(value: Any) -> bool:
    """True if a single ground-truth value is clearly language-invariant."""
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return value == "" or bool(NON_TRANSLATABLE_STRING_RE.match(value))
    if isinstance(value, list):
        return all(is_non_translatable_value(v) for v in value)
    if isinstance(value, dict):
        return all(is_non_translatable_value(v) for v in value.values())
    return False


def prefilter_non_translatable(ground_truth: list[dict]) -> bool:
    """
    True if the regex pre-filter can prove the entry is "non-translatable":
    every acceptable value of every argument is numeric, boolean (including
    "true"/"false" strings), a date/time, empty, or a nested structure of such
    values. These entries need no LLM call.

    False means the entry has at least one free-form string and goes to the
    LLM — which may STILL classify it as non-translatable (code, math
    expressions, identifiers, etc. are strings the regex cannot rule out).
    """
    for call in ground_truth:
        for params in call.values():
            for values in params.values():
                if not all(is_non_translatable_value(v) for v in values):
                    return False
    return True


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

VALID_BOOL = {"true", "false"}

def parse_classification(
    raw: str, entry_id: str, include_localizable: bool
) -> dict[str, str] | None:
    """
    Parse the JSON object the model returned. Always reads translatable_params; reads
    the localizable_* dimensions only when include_localizable is True.
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

    result = {
        "translatable_params": str(obj.get("translatable_params", "")).lower().strip(),
    }

    if include_localizable:
        # Localizable dims are only meaningful for translatable parameters —
        # blank them out when the LLM says non-translatable (mirrors the
        # regex-prefiltered rows).
        if result["translatable_params"] == "false":
            result["localizable_query"] = ""
            result["localizable_parameters"] = ""
        else:
            for key in ("localizable_query", "localizable_parameters"):
                result[key] = str(obj.get(key, "")).lower().strip()
                if result[key] not in VALID_BOOL:
                    print(f"[WARN] {entry_id}: unexpected {key}={result[key]!r}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Output / manifest
# ---------------------------------------------------------------------------

LOCALIZABLE_COLUMNS = ["localizable_query", "localizable_parameters"]


def csv_columns(include_localizable: bool) -> list[str]:
    """Output columns for the selected classifications. translatable_params is always present."""
    cols = ["id", "translatable_params"]
    if include_localizable:
        cols += LOCALIZABLE_COLUMNS
    return cols


def output_filename(include_localizable: bool) -> str:
    """CSV filename for the selected classifications."""
    return "base_classifications.csv" if include_localizable else "base_classifications_params.csv"


def classification_blocks(include_localizable: bool) -> tuple[str, str]:
    """Build the prompt's classification-dimension text and JSON-output fragment."""
    params = TRANSLATABILITY_PARAMETERS
    output = TRANSLATABILITY_OUTPUT
    if include_localizable:
        params = params + "\n" + LOCALIZABILITY_PARAMETERS
        output = output + ", " + LOCALIZABILITY_OUTPUT
    return params, output


def non_translatable_row(entry_id: str, include_localizable: bool) -> dict[str, str]:
    """Pre-classified row for an entry the regex filter proved non-translatable."""
    row = {"id": entry_id, "translatable_params": "false"}
    if include_localizable:
        row["localizable_query"] = ""
        row["localizable_parameters"] = ""
    return row


def error_row(entry_id: str, marker: str, include_localizable: bool) -> dict[str, str]:
    row = {"id": entry_id, "translatable_params": marker}
    if include_localizable:
        row["localizable_query"] = marker
        row["localizable_parameters"] = marker
    return row


def manifest_path(category: str) -> Path:
    return DATA_ROOT / category / "batch_manifest.json"

def write_csv(rows: list[dict[str, str]], output_path: Path, columns: list[str]) -> None:
    """Write classification rows to a CSV file."""
    rows.sort(key=lambda row: int(row["id"].split("_")[-1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    error_count = sum(1 for r in rows if "ERROR" in r.get("translatable_params", ""))
    print(f"\nDone. {len(rows)} rows written to {output_path}")
    if error_count:
        print(f"  {error_count} rows had errors — check stderr output above.")


def retrieve_and_write(batch_id: str, category: str) -> None:
    """
    Retrieve results for a completed Anthropic Message Batch and write the CSV.
    Uses the Anthropic SDK directly (not langasync) — provider must be Anthropic.

    The batch only contains entries the regex pre-filter could not rule out;
    the pre-filtered non-translatable rows and the batch-index → entry-id
    mapping are read from the manifest written at submit time
    (batch_manifest.json in the category folder).
    """
    import anthropic as anthropic_sdk

    mpath = manifest_path(category)
    if not mpath.exists():
        sys.exit(
            f"ERROR: {mpath} not found. It is written by the submit step and is "
            "required to map batch results back to entry IDs."
        )
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("batch_id") not in (None, "?", batch_id):
        print(
            f"[WARN] Manifest was written for batch {manifest['batch_id']!r}, "
            f"but you are retrieving {batch_id!r}.",
            file=sys.stderr,
        )
    # The classification selection is recorded at submit time so retrieval
    # produces the matching columns and output filename.
    include_localizable: bool = manifest.get("include_localizable", True)
    columns = csv_columns(include_localizable)
    llm_ids: list[str] = manifest["llm_ids"]               # batch index → entry id
    rows: list[dict[str, str]] = [
        non_translatable_row(eid, include_localizable) for eid in manifest["prefiltered_ids"]
    ]

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

    for item in client.messages.batches.results(batch_id):
        cid = item.custom_id
        # langasync uses sequential indices as custom IDs — map back via manifest
        try:
            eid = llm_ids[int(cid)]
        except (ValueError, IndexError):
            eid = cid

        if item.result.type == "succeeded":
            content_blocks = item.result.message.content
            raw = next((b.text for b in content_blocks if hasattr(b, "text")), "")
            parsed = parse_classification(raw, eid, include_localizable)
            if parsed is None:
                rows.append(error_row(eid, "PARSE_ERROR", include_localizable))
            else:
                rows.append({"id": eid, **parsed})

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
                    print(f"[ERR] {eid}: {err_kind} — (no message) raw={err_detail}", file=sys.stderr)
                else:
                    print(f"[ERR] {eid}: {err_kind} — {err_msg}", file=sys.stderr)
            else:
                print(f"[{err_type.upper()}] {eid}", file=sys.stderr)
            rows.append(error_row(eid, err_type.upper(), include_localizable))

    write_csv(rows, DATA_ROOT / category / output_filename(include_localizable), columns)


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def classify(
    category: str,
    provider: str,
    model_name: str,
    dry_run: bool,
    submit_only: bool,
    include_localizable: bool,
) -> None:
    bench_dir = DATA_ROOT / category
    source_path = bench_dir / "eng_base.json"
    answer_path = bench_dir / "possible_answer" / "eng_base.json"
    output_path = bench_dir / output_filename(include_localizable)
    columns = csv_columns(include_localizable)
    class_params, class_output = classification_blocks(include_localizable)

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

    # Split entries: ground truths the regex filter proves non-translatable are
    # classified locally (no API call); everything else goes to the LLM batch,
    # which makes the final translatable_params call (strings may still be code/math).
    llm_pairs: list[tuple[str, dict[str, str]]] = []
    prefiltered_ids: list[str] = []
    missing = []
    for entry in entries:
        eid = entry["id"]
        if eid not in answer_index:
            missing.append(eid)
            continue
        answer = answer_index[eid]
        if prefilter_non_translatable(answer.get("ground_truth", [])):
            prefiltered_ids.append(eid)
        else:
            llm_pairs.append((eid, build_prompt_input(entry, answer)))

    if missing:
        print(f"[WARN] {len(missing)} entries have no ground truth and will be skipped: {missing[:5]}...", file=sys.stderr)

    print(f"Entries total      : {len(llm_pairs) + len(prefiltered_ids)}")
    print(f"  non-translatable by regex (no API call) : {len(prefiltered_ids)}")
    print(f"  sent to LLM for classification          : {len(llm_pairs)}")

    if dry_run:
        if llm_pairs:
            print("\n--- DRY RUN: showing first LLM prompt ---")
            eid, vars_ = llm_pairs[0]
            p = build_prompt(class_params, class_output)
            print(p.format(**vars_))
        print(f"\n(Would submit {len(llm_pairs)} items to {provider}/{model_name}; "
              f"{len(prefiltered_ids)} classified locally as non-translatable)")
        return

    rows: list[dict[str, str]] = [
        non_translatable_row(eid, include_localizable) for eid in prefiltered_ids
    ]

    if not llm_pairs:
        print("All entries pre-filtered — nothing to send to the LLM.")
        write_csv(rows, output_path, columns)
        return

    # Build chain and wrap with langasync
    from langasync import batch_chain

    chain = make_chain(provider, model_name, class_params, class_output)
    batch_wrapper = batch_chain(chain)

    print(f"Submitting batch of {len(llm_pairs)} items to {provider}/{model_name}...")
    ids = [p[0] for p in llm_pairs]
    inputs = [p[1] for p in llm_pairs]

    job = await batch_wrapper.submit(inputs)
    job_id = getattr(job, "job_id", "?")
    print(f"Batch job submitted.")
    print(f"  Batch ID : {job_id}")
    print(f"  Category : {category}")
    print(f"  Output   : {output_path}")

    # Manifest lets --retrieve map batch indices back to entry IDs and merge
    # the locally pre-filtered non-translatable rows.
    mpath = manifest_path(category)
    mpath.write_text(json.dumps({
        "batch_id": job_id,
        "provider": provider,
        "model": model_name,
        "include_localizable": include_localizable,
        "llm_ids": ids,
        "prefiltered_ids": prefiltered_ids,
    }, indent=2), encoding="utf-8")
    print(f"  Manifest : {mpath}")

    if submit_only:
        print(
            "\nRun with --retrieve to fetch results when the batch finishes:\n"
            f"  python scripts/classify_benchmark.py --category {category} --retrieve {job_id}"
        )
        return

    print("Waiting for results (this can take up to 24 hours for large batches)...")

    # get_results() polls until the batch is complete
    batch_result = await job.get_results()

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
            rows.append(error_row(eid, "ERROR", include_localizable))
            continue

        raw = result_item.content if hasattr(result_item, "content") else str(result_item)
        parsed = parse_classification(raw, eid, include_localizable)
        if parsed is None:
            rows.append(error_row(eid, "PARSE_ERROR", include_localizable))
        else:
            rows.append({"id": eid, **parsed})


    write_csv(rows, output_path, columns)


# ---------------------------------------------------------------------------
# Model factory — swap provider here
# ---------------------------------------------------------------------------

def build_prompt(classification_parameters: str, classification_output: str):
    """
    Build the ChatPromptTemplate with the selected classification dimensions baked
    in. classification_parameters / classification_output are constant across all
    entries, so they are partialled in here, leaving query/functions/ground_truth
    as the per-entry variables.
    """
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", CLASSIFICATION_TEMPLATE),
    ]).partial(
        classification_parameters=classification_parameters,
        classification_output=classification_output,
    )


def make_chain(
    provider: str, model_name: str, classification_parameters: str, classification_output: str
):
    """
    Build a LangChain chain: SystemMessage + HumanMessage template → model → str.
    langasync wraps this chain, so switching the model is just changing this factory.
    """
    from langchain_core.output_parsers import StrOutputParser

    prompt = build_prompt(classification_parameters, classification_output)

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
        "--translatable_params",
        action="store_true",
        help="Classify translatable_params (true / false).",
    )
    parser.add_argument(
        "--localizable",
        action="store_true",
        help="Classify localizable_query and localizable_parameters. Auto-includes "
             "translatable_params, since localizable_parameters depends on it.",
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

    # Retrieve-only path — no asyncio needed. The classification selection is read
    # from the manifest, so the --translatable_params / --localizable flags are ignored.
    if args.retrieve:
        if args.provider != "anthropic":
            parser.error("--retrieve only supports --provider anthropic.")
        retrieve_and_write(args.retrieve, args.category)
        return

    # Selection: no classification flag => run everything. --localizable auto-includes
    # translatable_params (localizable_parameters depends on it), so translatable_params is
    # always computed; include_localizable is the only real switch.
    include_localizable = args.localizable or not (args.translatable_params or args.localizable)

    asyncio.run(classify(
        category=args.category,
        provider=args.provider,
        model_name=args.model,
        dry_run=args.dry_run,
        submit_only=args.submit_only,
        include_localizable=include_localizable,
    ))


if __name__ == "__main__":
    main()
