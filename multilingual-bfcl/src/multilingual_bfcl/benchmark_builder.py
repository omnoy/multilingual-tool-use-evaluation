"""
Benchmark builder: batch-translates local BFCL benchmark files into target locales.

Unlike the old per-string synchronous translator, this uses the provider's native
Batch API (via langasync) — one batch item per (test case, locale) — mirroring
scripts/classify_benchmark.py.

Input/output live under data/benchmarks/<category>/:
  - source question file : <source>                 (e.g. eng_base.json, JSONL)
  - source ground truth  : possible_answer/<source> (required for level=full)
  - output question file : <locale>_<level>.json    (e.g. he_full.json)
  - output ground truth  : possible_answer/<locale>_<level>.json

`--source` chooses the input (e.g. eng_base_translatable.json → he_translatable_full.json).

Levels (see translator.LocalizationLevel):
  - query : translate only the user query; ground truth left unchanged.
  - full  : translate query + textual ground-truth parameter values. Function
            names, descriptions, enums, numbers, etc. stay in English.

A manifest (translate_manifest.json) records the batch id and the
batch-index → (id, locale) mapping so results can be retrieved later.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from multilingual_bfcl.localization.locale_config import get_locale
from multilingual_bfcl.localization.translator import (
    DEFAULT_TRANSLATION_MODEL,
    LocalizationLevel,
    apply_translation,
    build_input,
    build_prompt,
    make_chain,
    parse_translation,
)

# Root of the multilingual-bfcl package (two levels up from this file)
_PACKAGE_ROOT = Path(__file__).parent.parent.parent
_BENCHMARK_DIR = _PACKAGE_ROOT / "data" / "benchmarks"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON (one object per line)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda row: int(row["id"].split("_")[-2]))
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def output_filename(source: str, locale_code: str, level: LocalizationLevel) -> str:
    """Map a source filename to its translated counterpart.

    eng_base.json              -> <locale>_<level>.json
    eng_base_translatable.json -> <locale>_translatable_<level>.json
    anything else              -> <stem>_<locale>_<level>.json
    """
    stem = Path(source).stem
    lv = level.value
    if stem.startswith("eng_base"):
        rest = stem[len("eng_base"):]          # "" or "_translatable"
        return f"{locale_code}{rest}_{lv}.json"
    return f"{stem}_{locale_code}_{lv}.json"


def manifest_path(category: str) -> Path:
    return _BENCHMARK_DIR / category / "translate_manifest.json"


def tag_batch_job(handle, **fields) -> None:
    """Write human-readable labels into the langasync job file's `metadata` dict.

    e.g. tag_batch_job(handle, job="translate_he_full", task="translate", ...).
    Only `metadata` (not arbitrary top-level keys) is persisted by langasync, and
    its status-update path round-trips the file, so values written here survive
    later saves. Best-effort: a failure here must not fail the submission.
    """
    try:
        path = handle.repository.storage_dir / f"{handle.job_id.replace('/', '_')}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("metadata", {}).update(fields)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - tagging is non-critical
        print(f"[WARN] could not tag batch job file with metadata: {e}", file=sys.stderr)


def _id_sort_key(record: dict[str, Any]):
    """Sort by the trailing integer of the source id when possible."""
    base = record.get("source_id", record.get("id", ""))
    try:
        return (0, int(str(base).split("_")[-1]))
    except (ValueError, IndexError):
        return (1, str(base))


# ---------------------------------------------------------------------------
# Reassembly + writing
# ---------------------------------------------------------------------------

def _write_outputs(
    category: str,
    source: str,
    level: LocalizationLevel,
    units: list[dict[str, str]],
    raws: list[str | None],
) -> None:
    """Reassemble translated entries from raw model replies and write per-locale files."""
    bench = _BENCHMARK_DIR / category
    entries_by_id = {e["id"]: e for e in load_jsonl(bench / source)}

    apath = bench / "possible_answer" / source
    answers = {a["id"]: a for a in load_jsonl(apath)} if apath.exists() else {}

    q_by_locale: dict[str, list] = defaultdict(list)
    a_by_locale: dict[str, list] = defaultdict(list)
    n_fail = 0

    for unit, raw in zip(units, raws):
        eid = unit["id"]
        locale = get_locale(unit["locale"])
        sid = f"{eid}_{locale.code}"

        entry = entries_by_id.get(eid)
        if entry is None:
            print(f"[WARN] {sid}: source entry not found, skipping.", file=sys.stderr)
            n_fail += 1
            continue

        parsed = parse_translation(raw, sid) if raw is not None else None
        result = (
            apply_translation(entry, answers.get(eid), locale, level, parsed)
            if parsed is not None else None
        )
        if result is None:
            n_fail += 1
            continue

        q_entry, a_entry = result
        q_by_locale[locale.code].append(q_entry)
        if a_entry is not None:
            a_by_locale[locale.code].append(a_entry)

    for locale_code, q_entries in q_by_locale.items():
        q_entries.sort(key=_id_sort_key)
        out_q = bench / output_filename(source, locale_code, level)
        write_jsonl(q_entries, out_q)
        print(f"[done] {len(q_entries)} entries -> {out_q}")

        a_entries = a_by_locale.get(locale_code)
        if a_entries:
            a_entries.sort(key=_id_sort_key)
            out_a = bench / "possible_answer" / output_filename(source, locale_code, level)
            write_jsonl(a_entries, out_a)
            print(f"[done] {len(a_entries)} answers -> {out_a}")

    if n_fail:
        print(f"  {n_fail} item(s) failed to translate — see warnings above.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Submit (and optionally wait) — async, uses langasync
# ---------------------------------------------------------------------------

async def translate_benchmark(
    category: str,
    locales: list[str],
    level: LocalizationLevel = LocalizationLevel.QUERY,
    source: str = "eng_base.json",
    provider: str = "anthropic",
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    limit: int | None = None,
    dry_run: bool = False,
    submit_only: bool = False,
) -> str | None:
    """Submit a translation batch (and wait for results unless submit_only).

    Returns the batch id (or None for dry-run / nothing-to-do).
    """
    bench = _BENCHMARK_DIR / category
    src_q = bench / source
    if not src_q.exists():
        sys.exit(f"ERROR: {src_q} not found.")

    entries = load_jsonl(src_q)
    if limit is not None:
        entries = entries[:limit]

    apath = bench / "possible_answer" / source
    answers = {a["id"]: a for a in load_jsonl(apath)} if apath.exists() else {}
    if level == LocalizationLevel.FULL and not apath.exists():
        sys.exit(
            f"ERROR: level=full needs ground truth, but {apath} not found."
        )

    # Build one batch item per (locale, test case), in a deterministic order.
    units: list[dict[str, str]] = []
    inputs: list[dict[str, str]] = []
    missing_gt = 0
    for locale_code in locales:
        locale = get_locale(locale_code)
        for entry in entries:
            ans = answers.get(entry["id"])
            if level == LocalizationLevel.FULL and ans is None:
                missing_gt += 1
                continue
            inputs.append(build_input(entry, ans, locale, level))
            units.append({"id": entry["id"], "locale": locale_code})

    if missing_gt:
        print(f"[WARN] {missing_gt} entries have no ground truth and were skipped (level=full).",
              file=sys.stderr)

    print(f"Source     : {src_q.name}")
    print(f"Locales    : {', '.join(locales)}")
    print(f"Level      : {level.value}")
    print(f"Batch items: {len(inputs)} ({len(entries)} entries × {len(locales)} locale(s))")

    if dry_run:
        if inputs:
            print("\n--- DRY RUN: first prompt ---")
            print(build_prompt(level).format(**inputs[0]))
        print(f"\n(Would submit {len(inputs)} items to {provider}/{model_name})")
        return None

    if not inputs:
        print("Nothing to translate.")
        return None

    from langasync import batch_chain

    chain = make_chain(provider, model_name, level)
    wrapper = batch_chain(chain)

    print(f"Submitting batch of {len(inputs)} items to {provider}/{model_name}...")
    job = await wrapper.submit(inputs)
    job_id = getattr(job, "job_id", "?")
    print(f"Batch job submitted.\n  Batch ID : {job_id}\n  Category : {category}")

    # Tag the langasync job file so the batch is identifiable by what it did.
    job_label = f"translate_{'+'.join(locales)}_{level.value}"
    tag_batch_job(
        job,
        job=job_label,
        task="translate",
        category=category,
        source=source,
        level=level.value,
        locales=locales,
    )

    mpath = manifest_path(category)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({
        "batch_id": job_id,
        "job": job_label,
        "provider": provider,
        "model": model_name,
        "category": category,
        "source": source,
        "level": level.value,
        "locales": locales,
        "units": units,
    }, indent=2), encoding="utf-8")
    print(f"  Manifest : {mpath}")

    if submit_only:
        print(
            "\nRetrieve results when the batch finishes:\n"
            f"  python scripts/translate_benchmark.py --category {category} --retrieve {job_id}"
        )
        return job_id

    print("Waiting for results (can take up to 24h for large batches)...")
    batch_result = await job.get_results()
    results_list = batch_result.results if hasattr(batch_result, "results") else list(batch_result)
    if len(results_list) != len(units):
        print(f"[WARN] Expected {len(units)} results, got {len(results_list)}; ids may misalign.",
              file=sys.stderr)

    raws: list[str | None] = []
    for item in results_list:
        if hasattr(item, "success") and not item.success:
            print(f"[WARN] batch item failed — {getattr(item, 'error', '?')}", file=sys.stderr)
            raws.append(None)
        else:
            raws.append(item.content if hasattr(item, "content") else str(item))

    _write_outputs(category, source, level, units, raws)
    return job_id


# ---------------------------------------------------------------------------
# Retrieve a previously submitted Anthropic batch — sync, uses the SDK directly
# ---------------------------------------------------------------------------

def retrieve_translation(batch_id: str, category: str) -> None:
    """Fetch a completed Anthropic Message Batch and write the translated files."""
    import anthropic as anthropic_sdk

    mpath = manifest_path(category)
    if not mpath.exists():
        sys.exit(f"ERROR: {mpath} not found (written at submit time; required to map results).")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("batch_id") not in (None, "?", batch_id):
        print(f"[WARN] Manifest is for batch {manifest['batch_id']!r}, retrieving {batch_id!r}.",
              file=sys.stderr)

    units: list[dict[str, str]] = manifest["units"]
    source: str = manifest["source"]
    level = LocalizationLevel(manifest["level"])

    client = anthropic_sdk.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        rc = batch.request_counts
        done = rc.succeeded + rc.errored + rc.expired + rc.canceled
        total = rc.processing + done
        sys.exit(f"ERROR: Batch {batch_id!r} is '{batch.processing_status}' ({done}/{total}). Try later.")

    print(f"Retrieving results for batch {batch_id!r}...")
    raws: list[str | None] = [None] * len(units)
    for item in client.messages.batches.results(batch_id):
        try:
            idx = int(item.custom_id)
        except (ValueError, TypeError):
            print(f"[WARN] unexpected custom_id {item.custom_id!r}; skipping.", file=sys.stderr)
            continue
        if not (0 <= idx < len(units)):
            print(f"[WARN] custom_id {idx} out of range; skipping.", file=sys.stderr)
            continue

        if item.result.type == "succeeded":
            blocks = item.result.message.content
            raws[idx] = next((b.text for b in blocks if hasattr(b, "text")), "")
        else:
            err = item.result.type
            detail = getattr(getattr(item.result, "error", None), "message", "")
            print(f"[{err.upper()}] unit {idx}{(' — ' + detail) if detail else ''}", file=sys.stderr)

    _write_outputs(category, source, level, units, raws)


# ---------------------------------------------------------------------------
# Discovery helpers (used by the CLI status/categories commands)
# ---------------------------------------------------------------------------

def _bfcl_data_dir() -> Path:
    import bfcl_eval
    return Path(bfcl_eval.__file__).parent / "data"


def list_available_categories() -> list[str]:
    """All BFCL category names for which a source file exists in the bfcl_eval package."""
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    prefix = f"{VERSION_PREFIX}_"
    return sorted(p.stem[len(prefix):] for p in _bfcl_data_dir().glob(f"{prefix}*.json"))


def list_built_benchmarks() -> dict[str, list[str]]:
    """{category: [output json stems]} for already-built benchmark files."""
    result: dict[str, list[str]] = {}
    if not _BENCHMARK_DIR.exists():
        return result
    for cat_dir in sorted(_BENCHMARK_DIR.iterdir()):
        if cat_dir.is_dir():
            files = sorted(p.stem for p in cat_dir.glob("*.json"))
            if files:
                result[cat_dir.name] = files
    return result
