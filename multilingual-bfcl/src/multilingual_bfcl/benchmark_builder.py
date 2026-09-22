"""
Benchmark builder: batch-translates local BFCL benchmark files into target locales.

This uses the Azure OpenAI Batch API — one batch item per (test case, locale) —
mirroring scripts/classify_benchmark.py.

Input/output live under data/benchmarks/<category>/:
  - source question file : <source>                 (e.g. eng_base.json, JSONL)
  - source ground truth  : possible_answer/<source> (required for level=full)
  - output question file : <locale>_<level>.json    (e.g. he_full.json)
  - output ground truth  : possible_answer/<locale>_<level>.json

`--source` chooses the input (e.g. eng_translatable.json → he_translatable_full.json).

Levels (see translator.LocalizationLevel):
  - query : translate only the user query; ground truth left unchanged.
  - full  : translate query + textual ground-truth parameter values. Function
            names, descriptions, enums, numbers, etc. stay in English.

A manifest (translate_manifest.json) records the batch id and the
batch-index → (id, locale) mapping so results can be retrieved later.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from multilingual_bfcl.azure_batch import (
    build_chat_request,
    collect_results,
    get_batch,
    is_finished,
    status_line,
    submit_batch,
)
from multilingual_bfcl.azure_client import (
    ModelType,
    build_chat_params,
    make_sync_client,
)
from multilingual_bfcl.localization.locale_config import get_locale
from multilingual_bfcl.localization.translator import (
    DEFAULT_TRANSLATION_MODEL,
    LocalizationLevel,
    add_language_descriptors_to_entry,
    apply_translation,
    build_input,
    build_prompt,
    parse_translation,
    render_messages,
)

# Cap on translation output tokens per batch item.
_TRANSLATION_MAX_TOKENS = 4096
# Seconds between batch status polls while waiting for results.
_POLL_INTERVAL_S = 30

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
    eng_translatable.json -> <locale>_translatable_<level>.json
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
# Submit (and optionally wait) — async, uses the Azure OpenAI Batch API
# ---------------------------------------------------------------------------

async def translate_benchmark(
    category: str,
    locales: list[str],
    level: LocalizationLevel = LocalizationLevel.QUERY,
    source: str = "eng_base.json",
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    model_type: ModelType = ModelType.STANDARD,
    limit: int | None = None,
    dry_run: bool = False,
    submit_only: bool = False,
) -> str | None:
    """Submit an Azure translation batch (and wait for results unless submit_only).

    `model_name` is the Azure deployment name. Returns the batch id (or None for
    dry-run / nothing-to-do).
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
            for msg in render_messages(level, inputs[0]):
                print(f"[{msg['role']}]\n{msg['content']}\n")
        print(f"\n(Would submit {len(inputs)} items to Azure deployment {model_name!r})")
        return None

    if not inputs:
        print("Nothing to translate.")
        return None

    # Build one Batch API request per item; custom_id is the item index (string),
    # matching the manifest's `units` order so retrieval maps results back positionally.
    requests = [
        build_chat_request(
            idx,
            build_chat_params(
                model_name,
                render_messages(level, item),
                model_type=model_type,
                max_tokens=_TRANSLATION_MAX_TOKENS,
            ),
        )
        for idx, item in enumerate(inputs)
    ]

    client = make_sync_client()
    print(f"Submitting batch of {len(requests)} items to Azure deployment {model_name!r}...")
    batch = submit_batch(
        client,
        requests,
        metadata={"task": "translate", "category": category, "level": level.value},
    )
    batch_id = batch.id
    print(f"Batch job submitted.\n  Batch ID : {batch_id}\n  Category : {category}")

    mpath = manifest_path(category)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({
        "batch_id": batch_id,
        "task": "translate",
        "model": model_name,
        "model_type": model_type.value,
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
            f"  python scripts/translate_benchmark.py --category {category} --retrieve {batch_id}"
        )
        return batch_id

    print("Waiting for results (can take up to 24h for large batches)...")
    while True:
        batch = get_batch(client, batch_id)
        if is_finished(batch):
            break
        print(f"  {status_line(batch)} — checking again in {_POLL_INTERVAL_S}s", file=sys.stderr)
        await asyncio.sleep(_POLL_INTERVAL_S)

    _retrieve_and_write(client, batch_id, category, source, level, units)
    return batch_id


# ---------------------------------------------------------------------------
# Retrieve a previously submitted Azure batch — sync, uses the SDK directly
# ---------------------------------------------------------------------------

def retrieve_translation(batch_id: str, category: str) -> None:
    """Fetch a completed Azure OpenAI batch and write the translated files."""
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

    client = make_sync_client()
    _retrieve_and_write(client, batch_id, category, source, level, units)


def _retrieve_and_write(
    client,
    batch_id: str,
    category: str,
    source: str,
    level: LocalizationLevel,
    units: list[dict[str, str]],
) -> None:
    """Download a completed batch's results and write per-locale output files."""
    print(f"Retrieving results for batch {batch_id!r}...")
    results = collect_results(client, batch_id)

    raws: list[str | None] = [None] * len(units)
    for cid, item in results.items():
        try:
            idx = int(cid)
        except (ValueError, TypeError):
            print(f"[WARN] unexpected custom_id {cid!r}; skipping.", file=sys.stderr)
            continue
        if not (0 <= idx < len(units)):
            print(f"[WARN] custom_id {idx} out of range; skipping.", file=sys.stderr)
            continue
        if item.success:
            raws[idx] = item.content or ""
        else:
            print(f"[ERR] unit {idx} — {item.error}", file=sys.stderr)

    _write_outputs(category, source, level, units, raws)


# ---------------------------------------------------------------------------
# Language-descriptor benchmark — post-process a built benchmark file, appending
# a "Language: <langs>." hint to every natural-language parameter description.
# ---------------------------------------------------------------------------

def _entry_languages(entry: dict[str, Any]) -> str:
    """The languages a value may appear in for this entry: English + its locale.

    English is the canonical/base language of the source data; the entry's own
    locale (from its `locale` field) is the translation target. An English-only
    entry yields just "English".
    """
    locale_code = entry.get("locale", "en")
    try:
        locale = get_locale(locale_code)
    except ValueError:
        return "English"
    return "English" if locale.code == "en" else f"English, {locale.name}"


def add_language_descriptors_file(source: Path, suffix: str = "langdesc") -> Path:
    """Read a built benchmark .jsonl, append language descriptors, write a new file.

    The output is written next to the source as `<stem>_<suffix><ext>` (e.g.
    he_translatable_full.jsonl -> he_translatable_full_lang_desc.jsonl), preserving
    input order. Returns the output path.
    """
    if not source.exists():
        sys.exit(f"ERROR: {source} not found.")

    entries = load_jsonl(source)
    total_modified = 0
    tagged_entries = 0
    for entry in entries:
        n = add_language_descriptors_to_entry(entry, _entry_languages(entry))
        total_modified += n
        if n:
            tagged_entries += 1

    out = source.with_name(f"{source.stem}_{suffix}{source.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[done] {len(entries)} entries -> {out}")
    print(f"       {total_modified} parameter description(s) tagged "
          f"across {tagged_entries} entrie(s).")
    return out


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
