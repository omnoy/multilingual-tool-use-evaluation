"""
remove_benchmark_entries.py — Remove a set of entries from every benchmark file in a folder.

Walks <folder> recursively and, in every *.jsonl file (one entry per line) and every *.json
file whose top level is a list of entries, drops the entries whose id matches one of the
given ids. Everything else is left untouched: other lines are written back byte-for-byte,
and JSON files that are not a list of entries (manifests, caches) are skipped. Hidden files
(names starting with ".") are skipped too.

Matching is exact (multiple_2 never touches multiple_21). An entry matches an id when any of
these equals it:
  * the entry's "id"                      (multiple_2_de)
  * the entry's "source_id"               (multiple_2, on localized entries)
  * the entry's "id" minus "_<locale>"    (multiple_2, when "locale" is set)
So a base id like multiple_2 removes the entry from every language, while a localized id
like multiple_2_de removes it only from the German files.

Usage:
    # Preview what would be removed and from where:
    python scripts/remove_benchmark_entries.py data/benchmarks/bfcl_multiple multiple_2 multiple_7 --dry-run

    # Read ids from a file (one per line, '#' comments allowed) and remove them:
    python scripts/remove_benchmark_entries.py data/benchmarks/bfcl_multiple --ids-file to_remove.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def entry_keys(entry: dict) -> set[str]:
    """All ids under which an entry can be addressed."""
    keys: set[str] = set()
    eid = entry.get("id")
    if isinstance(eid, str):
        keys.add(eid)
        locale = entry.get("locale")
        if isinstance(locale, str) and eid.endswith(f"_{locale}"):
            keys.add(eid[: -len(locale) - 1])
    sid = entry.get("source_id")
    if isinstance(sid, str):
        keys.add(sid)
    return keys


def process_jsonl(path: Path, ids: set[str]) -> tuple[list[str], list[str], int]:
    """Return (kept raw lines, removed entry ids, total entries)."""
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    kept, removed, total = [], [], 0
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            kept.append(line)
            continue
        total += 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: {path}:{lineno} is not valid JSON ({e}); aborting.")
        if isinstance(entry, dict) and entry_keys(entry) & ids:
            removed.append(str(entry.get("id")))
        else:
            kept.append(line)
    return kept, removed, total


def process_json_list(path: Path, ids: set[str]) -> tuple[list | None, list[str], int]:
    """Return (kept entries, removed entry ids, total), or (None, [], 0) if not a list of entries."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None, [], 0
    if not isinstance(data, list) or not data or not all(isinstance(e, dict) and "id" in e for e in data):
        return None, [], 0
    kept, removed = [], []
    for entry in data:
        if entry_keys(entry) & ids:
            removed.append(str(entry.get("id")))
        else:
            kept.append(entry)
    return kept, removed, len(data)


def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_ids(args: argparse.Namespace, parser: argparse.ArgumentParser) -> set[str]:
    ids = {i.strip() for i in args.ids if i.strip()}
    if args.ids_file:
        for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            ids.update(tok for tok in line.replace(",", " ").split() if tok)
    if not ids:
        parser.error("no entry ids given (pass them as arguments or via --ids-file).")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove entries (by id) from every benchmark file in a folder.",
    )
    parser.add_argument("folder", help="Benchmark folder, e.g. data/benchmarks/bfcl_multiple.")
    parser.add_argument("ids", nargs="*", help="Entry ids to remove, e.g. multiple_2 multiple_7.")
    parser.add_argument("--ids-file", help="File with ids to remove (one per line or comma/space separated).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be removed and from where; change nothing.")
    args = parser.parse_args()

    root = Path(f'../{args.folder}').resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    ids = read_ids(args, parser)

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in {".jsonl", ".json"}
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )

    verb = "would remove" if args.dry_run else "removed"
    matched_ids: set[str] = set()
    total_removed = files_changed = 0
    skipped: list[Path] = []

    for path in files:
        rel = path.relative_to(root)
        if path.suffix == ".jsonl":
            kept, removed, total = process_jsonl(path, ids)
            new_text = "".join(kept) if removed else None
        else:
            kept, removed, total = process_json_list(path, ids)
            if kept is None:
                skipped.append(rel)
                continue
            new_text = json.dumps(kept, ensure_ascii=False, indent=2) + "\n" if removed else None

        for key in ids:
            if any(key == r or r.startswith(f"{key}_") for r in removed):
                matched_ids.add(key)
        if not removed:
            print(f"  {rel}: nothing to remove ({total} entries)")
            continue

        files_changed += 1
        total_removed += len(removed)
        print(f"  {rel}: {verb} {len(removed)} of {total} -> {total - len(removed)} left")
        print(f"      {', '.join(removed)}")
        if not args.dry_run:
            atomic_write(path, new_text)

    if skipped:
        print("\nSkipped (not a list of entries): " + ", ".join(str(s) for s in skipped))

    unmatched = sorted(ids - matched_ids)
    if unmatched:
        print(f"\nWarning: no entries found for {len(unmatched)} id(s): {', '.join(unmatched)}")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}{verb.capitalize()} {total_removed} entr"
          f"{'y' if total_removed == 1 else 'ies'} across {files_changed} of {len(files) - len(skipped)} file(s).")
    if args.dry_run and total_removed:
        print("Nothing was changed. Run again without --dry-run to apply.")


if __name__ == "__main__":
    sys.exit(main())
