"""
Azure OpenAI Batch API helper — file-based submit / poll / retrieve.

Replaces the previous Anthropic Message Batches (langasync) workflow. The two-step
UX is preserved: submit returns a batch id; retrieve fetches results later.

Azure batch flow:
  1. Build a JSONL file, one line per request:
       {"custom_id": "...", "method": "POST", "url": "/chat/completions", "body": {...}}
  2. Upload it with purpose="batch"  -> input_file_id
  3. Create a batch over that file   -> batch id
  4. Poll until status == "completed"
  5. Download the output (and error) files and map results back by custom_id.

`custom_id` is the caller's key back to its own bookkeeping (this project uses the
batch-item index as a string, matching the old manifest format).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

# Statuses Azure reports for a batch job.
_TERMINAL_OK = {"completed"}
_TERMINAL_BAD = {"failed", "expired", "cancelled", "canceled"}

CHAT_COMPLETIONS_URL = "/chat/completions"


@dataclass
class BatchItemResult:
    custom_id: str
    success: bool
    content: str | None = None
    error: str | None = None


def build_chat_request(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Wrap a chat.completions request body into a Batch API JSONL line.

    `body` is what build_chat_params() produces (model, messages, temperature/...).
    """
    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": CHAT_COMPLETIONS_URL,
        "body": body,
    }


def submit_batch(
    client,
    requests: list[dict[str, Any]],
    *,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
):
    """Upload the requests as a JSONL file and create a batch. Returns the batch object."""
    if not requests:
        raise ValueError("submit_batch called with no requests.")

    buf = io.BytesIO()
    for req in requests:
        buf.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
    buf.seek(0)

    input_file = client.files.create(file=("batch_input.jsonl", buf), purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=CHAT_COMPLETIONS_URL,
        completion_window=completion_window,
        metadata=metadata or {},
    )
    return batch


def get_batch(client, batch_id: str):
    """Retrieve the current batch object (status, counts, output/error file ids)."""
    return client.batches.retrieve(batch_id)


def status_line(batch) -> str:
    """A short human-readable progress string for a batch object."""
    rc = getattr(batch, "request_counts", None)
    if rc is None:
        return f"status={batch.status}"
    total = getattr(rc, "total", 0)
    completed = getattr(rc, "completed", 0)
    failed = getattr(rc, "failed", 0)
    return f"status={batch.status} ({completed} done, {failed} failed / {total} total)"


def is_finished(batch) -> bool:
    return batch.status in _TERMINAL_OK or batch.status in _TERMINAL_BAD


def _iter_jsonl_file(client, file_id: str):
    """Yield parsed JSON objects from an uploaded output/error file."""
    if not file_id:
        return
    text = client.files.content(file_id).text
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def collect_results(client, batch_id: str) -> dict[str, BatchItemResult]:
    """Download a completed batch's output/error files and index results by custom_id.

    For each request, extracts the assistant message text on success, or a short error
    string on failure. Raises RuntimeError if the batch is not finished yet.
    """
    batch = get_batch(client, batch_id)
    if not is_finished(batch):
        raise RuntimeError(
            f"Batch {batch_id!r} is not finished ({status_line(batch)}). Try again later."
        )

    results: dict[str, BatchItemResult] = {}

    for obj in _iter_jsonl_file(client, getattr(batch, "output_file_id", None)):
        cid = str(obj.get("custom_id"))
        response = obj.get("response") or {}
        err = obj.get("error")
        status_code = response.get("status_code")
        body = response.get("body") or {}
        if err or (status_code is not None and status_code >= 400):
            msg = _error_message(err) or f"HTTP {status_code}"
            results[cid] = BatchItemResult(cid, success=False, error=msg)
            continue
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        results[cid] = BatchItemResult(cid, success=content is not None,
                                       content=content,
                                       error=None if content is not None else "empty response")

    # The error file lists requests that never produced a response.
    for obj in _iter_jsonl_file(client, getattr(batch, "error_file_id", None)):
        cid = str(obj.get("custom_id"))
        if cid in results:
            continue
        results[cid] = BatchItemResult(cid, success=False,
                                       error=_error_message(obj.get("error")) or "batch error")

    return results


def _error_message(err: Any) -> str | None:
    if not err:
        return None
    if isinstance(err, dict):
        return err.get("message") or err.get("code") or json.dumps(err)
    return str(err)
