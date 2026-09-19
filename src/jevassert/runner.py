"""Record predictions against a live endpoint, and replay them offline.

``record`` calls Jev once per case and writes one JSONL line per case.
``load_predictions`` reads that file back for ``check`` — CI runs on the
recording, so tests are deterministic, free and rate-limit free. Re-record
when the pack changes or when the model version is bumped.

Operational flags:

- ``resume_from`` — skip cases that already have a successful record (retry
  only the errored ones); the merged output keeps pack order.
- ``rpm`` — requests-per-minute pacing for the early-access rate limit.
- ``shuffle_options`` — seed for a robustness pass that permutes Choice option
  order (Score levels are ordinal and are never shuffled).
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .client import JevClient, JevError
from .packs import Case, Pack


def record(
    pack: Pack,
    client: JevClient,
    concurrency: int = 8,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    model: str | None = None,
    resume_from: dict[str, dict[str, Any]] | None = None,
    rpm: int = 0,
    shuffle_options: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every case in ``pack``; returns records in case order.

    Auth failures (401/403, missing key) abort immediately instead of burning
    one failed request per case.
    """
    if not client.api_key:
        raise JevError("TYPESAFE_API_KEY is not set (env var or api_key=)")

    selected: tuple[Case, ...] = pack.cases[:limit] if limit else pack.cases
    api_questions = pack.to_api_questions()
    if shuffle_options is not None:
        api_questions = shuffle_choice_options(api_questions, shuffle_options)
    api_model = model or pack.record_model
    limiter = _RateLimiter(rpm) if rpm and rpm > 0 else None

    existing = resume_from or {}
    done = {case_id for case_id, item in existing.items() if not item.get("error")}
    pending = [case for case in selected if case.id not in done]
    completed = 0

    def evaluate(case: Case) -> dict[str, Any]:
        nonlocal completed
        if limiter:
            limiter.wait()
        try:
            response, latency_ms = client.system_one(case.state, api_questions, api_model)
        except JevError as exc:
            if exc.is_auth_error:
                raise
            result = {
                "case_id": case.id,
                "model": None,
                "latency_ms": None,
                "usage": None,
                "answers": None,
                "error": str(exc),
            }
        else:
            result = {
                "case_id": case.id,
                "model": response.get("model"),
                "latency_ms": round(latency_ms, 1),
                "usage": response.get("usage"),
                "answers": response.get("answers"),
                "error": None,
            }
        completed += 1
        if progress:
            progress(completed, len(pending))
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        fresh = list(pool.map(evaluate, pending))

    by_id = {item["case_id"]: item for item in existing.values()}
    by_id.update({item["case_id"]: item for item in fresh})
    return [by_id[case.id] for case in selected if case.id in by_id]


def shuffle_choice_options(questions: dict[str, Any], seed: int) -> dict[str, Any]:
    """Permute Choice option order deterministically (Score levels stay ordered)."""
    rng = random.Random(seed)
    shuffled: dict[str, Any] = {}
    for qid, question in questions.items():
        if question.get("type") == "choice" and isinstance(question.get("criteria"), dict):
            items = list(question["criteria"].items())
            rng.shuffle(items)
            question = {**question, "criteria": dict(items)}
        shuffled[qid] = question
    return shuffled


class _RateLimiter:
    """Simple global pacer: at most ``rpm`` request starts per minute."""

    def __init__(self, rpm: int) -> None:
        self.min_interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.min_interval
        if delay:
            time.sleep(delay)


def write_predictions(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a predictions JSONL file, keyed by case id."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"predictions file not found: {path}")
    predictions: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{lineno}: case_id is required")
        if case_id in predictions:
            raise ValueError(f"{path}:{lineno}: duplicate case_id '{case_id}'")
        predictions[case_id] = record
    return predictions
