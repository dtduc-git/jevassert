"""Shared builders for tests: minimal packs, cases and canned predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_pack(
    directory: Path,
    questions: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    state_fields: list[str] | None = None,
    thresholds: dict[str, Any] | None = None,
    gates: dict[str, Any] | None = None,
    **overrides: Any,
) -> Path:
    """Write a spec v0 pack; the directory name becomes the pack id."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "spec": 0,
        "id": directory.name,
        "version": "0.1.0",
        "license": "CC0-1.0",
        "tested": None,
        "description": "Test pack.",
        "state": {"description": "Test state.", "fields": state_fields or ["text"]},
        "questions": questions,
    }
    if thresholds:
        meta["thresholds"] = thresholds
    meta.update(overrides)
    (directory / "pack.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    if gates:
        (directory / "gates.yaml").write_text(yaml.safe_dump(gates), encoding="utf-8")
    lines = [json.dumps(case, ensure_ascii=False) for case in cases]
    (directory / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def write_predictions(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def record(
    case_id: str,
    answers: dict[str, Any] | None,
    error: str | None = None,
    usage: dict[str, int] | None = None,
    latency_ms: float | None = 100.0,
    model: str | None = "jev-1.13.0",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "model": model,
        "latency_ms": latency_ms,
        "usage": usage,
        "answers": answers,
        "error": error,
    }


def noul(probability: float) -> dict[str, Any]:
    return {"type": "noul", "noul": probability}


def choice(selected: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
    }


def score(value: float, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "type": "score",
        "score": value,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
    }
