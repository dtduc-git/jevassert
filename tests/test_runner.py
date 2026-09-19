from pathlib import Path

import pytest
from helpers import write_pack

from jevassert.client import JevClient, JevError
from jevassert.packs import load_pack
from jevassert.runner import (
    _RateLimiter,
    load_predictions,
    record,
    shuffle_choice_options,
    write_predictions,
)

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Urgent?"}}
CASES = [
    {"id": f"c-{index}", "state": {"text": f"state {index}"}, "expect": {"urgent": True}}
    for index in range(3)
]


def api_ok() -> tuple:
    return (
        200,
        {
            "model": "jev-1.13.0",
            "answers": {"urgent": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
        None,
    )


class FakeTransport:
    def __init__(self, script: list[tuple]) -> None:
        self.script = script
        self.calls: list[tuple] = []

    def __call__(self, method: str, url: str, headers: dict, json_body: dict) -> tuple:
        self.calls.append((method, url, headers, json_body))
        return self.script[min(len(self.calls) - 1, len(self.script) - 1)]


def make_pack(tmp_path: Path, **overrides):
    return load_pack(write_pack(tmp_path / "test-pack", QUESTIONS, CASES, **overrides))


def make_client(transport: FakeTransport) -> JevClient:
    return JevClient(
        api_key="test-key", base_url="https://example.test", transport=transport, backoff_seconds=0
    )


def test_record_preserves_case_order_and_payload(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    transport = FakeTransport([api_ok()])
    records = record(pack, make_client(transport), concurrency=3)

    assert [item["case_id"] for item in records] == ["c-0", "c-1", "c-2"]
    assert all(item["error"] is None for item in records)
    assert records[0]["model"] == "jev-1.13.0"
    assert records[0]["latency_ms"] is not None

    method, url, headers, payload = transport.calls[0]
    assert method == "POST"
    assert url == "https://example.test/v1/systemone"
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "jev-latest"
    assert payload["questions"] == pack.to_api_questions()


def test_record_uses_pinned_tested_and_model_override(tmp_path: Path) -> None:
    pack = make_pack(tmp_path, tested="jev-1.13.0")
    transport = FakeTransport([api_ok()])
    record(pack, make_client(transport), concurrency=1)
    assert transport.calls[0][3]["model"] == "jev-1.13.0"

    transport = FakeTransport([api_ok()])
    record(pack, make_client(transport), concurrency=1, model="jev-preview")
    assert transport.calls[0][3]["model"] == "jev-preview"


def test_retries_retryable_status_then_succeeds(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    transport = FakeTransport([(429, {"error": "slow down"}, 0.0), api_ok()])
    records = record(pack, make_client(transport), concurrency=1)
    assert len(transport.calls) == 4  # 429 + retry for c-0, then c-1, c-2
    assert all(item["error"] is None for item in records)


def test_non_retryable_error_is_recorded_per_case(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    transport = FakeTransport([(400, {"error": "bad request"}, None)])
    records = record(pack, make_client(transport), concurrency=1)
    assert len(transport.calls) == 3  # one call per case, no retries
    assert all("HTTP 400" in item["error"] for item in records)
    assert all(item["answers"] is None for item in records)


def test_missing_api_key_fails_fast(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    transport = FakeTransport([api_ok()])
    client = JevClient(api_key="", transport=transport)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        record(pack, client, concurrency=1)
    assert transport.calls == []


def test_auth_error_fails_fast(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    transport = FakeTransport([(401, {"error": "invalid key"}, None)])
    with pytest.raises(JevError) as excinfo:
        record(pack, make_client(transport), concurrency=1)
    assert excinfo.value.status == 401
    assert len(transport.calls) == 1  # not one failed request per case


def test_resume_skips_successful_cases_and_keeps_order(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    existing = {
        "c-0": {
            "case_id": "c-0",
            "model": "jev-1.13.0",
            "latency_ms": 1.0,
            "usage": None,
            "answers": {"urgent": {"type": "noul", "noul": 1.0}},
            "error": None,
        }
    }
    transport = FakeTransport([api_ok()])
    records = record(pack, make_client(transport), concurrency=1, resume_from=existing)
    assert [item["case_id"] for item in records] == ["c-0", "c-1", "c-2"]
    assert records[0]["answers"]["urgent"]["noul"] == 1.0
    assert len(transport.calls) == 2


def test_shuffle_options_permutes_choices_only(tmp_path: Path) -> None:
    questions = {
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Team?",
            "options": {"a": "a", "b": "b", "c": "c", "d": "d", "unknown": "can't tell"},
        },
    }
    case = {"id": "c-1", "state": {"text": "x"}, "expect": {"urgent": True, "team": "a"}}
    pack = load_pack(write_pack(tmp_path / "test-pack", questions, [case]))
    base = pack.to_api_questions()

    seed = None
    for candidate in range(20):
        shuffled = shuffle_choice_options(base, candidate)
        if list(shuffled["team"]["criteria"]) != list(base["team"]["criteria"]):
            seed = candidate
            break
    assert seed is not None, "no seed permuted the option order"

    again = shuffle_choice_options(base, seed)
    assert list(again["team"]["criteria"]) == list(
        shuffle_choice_options(base, seed)["team"]["criteria"]
    )
    assert set(again["team"]["criteria"]) == set(base["team"]["criteria"])
    assert again["urgent"] == base["urgent"]


def test_rate_limiter_interval() -> None:
    limiter = _RateLimiter(600)
    assert limiter.min_interval == pytest.approx(0.1)
    limiter.wait()  # first request is never delayed


def test_client_requires_key() -> None:
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient(api_key="", transport=FakeTransport([api_ok()])).system_one("x", QUESTIONS)


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    records = record(make_pack(tmp_path), make_client(FakeTransport([api_ok()])), concurrency=2)
    write_predictions(path, records)
    loaded = load_predictions(path)
    assert sorted(loaded) == ["c-0", "c-1", "c-2"]


def test_load_predictions_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text('{"case_id": "c-0"}\n{"case_id": "c-0"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_predictions(path)
