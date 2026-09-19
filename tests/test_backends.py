"""Adapter backends: LLM recording via system-one-adapter, prices, model reporting."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from helpers import noul, record, write_pack, write_predictions

from jevassert import cli
from jevassert.backends import AdapterClient
from jevassert.client import JevError
from jevassert.metrics import compute
from jevassert.packs import load_pack

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}


class _ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _install_fake_adapter(monkeypatch, response: Any = None, error: Exception | None = None):
    """Minimal stand-ins for the system-one-adapter public surface."""

    class FakeRetryPolicy:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeSystemOneAdapterClient:
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.options = kwargs

        def system_one(self, state, questions, model):
            if error is not None:
                raise error
            return response

        def close(self) -> None:
            FakeSystemOneAdapterClient.closed = True

    class FakeOpenAIProvider:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

    package = types.ModuleType("system_one_adapter")
    package.RetryPolicy = FakeRetryPolicy  # type: ignore[attr-defined]
    package.SystemOneAdapterClient = FakeSystemOneAdapterClient  # type: ignore[attr-defined]
    providers = types.ModuleType("system_one_adapter.providers")
    openai_module = types.ModuleType("system_one_adapter.providers.openai")
    openai_module.OpenAIProvider = FakeOpenAIProvider  # type: ignore[attr-defined]
    anthropic_module = types.ModuleType("system_one_adapter.providers.anthropic")

    class FakeAnthropicProvider:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name

    anthropic_module.AnthropicProvider = FakeAnthropicProvider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "system_one_adapter", package)
    monkeypatch.setitem(sys.modules, "system_one_adapter.providers", providers)
    monkeypatch.setitem(sys.modules, "system_one_adapter.providers.openai", openai_module)
    monkeypatch.setitem(sys.modules, "system_one_adapter.providers.anthropic", anthropic_module)


class _FakeResponse:
    def model_dump(self) -> dict[str, Any]:
        return {
            "model": "qwen2.5:7b",
            "answers": {"urgent": noul(0.9)},
            "usage": {"input_tokens": 42, "output_tokens": 7},
        }


def test_adapter_client_maps_response(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch, response=_FakeResponse())
    client = AdapterClient("openai", "qwen2.5:7b", base_url="http://localhost:11434/v1")
    body, latency_ms = client.system_one({"text": "hi"}, QUESTIONS)
    assert body["model"] == "qwen2.5:7b"
    assert body["usage"]["input_tokens"] == 42
    assert latency_ms >= 0
    client.close()


def test_adapter_auth_error_fails_fast(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch, error=_ProviderError("invalid api key provided", status=401))
    client = AdapterClient("openai", "gpt-4o-mini", base_url="http://localhost:11434/v1")
    with pytest.raises(JevError) as excinfo:
        client.system_one({"text": "hi"}, QUESTIONS)
    assert excinfo.value.is_auth_error


def test_adapter_message_heuristic_sets_auth_status(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch, error=_ProviderError("Authentication failed"))
    client = AdapterClient("openai", "gpt-4o-mini", base_url="http://localhost:11434/v1")
    with pytest.raises(JevError) as excinfo:
        client.system_one({"text": "hi"}, QUESTIONS)
    assert excinfo.value.status == 401


def test_adapter_anthropic_requires_key(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(JevError, match="ANTHROPIC_API_KEY"):
        AdapterClient("anthropic", "claude-haiku-4-5")


def test_bedrock_uses_anthropic_bedrock_client(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch)
    captured: dict[str, Any] = {}

    class FakeAnthropicBedrock:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class FakeAnthropicModule(types.ModuleType):
        AnthropicBedrock = FakeAnthropicBedrock

    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicModule("anthropic"))
    client = AdapterClient("bedrock", "global.anthropic.claude-sonnet-4-6")
    assert captured == {"max_retries": 0}
    assert client.model_name == "global.anthropic.claude-sonnet-4-6"


def test_bedrock_credentials_error_fails_fast(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch, error=_ProviderError("Unable to locate credentials"))

    class FakeAnthropicBedrock:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class FakeAnthropicModule(types.ModuleType):
        AnthropicBedrock = FakeAnthropicBedrock

    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicModule("anthropic"))
    client = AdapterClient("bedrock", "global.anthropic.claude-sonnet-4-6")
    with pytest.raises(JevError) as excinfo:
        client.system_one({"text": "hi"}, QUESTIONS)
    assert excinfo.value.is_auth_error


def test_record_backend_requires_model(tmp_path: Path, capsys) -> None:
    pack = write_pack(
        tmp_path / "bk-pack",
        QUESTIONS,
        [{"id": "c-1", "state": {"text": "a"}, "expect": {"urgent": True}}],
    )
    code = cli.main(["record", str(pack), "--backend", "openai"])
    assert code == 2
    assert "requires --model" in capsys.readouterr().err


def test_record_uses_adapter_backend(tmp_path: Path, monkeypatch) -> None:
    pack = write_pack(
        tmp_path / "bk-pack",
        QUESTIONS,
        [{"id": "c-1", "state": {"text": "a"}, "expect": {"urgent": True}}],
    )
    captured: dict[str, Any] = {}

    class FakeAdapter:
        api_key = "unused"

        def __init__(self, backend, model, base_url=None):
            captured.update(backend=backend, model=model, base_url=base_url)

        def system_one(self, state, questions, model):
            return (
                {
                    "model": "qwen2.5:7b",
                    "answers": {"urgent": noul(0.9)},
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
                3.0,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "AdapterClient", FakeAdapter)
    out = tmp_path / "bk.jsonl"
    code = cli.main(
        [
            "record",
            str(pack),
            "-o",
            str(out),
            "--backend",
            "openai",
            "--model",
            "qwen2.5:7b",
            "--base-url",
            "http://localhost:11434/v1",
        ]
    )
    assert code == 0
    assert captured == {
        "backend": "openai",
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
    }
    assert "qwen2.5:7b" in out.read_text()


def test_compute_prices_include_output_tokens(tmp_path: Path) -> None:
    pack = load_pack(
        write_pack(
            tmp_path / "pricing-pack",
            QUESTIONS,
            [{"id": "c-1", "state": {"text": "a"}, "expect": {"urgent": True}}],
        )
    )
    predictions = {
        "c-1": record(
            "c-1",
            {"urgent": noul(0.9)},
            usage={"input_tokens": 1_000_000, "output_tokens": 500_000},
        )
    }
    default = compute(pack, predictions, bootstrap=0)
    assert default.total_cost_usd == pytest.approx(0.042)
    priced = compute(
        pack,
        predictions,
        bootstrap=0,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert priced.total_cost_usd == pytest.approx(2.0)
    free = compute(pack, predictions, bootstrap=0, input_usd_per_mtok=0.0)
    assert free.total_cost_usd == 0.0


def test_check_json_reports_recorded_models_and_pricing(tmp_path: Path, capsys) -> None:
    pack = write_pack(
        tmp_path / "rm-pack",
        QUESTIONS,
        [{"id": "c-1", "state": {"text": "a"}, "expect": {"urgent": True}}],
        tested="jev-1.13.0",
    )
    predictions = write_predictions(
        tmp_path / "rm.jsonl",
        [record("c-1", {"urgent": noul(0.9)}, model="qwen2.5:7b")],
    )
    code = cli.main(["check", str(pack), "-p", str(predictions), "--json", "--bootstrap", "0"])
    assert code == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["recorded_models"] == ["qwen2.5:7b"]
    assert payload["pricing"] == {"input_usd_per_mtok": 0.042, "output_usd_per_mtok": 0.0}
