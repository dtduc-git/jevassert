"""Recording backends: the TypeSafe API, or any LLM via ``system-one-adapter``.

``record --backend openai|anthropic`` runs the same pack, the same runner and
the same metrics against a general-purpose LLM through the official adapter
(https://github.com/typesafe-ai/system-one-adapter-python). Used to record
baseline evidence for the jev-packs benchmark, and to compare a candidate
endpoint against Jev on identical ground truth.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .client import JevError


class AdapterClient:
    """``JevClient``-shaped client backed by an LLM through system-one-adapter.

    ``backend`` selects the adapter provider: ``openai`` covers any
    OpenAI-compatible endpoint (Ollama, vLLM, gateways; set ``base_url``),
    ``anthropic`` calls Claude directly. Recording never touches the TypeSafe
    API, so no TypeSafe key is needed.
    """

    def __init__(self, backend: str, model: str, base_url: str | None = None) -> None:
        try:
            from system_one_adapter import RetryPolicy, SystemOneAdapterClient
        except ImportError as exc:
            raise JevError(
                "adapter backends need system-one-adapter: pip install 'jevassert[adapter]'"
            ) from exc
        self.backend = backend
        self.model_name = model
        self.api_key = "unused"  # runner.record only checks presence
        self._provider = _build_provider(backend, model, base_url)
        self._client = SystemOneAdapterClient(
            structured_outputs=True,
            llm_answer_mode="probabilities",
            normalize_probabilities=True,
            n_retry_malformed_structure=2,
            retry=RetryPolicy(max_retries=3),
        )

    def system_one(
        self, state: Any, questions: dict[str, Any], model: str = ""
    ) -> tuple[dict[str, Any], float]:
        """Evaluate questions against the configured LLM; returns (response, latency_ms)."""
        started = time.perf_counter()
        try:
            response = self._client.system_one(state, questions, model=self._provider)
        except Exception as exc:  # adapter raises SDK error types
            raise _as_jev_error(exc) from exc
        return response.model_dump(), (time.perf_counter() - started) * 1000

    def close(self) -> None:
        self._client.close()


def _build_provider(backend: str, model: str, base_url: str | None) -> Any:
    if backend == "openai":
        from system_one_adapter.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model, base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "not-needed"
        )
    if backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise JevError("ANTHROPIC_API_KEY is not set (required for --backend anthropic)")
        from system_one_adapter.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model)
    if backend == "bedrock":
        return _build_bedrock_provider(model)
    raise ValueError(f"unknown backend {backend!r} (expected openai, anthropic or bedrock)")


def _build_bedrock_provider(model: str) -> Any:
    """Claude through AWS Bedrock: the adapter's AnthropicProvider over AnthropicBedrock.

    Credentials come from the standard AWS chain (``AWS_PROFILE``/``AWS_REGION``
    or instance roles). Newer models need an inference profile ID, e.g.
    ``global.anthropic.claude-haiku-4-5-20251001-v1:0``.
    """
    import anthropic
    from system_one_adapter.providers.anthropic import AnthropicProvider

    class BedrockProvider(AnthropicProvider):  # type: ignore[misc, valid-type]
        def __init__(self, model_name: str) -> None:
            super().__init__(model_name)
            self._client = anthropic.AnthropicBedrock(max_retries=0)

    return BedrockProvider(model)


def _as_jev_error(exc: Exception) -> JevError:
    """Map adapter/SDK failures onto ``JevError`` so fail-fast paths keep working."""
    status = getattr(exc, "status", None)
    if status is None:
        text = str(exc).lower()
        if any(
            marker in text
            for marker in (
                "api key",
                "authentication",
                "unauthorized",
                "credential",
                "access denied",
                "accessdenied",
            )
        ):
            status = 401
    return JevError(str(exc), status=status)
