"""HTTP transport for the TypeSafe System One endpoint.

One small client with retries and an injectable transport so tests run
without network access and so any Jev-compatible endpoint (TypeSafe API,
gateway, local replica) can be used via ``base_url``.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from . import __version__

DEFAULT_BASE_URL = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}

# transport(method, url, headers, json_body) -> (status, body, retry_after_seconds)
Transport = Callable[[str, str, dict[str, str], dict[str, Any]], tuple[int, Any, float | None]]


class JevError(RuntimeError):
    """A System One call failed (after retries, if retryable)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def is_auth_error(self) -> bool:
        return self.status in (401, 403)


class _ResponseLike(Protocol):  # pragma: no cover - typing helper
    status_code: int
    headers: Any

    def json(self) -> Any: ...


def _httpx_transport(client: httpx.Client) -> Transport:
    def transport(
        method: str, url: str, headers: dict[str, str], json_body: dict[str, Any]
    ) -> tuple[int, Any, float | None]:
        response = client.request(method, url, headers=headers, json=json_body)
        retry_after: float | None = None
        raw = response.headers.get("retry-after")
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = None
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return response.status_code, body, retry_after

    return transport


class JevClient:
    """Thin client for ``POST {base_url}/v1/systemone``.

    Reads ``TYPESAFE_API_KEY`` and ``TYPESAFE_BASE_URL`` from the environment
    when not given explicitly. The API key is never logged or written to disk.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Transport | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = (
            base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._http = None if transport else httpx.Client(timeout=timeout)
        self._transport: Transport = transport or _httpx_transport(self._http)  # type: ignore[arg-type]

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def system_one(
        self, state: Any, questions: dict[str, Any], model: str = "jev-latest"
    ) -> tuple[dict[str, Any], float]:
        """Evaluate ``questions`` against ``state``; returns (response, latency_ms)."""
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is not set (env var or api_key=)")
        payload = {"state": state, "model": model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"jevassert/{__version__}",
        }
        url = f"{self.base_url}{ENDPOINT}"

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                status, body, retry_after = self._transport("POST", url, headers, payload)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise JevError(f"request failed after {attempt + 1} attempts: {exc}") from exc
                self._sleep(attempt, None)
                continue
            latency_ms = (time.perf_counter() - started) * 1000

            if status == 200 and isinstance(body, dict):
                return body, latency_ms
            if status in RETRYABLE_STATUS and attempt < self.max_retries:
                self._sleep(attempt, retry_after)
                continue
            raise JevError(f"HTTP {status}: {_excerpt(body)}", status=status)

        raise JevError("unreachable")  # pragma: no cover

    def _sleep(self, attempt: int, retry_after: float | None) -> None:
        delay = self.backoff_seconds * (2**attempt) + random.uniform(0, 0.25)
        if retry_after:
            delay = max(delay, retry_after)
        time.sleep(delay)


def _excerpt(body: Any, limit: int = 300) -> str:
    text = body if isinstance(body, str) else str(body)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")
