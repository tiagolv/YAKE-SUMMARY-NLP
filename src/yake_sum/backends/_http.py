"""HTTP helper with bounded retry/backoff shared by network backends."""

from __future__ import annotations

import time
from typing import Any, Callable

import requests

from .base import BackendConnectionError, BackendError, BackendTimeoutError

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    max_retries: int = 2,
    backoff: float = 1.0,
    post: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    service: str = "LLM backend",
    troubleshooting: str = "",
) -> dict[str, Any]:
    """POST JSON and return the decoded body.

    Retries only connection errors, timeouts and transient HTTP statuses
    (``RETRYABLE_STATUS``) with exponential backoff, at most ``max_retries`` times.
    Everything else (e.g. 400/401/404) fails immediately. The raised exception
    carries the service, URL and troubleshooting advice.
    """
    post = post or requests.post
    attempts = max_retries + 1
    last_exc: Exception | None = None
    kind = "connection"

    for attempt in range(attempts):
        try:
            resp = post(url, json=payload, headers=headers, timeout=timeout)
        except requests.Timeout as exc:
            last_exc, kind = exc, "timeout"
        except requests.ConnectionError as exc:
            last_exc, kind = exc, "connection"
        except requests.RequestException as exc:
            raise BackendError(f"{service} request failed: {exc}") from exc
        else:
            status = getattr(resp, "status_code", 200)
            if status in RETRYABLE_STATUS:
                last_exc, kind = BackendError(f"HTTP {status}"), "http"
            elif status >= 400:
                detail = (getattr(resp, "text", "") or "")[:300]
                raise BackendError(
                    f"{service} returned HTTP {status} for {url}. {detail}".strip()
                    + (f" {troubleshooting}" if troubleshooting else "")
                )
            else:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise BackendError(f"{service} returned invalid JSON from {url}.") from exc
        if attempt < attempts - 1:
            sleep(backoff * (2**attempt))

    msg = f"{service} at {url} failed after {attempts} attempt(s): {last_exc}."
    if troubleshooting:
        msg += f" {troubleshooting}"
    if kind == "timeout":
        raise BackendTimeoutError(msg) from last_exc
    raise BackendConnectionError(msg) from last_exc
