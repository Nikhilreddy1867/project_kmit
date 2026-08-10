"""
api_client.py
=============
HTTP client for the AI Governance Platform API (Phase 5).

This is the dashboard's **only** data source. The dashboard never opens a CSV or
Markdown file, never imports anything from ``src/`` or ``app/``, and never
recomputes a metric: every number it shows arrives over HTTP from
``/api/...`` exactly as the API served it.

The client is deliberately free of Streamlit imports so it stays testable on its
own; caching lives in the Streamlit layer.

Failure model
-------------
Network and protocol problems are translated into four typed exceptions so the UI
can render a specific, actionable message instead of a stack trace:

* :class:`ApiUnavailable`  - the server is not reachable (not started, wrong port).
* :class:`ApiNotFound`     - 404; carries the ``available`` alternatives the API
  returned, so the UI can say what *is* available.
* :class:`ApiServerError`  - 5xx; the API's own ``message``/``hint`` are preserved
  (e.g. "artefact missing - run python src/train.py").
* :class:`ApiMalformed`    - a 2xx response that is not JSON, or is missing keys
  the dashboard needs. Better to say so than to render a blank panel.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 10.0


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """Base class for all API access failures."""

    def __init__(self, message: str, hint: str | None = None):
        self.message = message
        self.hint = hint
        super().__init__(message)


class ApiUnavailable(ApiError):
    """The API could not be reached at all."""


class ApiNotFound(ApiError):
    """404 - the requested model or attribute does not exist in that artefact."""

    def __init__(self, message: str, hint: str | None = None,
                 available: list[str] | None = None):
        super().__init__(message, hint)
        self.available = available or []


class ApiServerError(ApiError):
    """5xx - the API is running but cannot serve this resource."""

    def __init__(self, message: str, hint: str | None = None, status_code: int = 500):
        super().__init__(message, hint)
        self.status_code = status_code


class ApiMalformed(ApiError):
    """A successful response whose shape the dashboard cannot use."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class GovernanceApiClient:
    """Thin, read-only wrapper over the governance API's GET endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    # -- plumbing ----------------------------------------------------------- #
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            response = httpx.get(url, params=clean, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise ApiUnavailable(
                f"The API at {self.base_url} did not respond within {self.timeout:g}s.",
                hint="The server may be starting up or overloaded. Retry, or restart it.",
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiUnavailable(
                f"Could not reach the API at {self.base_url}.",
                hint=(
                    "Start it in a separate PowerShell window:\n\n"
                    "    cd C:\\Users\\nreddy\\Downloads\\project_KMIT\n"
                    "    .\\.venv\\Scripts\\Activate.ps1\n"
                    "    uvicorn app.main:app --reload --port 8000"
                ),
            ) from exc

        if response.status_code >= 400:
            raise self._error_from(response, url)

        try:
            return response.json()
        except ValueError as exc:
            raise ApiMalformed(
                f"{url} returned a non-JSON response (HTTP {response.status_code}).",
                hint="Check that the base URL points at the governance API, not another service.",
            ) from exc

    @staticmethod
    def _error_from(response: httpx.Response, url: str) -> ApiError:
        """Turn an error response into a typed exception, preserving the API's message."""
        message, hint, available = None, None, None
        try:
            body = response.json()
            if isinstance(body, dict):
                message = body.get("message") or body.get("detail")
                hint = body.get("hint")
                raw = body.get("available")
                available = [str(a) for a in raw] if isinstance(raw, list) else None
        except ValueError:
            pass
        message = message or f"HTTP {response.status_code} from {url}."

        if response.status_code == 404:
            return ApiNotFound(message, hint=hint, available=available)
        if response.status_code == 405:
            return ApiServerError(
                "That request used a method the read-only API does not allow.",
                hint="The governance API exposes GET endpoints only.",
                status_code=405,
            )
        return ApiServerError(message, hint=hint, status_code=response.status_code)

    @staticmethod
    def _require(payload: Any, keys: tuple[str, ...], context: str) -> dict[str, Any]:
        """Validate that a payload is a dict containing the keys the UI relies on."""
        if not isinstance(payload, dict):
            raise ApiMalformed(
                f"{context}: expected a JSON object, got {type(payload).__name__}."
            )
        missing = [k for k in keys if k not in payload]
        if missing:
            raise ApiMalformed(
                f"{context}: response is missing expected field(s): {', '.join(missing)}.",
                hint="The API version may not match this dashboard.",
            )
        return payload

    # -- endpoints ---------------------------------------------------------- #
    def health(self) -> dict[str, Any]:
        return self._require(
            self._get("/health"),
            ("status", "service", "version", "artifacts_present", "artifacts_expected"),
            "GET /health",
        )

    def models(self) -> dict[str, Any]:
        return self._require(
            self._get("/api/models"), ("models", "count"), "GET /api/models"
        )

    def performance(self, model_name: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/models/{model_name}/performance"),
            ("model_name", "metrics", "confusion_matrix"),
            f"GET /api/models/{model_name}/performance",
        )

    def fairness(self, model_name: str, attribute: str | None = None) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/models/{model_name}/fairness", {"attribute": attribute}),
            ("model_name", "groups", "summary", "interpretation"),
            f"GET /api/models/{model_name}/fairness",
        )

    def explainability(self, model_name: str, top_n: int | None = None) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/models/{model_name}/explainability", {"top_n": top_n}),
            ("model_name", "global_importance", "proxy_assessment", "caveats"),
            f"GET /api/models/{model_name}/explainability",
        )

    def decision(self, include_markdown: bool = False) -> dict[str, Any]:
        return self._require(
            self._get("/api/governance/decision", {"include_markdown": include_markdown}),
            ("research_use", "real_world_deployment", "risk_profile", "disclaimer"),
            "GET /api/governance/decision",
        )

    def risks(
        self,
        overall_risk: str | None = None,
        category: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        return self._require(
            self._get(
                "/api/governance/risks",
                {"overall_risk": overall_risk, "category": category, "status": status},
            ),
            ("risks", "count", "counts_by_overall_risk", "rating_scale"),
            "GET /api/governance/risks",
        )

    def model_card(self, sections_only: bool = False) -> dict[str, Any]:
        return self._require(
            self._get("/api/governance/model-card", {"sections_only": sections_only}),
            ("sections", "source"),
            "GET /api/governance/model-card",
        )

    # -- Phase 7: deterministic governance agents --------------------------- #
    def agents(self) -> dict[str, Any]:
        return self._require(
            self._get("/api/agents"),
            ("agents", "count", "disclaimer"),
            "GET /api/agents",
        )

    def agent(self, agent_name: str, model_name: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/agents/{agent_name}", {"model_name": model_name}),
            ("agent_name", "status", "findings"),
            f"GET /api/agents/{agent_name}",
        )

    def agent_review(self, model_name: str) -> dict[str, Any]:
        return self._require(
            self._get("/api/agents/review", {"model_name": model_name}),
            ("agents", "findings_total", "preserved_decision", "overall_recommendation"),
            "GET /api/agents/review",
        )

    # -- Phase 8: governance audit registry --------------------------------- #
    def registry_runs(self, status: str | None = None) -> dict[str, Any]:
        return self._require(
            self._get("/api/registry/runs", {"status": status}),
            ("count", "runs", "database"),
            "GET /api/registry/runs",
        )

    def registry_run(self, run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/registry/runs/{run_id}"),
            ("run_id", "status", "audit_coverage", "artifacts"),
            f"GET /api/registry/runs/{run_id}",
        )

    def registry_integrity(self, run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/registry/runs/{run_id}/integrity"),
            ("integrity_status", "integrity_ok", "artifacts_checked"),
            f"GET /api/registry/runs/{run_id}/integrity",
        )

    def registry_timeline(self, run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/registry/runs/{run_id}/timeline"),
            ("run_id", "events", "count"),
            f"GET /api/registry/runs/{run_id}/timeline",
        )
