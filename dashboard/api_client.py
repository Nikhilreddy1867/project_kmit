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
* :class:`ApiRejected`     - 422; the submission itself was refused (unacknowledged
  security warning, denied file type, missing target column, ineligible waiver).
  This is a *normal outcome of the model-intake flow*, not a fault, and it carries
  the API's ``issues`` list so the UI can show every problem at once instead of
  making the user resubmit to discover the next one.

Write surface
-------------
The dashboard was read-only through Phase 8 and remains read-only over the
reference case. The model-intake flow (:meth:`GovernanceApiClient.validate_upload`,
:meth:`create_audit`, :meth:`evaluate_gates`, :meth:`create_waiver`,
:meth:`revoke_waiver`) adds POSTs, and those are the only writes. They reach
``/api/onboarding`` and ``/api/gates`` only, where the API confines every write to
``runtime/``. Nothing here can address the committed audit artefacts, and there is
no POST method for any ``/api/models``, ``/api/governance``, ``/api/agents`` or
``/api/registry`` path.
"""

from __future__ import annotations

import json as _json
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 10.0

#: Creating an audit runs inference and permutation importance server-side, so it is
#: allowed far longer than a read before the client gives up on it.
AUDIT_TIMEOUT = 300.0

#: The only prefixes this client will POST to. Asserted in :meth:`_post`, so a wrong
#: path in the UI layer fails here rather than becoming a write attempt against the
#: committed reference-case artefacts.
WRITE_SCOPED_PREFIXES = ("/api/onboarding", "/api/gates")


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


class ApiRejected(ApiError):
    """
    422 - the API understood the submission and refused it.

    Distinct from :class:`ApiServerError` because it is not a malfunction: an
    unacknowledged security warning, a ``.pkl`` upload, a target column that is not
    in the CSV and a waiver against the Release Gate are all *correct* refusals, and
    the UI should present them as things the user can fix rather than as breakage.

    ``issues`` carries the API's full validation list so every problem is shown at
    once. ``code`` is the stable machine-readable reason (e.g. ``denied_file_type``).
    """

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        code: str | None = None,
        issues: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message, hint)
        self.code = code
        self.issues = issues or []


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class GovernanceApiClient:
    """
    Thin wrapper over the governance API.

    Read-only over the reference case; the model-intake methods POST to
    ``/api/onboarding`` and ``/api/gates``, which the API confines to ``runtime/``.
    """

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

    def _post(
        self,
        path: str,
        *,
        json: Any = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """
        POST to one of the two write-scoped prefixes.

        The path is asserted rather than merely documented: a bug elsewhere in the
        dashboard must not be able to turn into a write attempt against the reference
        case. ``timeout`` is separate from the GET default because creating an audit
        runs inference and permutation importance, which legitimately takes longer
        than any read.
        """
        if not path.startswith(WRITE_SCOPED_PREFIXES):
            raise ApiError(
                f"Refusing to POST to {path}: the dashboard may only write via "
                + " or ".join(WRITE_SCOPED_PREFIXES) + ".",
                hint="Every other API surface is read-only. This is a client-side "
                "guard, in addition to the API's own restriction of writes to "
                "runtime/.",
            )

        url = f"{self.base_url}{path}"
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        body = None if data is None else {k: v for k, v in data.items() if v is not None}
        try:
            response = httpx.post(
                url,
                json=json,
                data=body,
                files=files,
                params=clean,
                timeout=timeout or self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise ApiUnavailable(
                f"The API did not finish this request within "
                f"{(timeout or self.timeout):g}s.",
                hint="Creating an audit runs inference and permutation importance. A "
                "large dataset may need a longer timeout, or fewer rows.",
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiUnavailable(
                f"Could not reach the API at {self.base_url}.",
                hint="Start it in a separate PowerShell window:\n\n"
                "    cd C:\\Users\\nreddy\\Downloads\\project_KMIT\n"
                "    .\\.venv\\Scripts\\Activate.ps1\n"
                "    uvicorn app.main:app --reload --port 8000",
            ) from exc

        if response.status_code >= 400:
            raise self._error_from(response, url)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiMalformed(
                f"{url} returned a non-JSON response (HTTP {response.status_code}).",
            ) from exc

    @staticmethod
    def _error_from(response: httpx.Response, url: str) -> ApiError:
        """Turn an error response into a typed exception, preserving the API's message."""
        message, hint, available = None, None, None
        code, issues = None, None
        try:
            body = response.json()
            if isinstance(body, dict):
                message = body.get("message") or body.get("detail")
                hint = body.get("hint")
                code = body.get("error")
                raw = body.get("available")
                available = [str(a) for a in raw] if isinstance(raw, list) else None
                raw_issues = body.get("issues")
                if isinstance(raw_issues, list):
                    issues = [i for i in raw_issues if isinstance(i, dict)]
        except ValueError:
            pass

        if response.status_code == 422:
            # FastAPI's own request-validation failures put a list under "detail",
            # which the branch above leaves as None. Say something useful anyway.
            return ApiRejected(
                message or "The API refused this submission (HTTP 422).",
                hint=hint,
                code=code,
                issues=issues,
            )
        message = message or f"HTTP {response.status_code} from {url}."

        if response.status_code == 404:
            return ApiNotFound(message, hint=hint, available=available)
        if response.status_code == 405:
            return ApiServerError(
                "That request used a method this endpoint does not allow.",
                hint="Only /api/onboarding and /api/gates accept POST; every other "
                "endpoint is read-only.",
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
    def registry_runs(
        self, status: str | None = None, run_type: str | None = None
    ) -> dict[str, Any]:
        return self._require(
            self._get("/api/registry/runs", {"status": status, "run_type": run_type}),
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

    # -- Phase 9: model intake (the only write paths) ------------------------ #
    @staticmethod
    def _intake_form(
        target_column: str,
        positive_class: str,
        decision_threshold: float,
        sensitive_columns: list[str] | None,
        security_acknowledged: bool,
        policy_profile_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Build the shared multipart form fields.

        ``sensitive_columns`` is sent as a JSON array; the API also accepts a
        comma-separated list, but a JSON array cannot be ambiguous about a column
        name that itself contains a comma.
        """
        return {
            "target_column": target_column,
            "positive_class": positive_class,
            "decision_threshold": str(decision_threshold),
            "sensitive_columns": _json.dumps(list(sensitive_columns or [])),
            "security_acknowledged": "true" if security_acknowledged else "false",
            "policy_profile_id": policy_profile_id,
        }

    @staticmethod
    def _upload_files(
        model_name: str, model_bytes: bytes, dataset_name: str, dataset_bytes: bytes
    ) -> dict[str, tuple[str, bytes, str]]:
        """
        Package the two uploads for multipart transport.

        The filenames are passed through only as *labels* -- the API generates a UUID
        filename for storage and never uses what arrives here as a path.
        """
        return {
            "model_file": (model_name, model_bytes, "application/octet-stream"),
            "dataset_file": (dataset_name, dataset_bytes, "text/csv"),
        }

    def validate_upload(
        self,
        *,
        model_name: str,
        model_bytes: bytes,
        dataset_name: str,
        dataset_bytes: bytes,
        target_column: str,
        positive_class: str,
        decision_threshold: float = 0.5,
        sensitive_columns: list[str] | None = None,
        security_acknowledged: bool = False,
        policy_profile_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Dry-run the intake checks. Writes the two uploads but creates no audit run.

        A 200 with ``valid: false`` is the normal way a fixable configuration problem
        (wrong target column, feature mismatch) comes back. A 422 -- raised as
        :class:`ApiRejected` -- means the envelope itself was refused.
        """
        return self._require(
            self._post(
                "/api/onboarding/validate",
                files=self._upload_files(
                    model_name, model_bytes, dataset_name, dataset_bytes
                ),
                data=self._intake_form(
                    target_column,
                    positive_class,
                    decision_threshold,
                    sensitive_columns,
                    security_acknowledged,
                    policy_profile_id,
                ),
                timeout=AUDIT_TIMEOUT,
            ),
            ("valid", "issues", "audit_capabilities", "next_step"),
            "POST /api/onboarding/validate",
        )

    def create_audit(
        self,
        *,
        target_column: str,
        positive_class: str,
        model_metadata: dict[str, str],
        decision_threshold: float = 0.5,
        sensitive_columns: list[str] | None = None,
        security_acknowledged: bool = False,
        policy_profile_id: str | None = None,
        upload_id: str | None = None,
        model_name: str | None = None,
        model_bytes: bytes | None = None,
        dataset_name: str | None = None,
        dataset_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """
        Create one audit run under ``runtime/audits/``.

        Either pass ``upload_id`` to reuse files already stored by
        :meth:`validate_upload` -- which is what the dashboard does, so validating
        does not mean uploading twice -- or pass both file payloads.
        """
        data = {
            **self._intake_form(
                target_column,
                positive_class,
                decision_threshold,
                sensitive_columns,
                security_acknowledged,
                policy_profile_id,
            ),
            **{k: v for k, v in model_metadata.items() if v is not None},
            "upload_id": upload_id,
        }
        files = None
        if upload_id is None:
            if not (model_bytes and dataset_bytes):
                raise ApiError(
                    "Pass either an upload_id or both file payloads.",
                    hint="Validate first, then submit the returned upload_id.",
                )
            files = self._upload_files(
                model_name or "model.joblib",
                model_bytes,
                dataset_name or "dataset.csv",
                dataset_bytes,
            )
        return self._require(
            self._post(
                "/api/onboarding/audits",
                data=data,
                files=files,
                timeout=AUDIT_TIMEOUT,
            ),
            ("audit_run_id", "governance_state", "gate_summary", "written_under"),
            "POST /api/onboarding/audits",
        )

    def uploaded_audits(self) -> dict[str, Any]:
        return self._require(
            self._get("/api/onboarding/audits"),
            ("count", "runs"),
            "GET /api/onboarding/audits",
        )

    def uploaded_audit(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}"),
            ("audit_run_id", "model_metadata", "audit_coverage", "governance_state"),
            f"GET /api/onboarding/audits/{audit_run_id}",
        )

    def uploaded_performance(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}/performance"),
            ("audit_run_id", "confusion_matrix", "decision_threshold", "caveats"),
            f"GET /api/onboarding/audits/{audit_run_id}/performance",
        )

    def uploaded_fairness(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}/fairness"),
            ("audit_run_id", "status", "status_detail", "four_fifths_notice"),
            f"GET /api/onboarding/audits/{audit_run_id}/fairness",
        )

    def uploaded_explainability(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}/explainability"),
            ("audit_run_id", "status", "status_detail"),
            f"GET /api/onboarding/audits/{audit_run_id}/explainability",
        )

    def uploaded_governance(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}/governance"),
            ("audit_run_id", "governance_state", "state_meaning", "limitations"),
            f"GET /api/onboarding/audits/{audit_run_id}/governance",
        )

    def uploaded_integrity(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}/integrity"),
            ("integrity_status", "integrity_ok", "artifacts_checked", "artifacts"),
            f"GET /api/onboarding/audits/{audit_run_id}/integrity",
        )

    def uploaded_timeline(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/onboarding/audits/{audit_run_id}/timeline"),
            ("audit_run_id", "count", "events"),
            f"GET /api/onboarding/audits/{audit_run_id}/timeline",
        )

    # -- Phase 9: Governance-as-Code gates ---------------------------------- #
    def policies(self) -> dict[str, Any]:
        return self._require(
            self._get("/api/gates/policies"),
            ("count", "policies"),
            "GET /api/gates/policies",
        )

    def gate_evaluation(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/gates/runs/{audit_run_id}/evaluation"),
            ("audit_run_id", "gates", "controls", "gate_summary"),
            f"GET /api/gates/runs/{audit_run_id}/evaluation",
        )

    def conformity_bundle(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/gates/runs/{audit_run_id}/bundle"),
            ("bundle_id", "evidence", "gate_decisions", "disclaimers"),
            f"GET /api/gates/runs/{audit_run_id}/bundle",
        )

    def traceability(self, audit_run_id: str) -> dict[str, Any]:
        return self._require(
            self._get(f"/api/gates/runs/{audit_run_id}/traceability"),
            ("audit_run_id", "rows", "unresolved_evidence"),
            f"GET /api/gates/runs/{audit_run_id}/traceability",
        )

    def evaluate_gates(
        self, audit_run_id: str, policy_profile_id: str | None = None
    ) -> dict[str, Any]:
        """Re-run the policy over stored evidence. Rewrites four runtime artefacts."""
        return self._require(
            self._post(
                f"/api/gates/runs/{audit_run_id}/evaluate",
                params={"policy_profile_id": policy_profile_id},
                timeout=60.0,
            ),
            ("audit_run_id", "gate_summary", "conformity_bundle_id", "changed"),
            f"POST /api/gates/runs/{audit_run_id}/evaluate",
        )

    def waivers(self, audit_run_id: str) -> list[dict[str, Any]]:
        payload = self._get(f"/api/gates/runs/{audit_run_id}/waivers")
        if not isinstance(payload, list):
            raise ApiMalformed(
                f"GET /api/gates/runs/{audit_run_id}/waivers: expected a JSON array."
            )
        return payload

    def create_waiver(self, audit_run_id: str, waiver: dict[str, Any]) -> dict[str, Any]:
        """
        Record one explicit, time-bounded waiver. Refused for ineligible controls.

        Every field is caller-supplied on purpose: there is no default owner, no
        default expiry and no default rationale, because a waiver with any of those
        invented would not be an accountable human decision.
        """
        return self._require(
            self._post(f"/api/gates/runs/{audit_run_id}/waivers", json=waiver),
            ("waiver_id", "control_id", "status", "expires_at"),
            f"POST /api/gates/runs/{audit_run_id}/waivers",
        )

    def revoke_waiver(self, audit_run_id: str, waiver_id: str) -> dict[str, Any]:
        return self._require(
            self._post(f"/api/gates/runs/{audit_run_id}/waivers/{waiver_id}/revoke"),
            ("waiver_id", "status"),
            f"POST /api/gates/runs/{audit_run_id}/waivers/{waiver_id}/revoke",
        )
