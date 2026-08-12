"""
main.py
=======
FastAPI application for the AI Governance Platform MVP (Phase 5).

A **read-only** HTTP surface over the completed Phase 1-4 audit artefacts. The
API never trains, never re-scores and never writes: it serves the committed
evidence in `results/` and nothing else. Only GET routes are defined, so any
mutating verb returns 405 by construction.

Run locally (Windows PowerShell, venv active):

    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for Swagger UI.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    Path as PathParam,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.agents import orchestrator
from app.agents.schemas import AgentListResponse, AgentReport, GovernanceReview
from app.gates import policy_engine
from app.gates import service as gates_service
from app.gates.schemas import (
    ConformityBundle,
    GateEvaluation,
    GateEvaluationResponse,
    PolicyListResponse,
    TraceabilityMatrix,
    Waiver,
    WaiverIn,
)
from app.onboarding import audit_service, model_validator, security
from app.onboarding import runtime_store as store
from app.onboarding import upload_service
from app.onboarding.schemas import (
    AuditConfigIn,
    AuditCreatedResponse,
    ModelMetadataIn,
    UploadedAuditDetail,
    UploadedAuditListResponse,
    UploadedExplainability,
    UploadedFairness,
    UploadedGovernance,
    UploadedIntegrityResponse,
    UploadedPerformance,
    UploadedTimelineResponse,
    ValidationResponse,
)
from app.registry import service as registry_service
from app.registry.schemas import (
    AuditRunDetail,
    IntegrityResponse,
    RunListResponse,
    TimelineResponse,
)
from app.schemas.models import (
    DecisionResponse,
    ErrorDetail,
    ExplainabilityResponse,
    FairnessResponse,
    HealthResponse,
    ModelCardResponse,
    ModelListResponse,
    PerformanceResponse,
    RiskRegisterResponse,
)
from app.services import artifact_reader as reader
from app.services import governance_service as svc

SERVICE_NAME = "ai-governance-platform"
VERSION = "0.9.0"

DESCRIPTION = """
Governance API for the **Multi-Agent AI Audit and Trust Framework (MAAT)**.

Two clearly separated surfaces:

1. **The built-in reference case** -- the completed Adult / Census Income audit.
   Served **read-only** and **verbatim** from `results/`. No endpoint can train,
   re-score, recompute or modify it.
2. **User-submitted audits** -- a trusted local `.joblib` model plus a labelled CSV,
   audited on request. Everything they produce is written **only** under `runtime/`.

The `POST` endpoints under `/api/onboarding` and `/api/gates` are the only writes in
the platform, and they can write nowhere except `runtime/`. Every other endpoint is a
read.

**All governance output here is deterministic decision-support evidence for human
review.** It is not a legal compliance finding, not proof of discrimination or
causation, and never an authorisation to deploy.

## The reference case (`/api/models`, `/api/governance`, `/api/agents`)

Every number is read **verbatim** from the audit artefacts in `results/`. This part of
the API performs no model training, no re-scoring and no metric recomputation, so it
cannot drift from the committed evidence.

### Phases exposed
* **Phase 1** - baseline model comparison and held-out performance
* **Phase 2** - fairness audit by `sex` and `race`
* **Phase 3** - explainability audit (permutation importance + exact TreeSHAP)
* **Phase 4** - governance risk register, model card and decision record

### Reading these numbers responsibly
* All rates use the default **0.5** decision threshold, which was never tuned.
  Only `roc_auc` is threshold-free.
* The majority-class accuracy floor is **0.7607**, so accuracy alone says little.
* Fairness metrics measure **outcome differences**. They are not findings of
  discrimination and identify no causal effect.
* Explainability output is **associational**. Low importance for `sex` / `race`
  is *not* evidence of fairness -- see `/api/models/xgboost/explainability`.

### Governance status
The audited model is **conditionally approved for research and educational use
only** and **blocked from real-world deployment**. See
`/api/governance/decision`. Nothing under `/api/onboarding` or `/api/gates` can
change that decision.

## User-submitted audits (`/api/onboarding`, `/api/gates`)

Upload a trusted local `.joblib` binary classifier and a labelled CSV, choose the
target column, positive class, sensitive columns and decision threshold, and the
platform runs performance, fairness, explainability, risk, policy-gate and evidence
integrity assessment over it.

> **Joblib files may execute arbitrary code. Upload only models from trusted sources.
> This local academic prototype must not accept untrusted model files in production.**

The warning must be acknowledged explicitly before a model is deserialised. A
production implementation should sandbox or isolate model loading and prefer safer
formats such as ONNX or skops.

Uploaded runs are **separate records**. They never supersede, alter or contribute to
the built-in reference case, and no gate result approves an uploaded model for
real-world deployment.
"""

TAGS_METADATA = [
    {"name": "system", "description": "Service health and artefact availability."},
    {
        "name": "models",
        "description": "Phase 1 model inventory, held-out performance, and the "
        "Phase 2/3 audit results for each model.",
    },
    {
        "name": "governance",
        "description": "Phase 4 decision record, risk register and model card.",
    },
    {
        "name": "agents",
        "description": "Phase 7 deterministic governance agents. Rule-based reporting "
        "over the existing evidence -- not autonomous decision-makers. They never "
        "train, write, recalculate a metric, or alter the governance decision.",
    },
    {
        "name": "registry",
        "description": "Phase 8 governance audit registry and evidence integrity. "
        "Records the audit as a content-addressed review run with SHA-256 checksums "
        "of every referenced artefact. The registry records evidence; it does not "
        "make decisions.",
    },
    {
        "name": "onboarding",
        "description": "Submit a trusted local .joblib binary classifier and a "
        "labelled CSV for a new governance audit, and read the resulting run. The "
        "POST endpoints here are intentional writes and write only under runtime/. "
        "Joblib files may execute arbitrary code: the security warning must be "
        "acknowledged before a model is deserialised. Nothing here can approve a "
        "model for real-world deployment.",
    },
    {
        "name": "gates",
        "description": "Governance-as-Code policy gates (Data, Training, Validation, "
        "Release, Operations), waivers, the Conformity Bundle and the clause-to-"
        "artefact traceability matrix. Gate results are configured research-policy "
        "outcomes -- PASS/WAIVE/BLOCK/NOT_EVALUATED -- not legal conclusions. Release "
        "and Operations are never automatically passed.",
    },
]

app = FastAPI(
    title="MAAT -- Multi-Agent AI Audit and Trust Framework API",
    summary="The Adult Income reference audit, plus governance audits of "
    "user-submitted local models.",
    description=DESCRIPTION,
    version=VERSION,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "MAAT -- Multi-Agent AI Audit and Trust Framework"},
    license_info={"name": "Academic / research use only"},
)

# --------------------------------------------------------------------------- #
# CORS -- for a local dashboard (Streamlit 8501, Vite 5173, CRA 3000, etc.)
# Any localhost/127.0.0.1 port is allowed; credentials are NOT enabled and the
# wildcard origin is deliberately avoided, since a browser would otherwise be
# able to read this API from any site the user visits.
#
# POST is allowed because model intake needs it. The write surface is bounded not by
# the allowed verb list but by the code behind it: every POST route delegates to
# app.onboarding or app.gates, which write only under runtime/ and refuse to touch
# data/, models/, predictions/ or results/ (app.onboarding.runtime_store.
# assert_not_evidence). No POST route exists for the reference case at all.
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Data-Source", "X-Write-Scope"],
)

#: Path prefixes served from the user-submission side of the platform.
_RUNTIME_PREFIXES = ("/api/onboarding", "/api/gates")


@app.middleware("http")
async def _tag_data_source(request: Request, call_next):
    """
    Declare, per response, which evidence store it came from and what it may write.

    The two headers are deliberately separate. A caller that used to rely on
    ``X-Data-Source: results/`` keeps getting exactly that for the reference case, and
    a response from the upload side never claims to be committed evidence.
    """
    response = await call_next(request)
    if request.url.path.startswith(_RUNTIME_PREFIXES):
        response.headers["X-Data-Source"] = "runtime/ (user-submitted audit runs)"
        response.headers["X-Write-Scope"] = (
            "runtime/ only" if request.method == "POST" else "none (read)"
        )
    else:
        response.headers["X-Data-Source"] = "results/ (read-only audit artefacts)"
        response.headers["X-Write-Scope"] = "none (read)"
    return response


# --------------------------------------------------------------------------- #
# Exception handlers -> meaningful HTTP responses
# --------------------------------------------------------------------------- #
@app.exception_handler(reader.ModelNotFoundError)
async def _model_not_found(request: Request, exc: reader.ModelNotFoundError):
    """404: the caller asked for a name that does not exist in that artefact."""
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            error="not_found",
            message=str(exc),
            hint=f"Use one of the available values for {exc.scope}.",
            available=exc.available,
        ).model_dump(),
    )


@app.exception_handler(reader.ArtifactMissingError)
async def _artifact_missing(request: Request, exc: reader.ArtifactMissingError):
    """503: the server is correctly configured but the evidence is not on disk."""
    return JSONResponse(
        status_code=503,
        content=ErrorDetail(
            error="artifact_unavailable",
            message=f"Required audit artefact '{exc.key}' is missing at {exc.path}.",
            hint=exc.how_to_fix,
        ).model_dump(),
    )


@app.exception_handler(orchestrator.AgentNotFoundError)
async def _agent_not_found(request: Request, exc: orchestrator.AgentNotFoundError):
    """404: unknown agent name."""
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            error="not_found",
            message=str(exc),
            hint="Call GET /api/agents to list the available agents.",
            available=exc.available,
        ).model_dump(),
    )


@app.exception_handler(registry_service.RunNotFoundError)
async def _run_not_found(request: Request, exc: registry_service.RunNotFoundError):
    """404: unknown audit-run id."""
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            error="not_found",
            message=str(exc),
            hint="Call GET /api/registry/runs to list registered runs.",
            available=exc.available,
        ).model_dump(),
    )


@app.exception_handler(FileNotFoundError)
async def _registry_uninitialised(request: Request, exc: FileNotFoundError):
    """
    503: the registry database has not been created yet.

    Deliberately not a 404 or an empty list -- an uninitialised registry is a
    setup step the caller must perform, and serving `[]` would read as
    "this audit was never registered".
    """
    return JSONResponse(
        status_code=503,
        content=ErrorDetail(
            error="registry_uninitialised",
            message=str(exc),
            hint="Create it with: python -m app.registry.cli register",
        ).model_dump(),
    )


@app.exception_handler(reader.ArtifactMalformedError)
async def _artifact_malformed(request: Request, exc: reader.ArtifactMalformedError):
    """500: the artefact exists but cannot be trusted, so refuse to guess."""
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            error="artifact_malformed",
            message=str(exc),
            hint="Regenerate the artefact by re-running the relevant audit script.",
        ).model_dump(),
    )


# --------------------------------------------------------------------------- #
# Exception handlers for the model-intake and gate surfaces
#
# Each failure gets its own status and its own remedy, because "the run does not
# exist", "the run exists but that artefact was never produced" and "the submission
# was rejected" are three different problems for the caller. Collapsing them into one
# 400 would leave a user guessing which.
# --------------------------------------------------------------------------- #
@app.exception_handler(store.AuditRunNotFound)
async def _audit_run_not_found(request: Request, exc: store.AuditRunNotFound):
    """404: no uploaded audit run with that id exists under runtime/audits/."""
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            error="not_found",
            message=str(exc),
            hint="Call GET /api/onboarding/audits to list uploaded audit runs.",
            available=exc.available,
        ).model_dump(),
    )


@app.exception_handler(store.AuditArtifactMissing)
async def _audit_artifact_missing(request: Request, exc: store.AuditArtifactMissing):
    """
    404: the run exists but never produced that artefact.

    Not a 503 -- the server is fine and re-running nothing will change it. The
    artefact is genuinely absent for this run, which is itself a finding.
    """
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            error="artifact_not_produced",
            message=str(exc),
            hint="GET /api/onboarding/audits/{audit_run_id} lists the artefacts this "
            "run actually produced, and its audit_coverage states which capabilities "
            "were unavailable.",
        ).model_dump(),
    )


@app.exception_handler(security.UploadRejected)
async def _upload_rejected(request: Request, exc: security.UploadRejected):
    """
    422: the upload envelope was refused before anything was stored or deserialised.

    Covers a wrong extension, an oversized file, a URL in place of a file and the
    missing security acknowledgement.
    """
    return JSONResponse(
        status_code=422,
        content=ErrorDetail(
            error=exc.code,
            message=exc.message,
            hint=exc.hint,
        ).model_dump(),
    )


@app.exception_handler(model_validator.ModelValidationError)
async def _model_invalid(request: Request, exc: model_validator.ModelValidationError):
    """
    422: the model file loaded but cannot be audited as a binary classifier.

    The model is never replaced, wrapped or substituted, and no probability is
    invented for it -- the submission is refused instead.
    """
    return JSONResponse(
        status_code=422,
        content=ErrorDetail(
            error=exc.code,
            message=exc.message,
            hint=exc.hint,
        ).model_dump(),
    )


@app.exception_handler(audit_service.AuditRejected)
async def _audit_rejected(request: Request, exc: audit_service.AuditRejected):
    """
    422 with *every* issue, not just the first.

    One round trip should tell the submitter everything they need to fix. Nothing was
    written: a rejected submission leaves no partial audit run behind.
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "submission_rejected",
            "message": "The submission was rejected and no audit run was created. "
            "Nothing was written under runtime/audits/.",
            "hint": "Fix every error-severity issue below and submit again.",
            "issues": [issue.model_dump() for issue in exc.issues],
        },
    )


@app.exception_handler(gates_service.WaiverRejected)
async def _waiver_rejected(request: Request, exc: gates_service.WaiverRejected):
    """422: the waiver would not be an accountable decision, so it was not recorded."""
    return JSONResponse(
        status_code=422,
        content=ErrorDetail(
            error=exc.code,
            message=exc.message,
            hint=exc.hint,
        ).model_dump(),
    )


@app.exception_handler(policy_engine.PolicyNotFound)
async def _policy_not_found(request: Request, exc: policy_engine.PolicyNotFound):
    """404: unknown policy profile id."""
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            error="not_found",
            message=str(exc),
            hint="Call GET /api/gates/policies to list the available policy profiles.",
            available=getattr(exc, "available", []),
        ).model_dump(),
    )


_ERRORS = {
    404: {"model": ErrorDetail, "description": "Unknown model or attribute name"},
    500: {"model": ErrorDetail, "description": "Audit artefact is malformed"},
    503: {"model": ErrorDetail, "description": "Audit artefact missing on disk"},
}

_INTAKE_ERRORS = {
    404: {"model": ErrorDetail, "description": "Unknown audit run, or an artefact "
          "this run did not produce"},
    422: {"model": ErrorDetail, "description": "Submission, model or waiver rejected. "
          "Nothing was written."},
}

_MODEL_PARAM = PathParam(
    description="Model name as recorded in results/model_metrics.csv "
    "(xgboost, random_forest, logistic_regression).",
    examples=["xgboost"],
)


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Service health and artefact availability",
    description="Reports liveness plus the presence, size and modification time of "
    "every audit artefact. Returns 200 with `status: degraded` when artefacts are "
    "missing, so a dashboard can show what is unavailable instead of erroring.",
)
async def health() -> HealthResponse:
    return HealthResponse(**svc.build_health(SERVICE_NAME, VERSION))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@app.get(
    "/api/models",
    response_model=ModelListResponse,
    tags=["models"],
    responses=_ERRORS,
    summary="Evaluated models and their Phase 1 metrics",
    description="All models with held-out Phase 1 metrics, plus flags for which "
    "have a Phase 2 fairness audit and a Phase 3 explainability audit. Values are "
    "read verbatim from results/model_metrics.csv.",
)
async def list_models() -> ModelListResponse:
    return ModelListResponse(**svc.build_model_list())


@app.get(
    "/api/models/{model_name}/performance",
    response_model=PerformanceResponse,
    tags=["models"],
    responses=_ERRORS,
    summary="Held-out performance for one model",
    description="Phase 1 metrics, confusion matrix and error analysis for a single "
    "model, with the caveats that govern their interpretation.",
)
async def model_performance(model_name: str = _MODEL_PARAM) -> PerformanceResponse:
    return PerformanceResponse(**svc.build_performance(model_name))


@app.get(
    "/api/models/{model_name}/fairness",
    response_model=FairnessResponse,
    tags=["models"],
    responses=_ERRORS,
    summary="Phase 2 fairness audit for one model",
    description="Per-group metrics (selection rate, TPR, FPR, precision, recall, F1) "
    "and disparity measures relative to the largest group, by `sex` and `race`. "
    "Includes an explicit statement of what these metrics do and do not establish.",
)
async def model_fairness(
    model_name: str = _MODEL_PARAM,
    attribute: str | None = Query(
        default=None,
        description="Restrict to one sensitive attribute (sex or race).",
        examples=["sex"],
    ),
) -> FairnessResponse:
    return FairnessResponse(**svc.build_fairness(model_name, attribute=attribute))


@app.get(
    "/api/models/{model_name}/explainability",
    response_model=ExplainabilityResponse,
    tags=["models"],
    responses=_ERRORS,
    summary="Phase 3 explainability audit for one model",
    description="Permutation importance over the 14 original human-readable features "
    "(not one-hot fragments), the proxy assessment, and -- for the primary model -- "
    "exact TreeSHAP local explanations for 5 held-out cases. Available for "
    "`xgboost` and `logistic_regression`; `random_forest` was not audited for "
    "explainability and returns 404 rather than fabricated numbers.",
)
async def model_explainability(
    model_name: str = _MODEL_PARAM,
    top_n: int | None = Query(
        default=None, ge=1, le=50, description="Limit the importance list.", examples=[5]
    ),
) -> ExplainabilityResponse:
    return ExplainabilityResponse(**svc.build_explainability(model_name, top_n=top_n))


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #
@app.get(
    "/api/governance/decision",
    response_model=DecisionResponse,
    tags=["governance"],
    responses=_ERRORS,
    summary="Deployment decision record",
    description="The Phase 4 decision: conditionally approved for research use only, "
    "blocked from real-world deployment. Grounds and conditions are parsed from the "
    "committed governance_summary.md so that document remains the source of truth.",
)
async def governance_decision(
    include_markdown: bool = Query(
        default=False, description="Include the full governance summary Markdown."
    ),
) -> DecisionResponse:
    return DecisionResponse(**svc.build_decision(include_markdown=include_markdown))


@app.get(
    "/api/governance/risks",
    response_model=RiskRegisterResponse,
    tags=["governance"],
    responses=_ERRORS,
    summary="Governance risk register",
    description="The 12-entry risk register with rating scale and framing. "
    "Optionally filtered by rating, category substring or status substring.",
)
async def governance_risks(
    overall_risk: str | None = Query(
        default=None,
        description="Exact match on rating (Critical, High, Medium, Low).",
        examples=["Critical"],
    ),
    category: str | None = Query(
        default=None, description="Case-insensitive substring match.", examples=["Fairness"]
    ),
    status: str | None = Query(
        default=None, description="Case-insensitive substring match.", examples=["blocking"]
    ),
) -> RiskRegisterResponse:
    return RiskRegisterResponse(
        **svc.build_risks(overall_risk=overall_risk, category=category, status=status)
    )


@app.get(
    "/api/governance/model-card",
    response_model=ModelCardResponse,
    tags=["governance"],
    responses=_ERRORS,
    summary="Model card content",
    description="The full Phase 4 model card as Markdown, plus its section headings. "
    "Pass `sections_only=true` for the table of contents without the body.",
)
async def governance_model_card(
    sections_only: bool = Query(
        default=False, description="Return section headings only, omitting content."
    ),
) -> ModelCardResponse:
    return ModelCardResponse(**svc.build_model_card(sections_only=sections_only))


# --------------------------------------------------------------------------- #
# Agents (Phase 7)
#
# ROUTE ORDER MATTERS: `/api/agents/review` must be declared BEFORE
# `/api/agents/{agent_name}`, or FastAPI would match "review" as an agent name and
# the review endpoint would become unreachable. Covered by a test.
# --------------------------------------------------------------------------- #
_AGENT_DISCLAIMER_NOTE = (
    "Deterministic, rule-based reporting agents -- NOT autonomous decision-makers. "
    "They read existing evidence, never train or write, never recalculate a metric, "
    "and never alter the governance decision."
)

_MODEL_QUERY = Query(
    default="xgboost",
    description="Model to review, as listed by GET /api/models.",
    examples=["xgboost"],
)


@app.get(
    "/api/agents",
    response_model=AgentListResponse,
    tags=["agents"],
    responses=_ERRORS,
    summary="List the governance agents",
    description="Self-description of each deterministic agent: what it reads, what it "
    "reports, and what it must never do. " + _AGENT_DISCLAIMER_NOTE,
)
async def agents_list() -> AgentListResponse:
    return orchestrator.list_agents()


@app.get(
    "/api/agents/review",
    response_model=GovernanceReview,
    tags=["agents"],
    responses=_ERRORS,
    summary="Orchestrated multi-agent governance review",
    description="Runs all four agents in a fixed order and returns one structured "
    "review: every finding with its severity, evidence source, limitations and "
    "recommended action, plus severity counts. The `overall_recommendation` is the "
    "committed governance decision **restated verbatim** — the agents do not produce "
    "it and cannot change it. " + _AGENT_DISCLAIMER_NOTE,
)
async def agents_review(model_name: str = _MODEL_QUERY) -> GovernanceReview:
    return orchestrator.run_review(model_name)


@app.get(
    "/api/agents/{agent_name}",
    response_model=AgentReport,
    tags=["agents"],
    responses=_ERRORS,
    summary="Run one governance agent",
    description="Runs a single agent for one model. Agent names: `performance`, "
    "`fairness`, `explainability`, `risk`. Where the underlying audit does not cover "
    "the model, the agent returns `status: unavailable` with an explicit "
    "'no evidence exists' finding rather than estimating anything. "
    + _AGENT_DISCLAIMER_NOTE,
)
async def agents_run_one(
    agent_name: str = PathParam(
        description="One of: performance, fairness, explainability, risk.",
        examples=["fairness"],
    ),
    model_name: str = _MODEL_QUERY,
) -> AgentReport:
    return orchestrator.run_agent(agent_name, model_name)


# --------------------------------------------------------------------------- #
# Registry (Phase 8)
#
# Read-only over the registry database. Creating or refreshing a run is a
# deliberate CLI action (`python -m app.registry.cli register`), not an HTTP call,
# so the API surface stays read-only.
# --------------------------------------------------------------------------- #
_RUN_ID_PARAM = PathParam(
    description="Content-addressed audit-run id, as listed by GET /api/registry/runs.",
    examples=["run-047074fcb8e0380a"],
)


@app.get(
    "/api/registry/runs",
    response_model=RunListResponse,
    tags=["registry"],
    responses=_ERRORS,
    summary="List registered audit runs",
    description="Every governance review run in the local registry, newest first, "
    "with its status (active / superseded / archived), run type (reference_case / "
    "uploaded_model), dataset and model version, artefact count and recorded decision. "
    "`active_run_id` is the active **reference-case** run: registering uploaded audits "
    "never moves it. Returns 503 if the registry has not been created yet.",
)
async def registry_runs(
    status: str | None = Query(
        default=None,
        description="Filter by run status.",
        examples=["active"],
    ),
    run_type: str | None = Query(
        default=None,
        description="Filter by run type: 'reference_case' for the built-in Adult "
        "Income audit, 'uploaded_model' for user submissions.",
        examples=["uploaded_model"],
    ),
) -> RunListResponse:
    return RunListResponse(
        **registry_service.list_runs(status=status, run_type=run_type)
    )


@app.get(
    "/api/registry/runs/{run_id}",
    response_model=AuditRunDetail,
    tags=["registry"],
    responses=_ERRORS,
    summary="Full detail for one audit run",
    description="Dataset and model identity, performance summary, recorded governance "
    "decision, blocking risk ids, audit coverage, and the complete artefact manifest "
    "with the SHA-256 checksum captured at registration time.",
)
async def registry_run_detail(run_id: str = _RUN_ID_PARAM) -> AuditRunDetail:
    return AuditRunDetail(**registry_service.get_run(run_id))


@app.get(
    "/api/registry/runs/{run_id}/integrity",
    response_model=IntegrityResponse,
    tags=["registry"],
    responses=_ERRORS,
    summary="Verify a run's evidence integrity",
    description="**Recomputes** the SHA-256 of every registered artefact right now "
    "and classifies each as verified, missing or changed, with an overall status. "
    "A 'changed' result means this run's conclusions no longer describe the files on "
    "disk — it is not by itself evidence of wrongdoing, since re-running an audit "
    "legitimately rewrites its outputs.",
)
async def registry_run_integrity(run_id: str = _RUN_ID_PARAM) -> IntegrityResponse:
    return IntegrityResponse(**registry_service.check_integrity(run_id))


@app.get(
    "/api/registry/runs/{run_id}/timeline",
    response_model=TimelineResponse,
    tags=["registry"],
    responses=_ERRORS,
    summary="Chronological history for a run",
    description="Merges the append-only registry event log (registration, refresh, "
    "status change, integrity check) with evidence events derived from artefact "
    "modification times, ordered oldest first. The log is append-only, so repeated "
    "integrity checks accumulate; `limit` returns the most recent events while "
    "`total_events` always reports the full count.",
)
async def registry_run_timeline(
    run_id: str = _RUN_ID_PARAM,
    limit: int = Query(
        default=200,
        ge=1,
        le=5000,
        description="Return at most this many of the most recent events.",
    ),
) -> TimelineResponse:
    return TimelineResponse(**registry_service.get_timeline(run_id, limit=limit))


# --------------------------------------------------------------------------- #
# Onboarding -- model intake and uploaded audit runs
#
# The two POST routes below are the only intentional writes in the platform. They
# reach the filesystem exclusively through app.onboarding.runtime_store, which
# refuses any path outside runtime/ and never replaces an existing file. There is
# deliberately no endpoint that can write to data/, models/, predictions/ or
# results/, and none that can alter the Adult Income reference case.
# --------------------------------------------------------------------------- #
_AUDIT_ID_PARAM = PathParam(
    description="Uploaded audit-run id, as listed by GET /api/onboarding/audits.",
    examples=["audit-6958a0305750505b"],
)

_INTAKE_NOTICE = (
    "All governance output from this endpoint is deterministic decision-support "
    "evidence for human review. It is not a legal compliance finding, does not prove "
    "discrimination or causation, and is never an authorisation to deploy."
)

_SECURITY_BLOCK = (
    "\n\n**Joblib files may execute arbitrary code. Upload only models from trusted "
    "sources. This local academic prototype must not accept untrusted model files in "
    "production.** `security_acknowledged=true` is required before the model is "
    "deserialised; without it the request is refused and nothing is loaded. A "
    "production implementation should sandbox or isolate model loading and prefer "
    "safer formats such as ONNX or skops."
)


def _parse_sensitive_columns(raw: str | None) -> list[str]:
    """
    Read the sensitive-column selection from a multipart form field.

    Accepts a JSON array or a comma-separated list, because a browser form and a
    scripted client naturally send different things and neither should have to know
    which one this endpoint prefers. Blank means *none selected*, which the fairness
    audit reports as ``not_provided_by_user`` -- not as a pass.
    """
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


async def _store_submission(
    model_file: UploadFile,
    dataset_file: UploadFile,
    security_acknowledged: bool,
) -> dict[str, Any]:
    """
    Persist both uploads under a fresh upload id, or raise before anything is written.

    The model arrives as opaque bytes and stays that way here: extension, size and the
    acknowledgement are checked first, and deserialisation happens later, in the
    validation step. The original filenames are kept only as display labels -- the
    stored paths are generated UUID filenames.
    """
    security.require_acknowledgement(security_acknowledged)
    return upload_service.store_uploads(
        await model_file.read(),
        model_file.filename or "model.joblib",
        await dataset_file.read(),
        dataset_file.filename or "dataset.csv",
        security_acknowledged=security_acknowledged,
    )


@app.post(
    "/api/onboarding/validate",
    response_model=ValidationResponse,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Validate a model and dataset submission without auditing it",
    description="Stores the two uploads under a new `upload_id` and runs every check "
    "an audit would run: extension and size limits, duplicate CSV column names, target "
    "column presence, at least two target classes, positive-class membership, "
    "sensitive-column presence, and model/dataset feature compatibility with precise "
    "missing and unexpected feature lists. Returns the honest capability map, so an "
    "unavailable audit (no `predict_proba`, no supported explainability) is known "
    "before the run exists.\n\n"
    "**No audit run is created.** Only the upload bundle is written, under "
    "`runtime/uploads/<upload_id>/`. Pass the returned `upload_id` to "
    "`POST /api/onboarding/audits` to audit the same files without re-uploading."
    + _SECURITY_BLOCK,
)
async def onboarding_validate(
    model_file: UploadFile = File(description="Trusted local .joblib model file."),
    dataset_file: UploadFile = File(description="Labelled CSV dataset."),
    target_column: str = Form(description="Ground-truth label column in the CSV."),
    positive_class: str = Form(
        description="The target value treated as the positive class."
    ),
    decision_threshold: float = Form(
        default=0.5, ge=0.0, le=1.0, description="Probability cut-off for a positive "
        "prediction. Informational for models without predict_proba."
    ),
    sensitive_columns: str | None = Form(
        default=None,
        description="JSON array or comma-separated column names for fairness "
        "reporting. Blank means none selected, reported as 'not_provided_by_user'.",
    ),
    policy_profile_id: str | None = Form(
        default=None, description="Policy profile to audit against. Defaults to "
        "research_governance_policy_v1."
    ),
    security_acknowledged: bool = Form(
        default=False,
        description="Must be true. Explicit acknowledgement of the joblib arbitrary-"
        "code-execution risk, recorded in the run's evidence.",
    ),
) -> ValidationResponse:
    manifest = await _store_submission(
        model_file, dataset_file, security_acknowledged
    )
    config = AuditConfigIn(
        target_column=target_column,
        positive_class=positive_class,
        decision_threshold=decision_threshold,
        sensitive_columns=_parse_sensitive_columns(sensitive_columns),
        **({"policy_profile_id": policy_profile_id} if policy_profile_id else {}),
    )
    return ValidationResponse(
        **audit_service.validate_upload(
            config, manifest, security_acknowledged=security_acknowledged
        )
    )


@app.post(
    "/api/onboarding/audits",
    response_model=AuditCreatedResponse,
    status_code=201,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Create a governance audit run for an uploaded model",
    description="Runs inference with the submitted model on the submitted labelled "
    "data, then computes performance, fairness (only for the selected sensitive "
    "columns), explainability, a risk summary, the five policy gates, the Conformity "
    "Bundle and the traceability matrix, and writes all thirteen artefacts to "
    "`runtime/audits/<audit_run_id>/`.\n\n"
    "Either upload both files here, or pass an `upload_id` returned by "
    "`POST /api/onboarding/validate` to audit files already stored. Validation runs "
    "again either way: if it fails, the response is a 422 listing every issue and "
    "**no run directory is created**.\n\n"
    "The resulting governance state is one of `review_required`, "
    "`insufficient_evidence` or `blocked_by_policy`. None of the three is a deployment "
    "approval, and the Release and Operations gates are never automatically passed. "
    "The built-in Adult Income reference case is untouched by this call."
    + _SECURITY_BLOCK,
)
async def onboarding_create_audit(
    target_column: str = Form(description="Ground-truth label column in the CSV."),
    positive_class: str = Form(
        description="The target value treated as the positive class."
    ),
    model_name: str = Form(description="Name for the submitted model."),
    model_version: str = Form(default="unspecified", description="Model version."),
    model_owner: str = Form(
        default="unspecified", description="Named human accountable for the model."
    ),
    intended_use: str = Form(
        description="What the model is for. Recorded in the run's evidence."
    ),
    decision_context: str = Form(
        description="Who or what the model's decisions affect."
    ),
    model_file: UploadFile | None = File(
        default=None, description="Trusted local .joblib model file. Omit when "
        "supplying upload_id."
    ),
    dataset_file: UploadFile | None = File(
        default=None, description="Labelled CSV dataset. Omit when supplying upload_id."
    ),
    upload_id: str | None = Form(
        default=None,
        description="Reuse a bundle already stored by POST /api/onboarding/validate.",
    ),
    decision_threshold: float = Form(default=0.5, ge=0.0, le=1.0),
    sensitive_columns: str | None = Form(
        default=None,
        description="JSON array or comma-separated column names for fairness "
        "reporting. Blank means none selected, reported as 'not_provided_by_user'.",
    ),
    policy_profile_id: str | None = Form(default=None),
    security_acknowledged: bool = Form(
        default=False,
        description="Must be true. Required even when reusing an upload_id, so the "
        "acknowledgement is evidenced for the run that actually loads the model.",
    ),
) -> AuditCreatedResponse:
    security.require_acknowledgement(security_acknowledged)

    if upload_id:
        manifest = upload_service.load_upload_manifest(upload_id)
    elif model_file is not None and dataset_file is not None:
        manifest = await _store_submission(
            model_file, dataset_file, security_acknowledged
        )
    else:
        raise security.UploadRejected(
            "missing_files",
            "Provide both model_file and dataset_file, or an upload_id from a previous "
            "validation.",
            hint="POST /api/onboarding/validate returns an upload_id you can reuse.",
        )

    config = AuditConfigIn(
        target_column=target_column,
        positive_class=positive_class,
        decision_threshold=decision_threshold,
        sensitive_columns=_parse_sensitive_columns(sensitive_columns),
        **({"policy_profile_id": policy_profile_id} if policy_profile_id else {}),
    )
    metadata = ModelMetadataIn(
        model_name=model_name,
        model_version=model_version,
        model_owner=model_owner,
        intended_use=intended_use,
        decision_context=decision_context,
    )
    result = audit_service.create_audit(
        metadata, config, manifest, security_acknowledged=security_acknowledged
    )
    return AuditCreatedResponse(
        **{key: value for key, value in result.items()
           if key not in ("manifest", "governance")},
        next_step=(
            f"Review GET /api/gates/runs/{result['audit_run_id']}/evaluation and "
            f"GET /api/gates/runs/{result['audit_run_id']}/bundle. A human must make "
            "the release decision: this platform does not authorise deployment."
        ),
    )


@app.get(
    "/api/onboarding/audits",
    response_model=UploadedAuditListResponse,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="List uploaded-model audit runs",
    description="Every completed user-submitted audit run under `runtime/audits/`, "
    "newest first, with its governance state, gate summary, fairness and "
    "explainability status. Listed separately from the built-in Adult Income "
    "reference case, which is served by `/api/models` and `/api/governance`. A run "
    "directory without a readable manifest is omitted rather than half-reported.",
)
async def onboarding_list_audits() -> UploadedAuditListResponse:
    return UploadedAuditListResponse(**audit_service.list_audits())


@app.get(
    "/api/onboarding/audits/{audit_run_id}",
    response_model=UploadedAuditDetail,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Full detail for one uploaded audit run",
    description="Model and dataset identity with both SHA-256 checksums, the target "
    "configuration, the model's inspected capabilities, the feature-compatibility "
    "result, the audit coverage map, and the artefact manifest with the checksum "
    "recorded at creation time. `security_acknowledged` evidences that the joblib "
    "warning was accepted before the model was loaded.",
)
async def onboarding_audit_detail(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedAuditDetail:
    return UploadedAuditDetail(**audit_service.get_audit(audit_run_id))


@app.get(
    "/api/onboarding/audits/{audit_run_id}/performance",
    response_model=UploadedPerformance,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Performance for one uploaded run",
    description="Accuracy, precision, recall, F1, the confusion matrix, the sample "
    "count, the decision threshold and the positive-class definition, read verbatim "
    "from the run's `performance.json`.\n\n"
    "`roc_auc` is **null** when the model exposes no `predict_proba`, with "
    "`roc_auc_unavailable_reason` stating why. It is never estimated, and no "
    "probability is synthesised to produce it.",
)
async def onboarding_audit_performance(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedPerformance:
    return UploadedPerformance(**store.read_json(audit_run_id, "performance.json"))


@app.get(
    "/api/onboarding/audits/{audit_run_id}/fairness",
    response_model=UploadedFairness,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Fairness screening for one uploaded run",
    description="Per-group metrics for the columns the user selected, and only those: "
    "n, actual positive rate, selection rate, TPR, FPR, precision, recall and F1, "
    "with disparities measured against the largest group.\n\n"
    "`status: not_provided_by_user` means no sensitive columns were selected. That is "
    "**not** a pass, not a failure and not a fairness claim. An undefined denominator "
    "yields `null` and is named in `undefined_metrics` -- it is never converted to "
    "zero. The four-fifths threshold is a screening heuristic: it is not a legal "
    "conclusion and does not prove discrimination or causation.",
)
async def onboarding_audit_fairness(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedFairness:
    return UploadedFairness(**store.read_json(audit_run_id, "fairness.json"))


@app.get(
    "/api/onboarding/audits/{audit_run_id}/explainability",
    response_model=UploadedExplainability,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Explainability for one uploaded run",
    description="Permutation importance over the original input feature names, the "
    "possible-proxy screening, and local TreeSHAP explanations where the model type "
    "supports them and additivity could be checked.\n\n"
    "Importance is **association, not causation**. For a model type this prototype "
    "cannot explain, the status is `unavailable` with the reason stated -- no "
    "importance score or local explanation is invented.",
)
async def onboarding_audit_explainability(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedExplainability:
    return UploadedExplainability(
        **store.read_json(audit_run_id, "explainability.json")
    )


@app.get(
    "/api/onboarding/audits/{audit_run_id}/governance",
    response_model=UploadedGovernance,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Governance summary for one uploaded run",
    description="The deterministic governance state -- `review_required`, "
    "`insufficient_evidence` or `blocked_by_policy` -- with its grounds, the risk "
    "items, the coverage map and the capabilities that were unavailable.\n\n"
    "None of the three states is a legal conclusion or a real-world deployment "
    "approval. `human_review_required` is always true and `deployment_authorisation` "
    "is always `not_granted`. The Adult Income reference case has its own separate, "
    "unchanged decision.",
)
async def onboarding_audit_governance(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedGovernance:
    return UploadedGovernance(
        **store.read_json(audit_run_id, "governance_summary.json")
    )


@app.get(
    "/api/onboarding/audits/{audit_run_id}/integrity",
    response_model=UploadedIntegrityResponse,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Verify an uploaded run's evidence integrity",
    description="**Recomputes** the SHA-256 of every artefact the run recorded and "
    "classifies each as verified, changed or missing. Deliberately independent of the "
    "SQLite registry: the baselines live in the run's own directory, so integrity "
    "stays checkable even if registration failed.\n\n"
    "`manifest.json` is reported as `not_baselined` -- a document cannot contain its "
    "own checksum. Checksums detect modification, not authorship: no digital "
    "signature is implemented, so this is change detection and not proof of "
    "provenance.",
)
async def onboarding_audit_integrity(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedIntegrityResponse:
    return UploadedIntegrityResponse(
        **audit_service.check_run_integrity(audit_run_id)
    )


@app.get(
    "/api/onboarding/audits/{audit_run_id}/timeline",
    response_model=UploadedTimelineResponse,
    tags=["onboarding"],
    responses=_INTAKE_ERRORS,
    summary="Chronological history for one uploaded run",
    description="Merges three labelled sources: `run` events derived from the "
    "timestamps the artefacts themselves carry (so the timeline cannot drift from the "
    "evidence), `registry` events from the append-only event log, and `waiver` events "
    "from the waiver register. A missing registry yields fewer events rather than an "
    "error, because the run's own history is complete without it.",
)
async def onboarding_audit_timeline(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> UploadedTimelineResponse:
    return UploadedTimelineResponse(**audit_service.build_timeline(audit_run_id))


# --------------------------------------------------------------------------- #
# Governance-as-Code gates
#
# The re-evaluate and waiver routes write, and only ever inside the target run's own
# runtime/audits/<id>/ directory or the runtime SQLite waiver table. No gate route can
# reach the reference case: it has no runtime audit directory and is not evaluated by
# this policy engine.
# --------------------------------------------------------------------------- #
@app.get(
    "/api/gates/policies",
    response_model=PolicyListResponse,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Versioned governance policy profiles",
    description="The Governance-as-Code policy profiles that ship with this build: "
    "control ids, gate ownership, thresholds, evidence requirements, waiver "
    "eligibility, decision semantics and the policy file's own SHA-256 -- so a reader "
    "can diff the exact policy that produced a decision.\n\n"
    "A gate status is `PASS`, `WAIVE`, `BLOCK` or `NOT_EVALUATED`. The Release and "
    "Operations gates can never be automatically passed.",
)
async def gates_policies() -> PolicyListResponse:
    policies = policy_engine.load_all_policies()
    return PolicyListResponse(count=len(policies), policies=policies)


@app.get(
    "/api/gates/runs/{audit_run_id}/evaluation",
    response_model=GateEvaluation,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Policy gate evaluation for one uploaded run",
    description="The five gates -- Data, Training, Validation, Release, Operations -- "
    "with each control's finding, the blocking controls, the applied waivers, and the "
    "evidence and control coverage scores.\n\n"
    "Coverage figures are **governance coverage metrics**: they describe how much of "
    "the policy could be assessed with the evidence supplied. They are not certified "
    "regulatory compliance scores. Where the Validation Gate blocks on fairness "
    "screening, that is a configured research-policy gate result, not a legal "
    "conclusion or proof of discrimination.",
)
async def gates_evaluation(audit_run_id: str = _AUDIT_ID_PARAM) -> GateEvaluation:
    return gates_service.get_evaluation(audit_run_id)


@app.get(
    "/api/gates/runs/{audit_run_id}/bundle",
    response_model=ConformityBundle,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Conformity Bundle for one uploaded run",
    description="The assembled bundle: bundle id, dataset and model identifiers with "
    "checksums, policy id and version, the gate sequence and decisions, control-level "
    "findings, every evidence path with its SHA-256, the governance and risk "
    "summaries, and the stated limitations.\n\n"
    "The bundle id is derived from the evidence digest and policy version, so "
    "re-evaluating unchanged evidence reproduces the same id. **No digital signature "
    "is implemented** -- the bundle detects change; it does not authenticate an author.",
)
async def gates_bundle(audit_run_id: str = _AUDIT_ID_PARAM) -> ConformityBundle:
    return gates_service.get_bundle(audit_run_id)


@app.get(
    "/api/gates/runs/{audit_run_id}/traceability",
    response_model=TraceabilityMatrix,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Control-to-artefact traceability matrix",
    description="One row per control: the policy requirement, the evidence artefact "
    "path, the source API endpoint where one exists, the expected and actual "
    "checksums, the evidence status, the gate result, the limitation and the "
    "recommended action. `unresolved_evidence` names the controls whose evidence is "
    "missing or whose checksum no longer matches.",
)
async def gates_traceability(
    audit_run_id: str = _AUDIT_ID_PARAM,
) -> TraceabilityMatrix:
    return gates_service.get_traceability(audit_run_id)


@app.post(
    "/api/gates/runs/{audit_run_id}/evaluate",
    response_model=GateEvaluationResponse,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Re-evaluate the policy gates for one uploaded run",
    description="Re-runs the policy engine over the run's existing evidence and "
    "rewrites the four governance artefacts inside that run's own directory. Nothing "
    "outside `runtime/audits/<audit_run_id>/` is touched, and no measurement is "
    "recomputed -- the model is not re-run and no metric changes.\n\n"
    "The engine is deterministic: identical evidence and policy version yield an "
    "identical evaluation, so `changed: false` is the expected result for unchanged "
    "evidence. Use this after recording a waiver, or to confirm that modified evidence "
    "produces a different bundle.",
)
async def gates_evaluate(
    audit_run_id: str = _AUDIT_ID_PARAM,
    policy_profile_id: str | None = Query(
        default=None,
        description="Override the policy. Defaults to the policy the run was "
        "originally audited against, so a re-evaluation does not silently switch "
        "policies.",
    ),
) -> GateEvaluationResponse:
    result = gates_service.evaluate_run(
        audit_run_id, policy_id=policy_profile_id, allow_replace=True
    )
    evaluation = result["evaluation"]
    return GateEvaluationResponse(
        audit_run_id=audit_run_id,
        evaluated_at=result["evaluated_at"],
        policy_profile_id=evaluation.policy_profile_id,
        policy_version=evaluation.policy_version,
        gate_summary=evaluation.gate_summary,
        governance_state=result["governance"]["governance_state"],
        conformity_bundle_id=result["bundle"].bundle_id,
        artifacts_rewritten=result["artifacts_rewritten"],
        changed=result["changed"],
    )


@app.get(
    "/api/gates/runs/{audit_run_id}/waivers",
    response_model=list[Waiver],
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Waivers recorded against one uploaded run",
    description="Every waiver in the runtime register for this run, with its status "
    "computed against the current time. An expired or revoked waiver has no effect "
    "and its control reverts to `BLOCK`.\n\n"
    "`created_by_platform` is always false: the platform never creates or approves a "
    "waiver. No waiver exists for the built-in Adult Income reference case.",
)
async def gates_list_waivers(audit_run_id: str = _AUDIT_ID_PARAM) -> list[Waiver]:
    # The run's existence is checked before the register is queried. Without this,
    # a typo in the run id would return `[]` -- indistinguishable from "this run has
    # no waivers", which is the one reading a governance reviewer must not be given.
    if not store.audit_dir(audit_run_id).is_dir():
        raise store.AuditRunNotFound(audit_run_id, store.list_audit_ids())
    return gates_service.list_waivers(audit_run_id)


@app.post(
    "/api/gates/runs/{audit_run_id}/waivers",
    response_model=Waiver,
    status_code=201,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Record an explicit, time-bounded waiver",
    description="Records a named human's acceptance of one unmet control. Every field "
    "is required: a waiver without an owner, an expiry or a rationale is an "
    "unattributed override, which is the failure mode a waiver register exists to "
    "prevent.\n\n"
    "Refused with 422 when the control is unknown, is not waiver-eligible under the "
    "policy, or the expiry is absent, malformed or already past. **A waiver can never "
    "satisfy or override the Release Gate**: release is a human accountability "
    "decision, not a risk that can be accepted by annotation. Recording a waiver does "
    "not re-evaluate the gates -- call `POST /api/gates/runs/{audit_run_id}/evaluate` "
    "to see its effect. Waivers are stored only in the runtime SQLite state.",
)
async def gates_create_waiver(
    payload: WaiverIn,
    audit_run_id: str = _AUDIT_ID_PARAM,
    policy_profile_id: str | None = Query(
        default=None, description="Policy whose waiver rules apply. Defaults to the "
        "policy the run was audited against."
    ),
) -> Waiver:
    policy = policy_engine.load_policy(
        policy_profile_id
        or str(
            store.read_json(audit_run_id, "manifest.json").get("policy_profile_id")
            or ""
        )
        or None
    )
    return gates_service.create_waiver(audit_run_id, payload, policy)


@app.post(
    "/api/gates/runs/{audit_run_id}/waivers/{waiver_id}/revoke",
    response_model=Waiver,
    tags=["gates"],
    responses=_INTAKE_ERRORS,
    summary="Revoke a recorded waiver",
    description="Marks a waiver revoked. The row is not deleted -- the register is a "
    "history, and erasing an accepted risk would erase the fact that it was once "
    "accepted. A revoked waiver has no effect and its control reverts to `BLOCK` on "
    "the next evaluation.",
)
async def gates_revoke_waiver(
    audit_run_id: str = _AUDIT_ID_PARAM,
    waiver_id: str = PathParam(description="Waiver id to revoke."),
) -> Waiver:
    return gates_service.revoke_waiver(audit_run_id, waiver_id)
