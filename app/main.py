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

from fastapi import FastAPI, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.agents import orchestrator
from app.agents.schemas import AgentListResponse, AgentReport, GovernanceReview
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
VERSION = "0.5.0"

DESCRIPTION = """
Read-only API over the completed **Adult / Census Income** governance audit.

Every number served here is read **verbatim** from the audit artefacts in
`results/`. The API performs no model training, no re-scoring and no metric
recomputation, so it cannot drift from the committed evidence.

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
`/api/governance/decision`.
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
]

app = FastAPI(
    title="AI Governance Platform API",
    summary="Read-only access to the Adult Income model audit evidence.",
    description=DESCRIPTION,
    version=VERSION,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "AI Governance Platform (Phase 5 MVP)"},
    license_info={"name": "Academic / research use only"},
)

# --------------------------------------------------------------------------- #
# CORS -- for a future local dashboard (Vite 5173, CRA 3000, etc.)
# Any localhost/127.0.0.1 port is allowed; credentials are NOT enabled and the
# wildcard origin is deliberately avoided, since a browser would otherwise be
# able to read this API from any site the user visits.
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],  # read-only API
    allow_headers=["*"],
    expose_headers=["X-Data-Source"],
)


@app.middleware("http")
async def _tag_read_only(request: Request, call_next):
    """Advertise the read-only contract on every response."""
    response = await call_next(request)
    response.headers["X-Data-Source"] = "results/ (read-only audit artefacts)"
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


_ERRORS = {
    404: {"model": ErrorDetail, "description": "Unknown model or attribute name"},
    500: {"model": ErrorDetail, "description": "Audit artefact is malformed"},
    503: {"model": ErrorDetail, "description": "Audit artefact missing on disk"},
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
