"""
app/registry/schemas.py
=======================
Pydantic response schemas for the governance audit registry (Phase 8).

The registry **records evidence; it does not make decisions.** The governance
decision it stores is a copy of the committed decision record, carried so a run
can be read without a second lookup. Nothing in this layer evaluates, re-derives
or alters it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_CFG = ConfigDict(protected_namespaces=())

RunStatus = Literal["active", "superseded", "archived"]
RunType = Literal["reference_case", "uploaded_model"]
IntegrityStatus = Literal[
    "verified", "incomplete", "modified", "modified_and_incomplete"
]

RUN_TYPE_NOTE = (
    "'reference_case' is the built-in Adult Income audit. 'uploaded_model' is a run "
    "produced from a user submission and stored under runtime/. The two are separate "
    "records: neither supersedes the other, and no uploaded run affects the reference "
    "case's decision."
)

REGISTRY_DISCLAIMER = (
    "The registry records what evidence existed and what its checksums were. It "
    "does not make, review or alter any governance decision -- the decision stored "
    "on a run is a copy of the committed decision record."
)


# --------------------------------------------------------------------------- #
# Coverage / decision blocks
# --------------------------------------------------------------------------- #
class AuditCoverage(BaseModel):
    """
    Which audit phases are evidenced by this run.

    This is the reference case's coverage shape. An uploaded-model run reports a
    different set of capability flags (``roc_auc``, ``explainability_global``,
    ``explainability_local_shap``), so the fields that carry it are typed as a union
    with a plain mapping. Coercing one shape into the other would mean reporting
    coverage the run does not have.
    """

    model_config = _CFG

    performance: bool
    fairness: bool
    explainability: bool
    governance: bool
    agents: bool
    complete: bool = Field(description="True when all five phases are covered.")
    models_evaluated: list[str] = Field(default_factory=list)
    models_with_explainability: list[str] = Field(default_factory=list)
    sensitive_attributes: list[str] = Field(default_factory=list)
    agent_names: list[str] = Field(default_factory=list)


class RecordedDecision(BaseModel):
    """The governance decision as recorded at registration time."""

    model_config = _CFG

    research_use: str
    real_world_deployment: str
    headline: str
    decision_date: str | None = None
    source: str
    note: str = (
        "Copied from the committed decision record. The registry does not make or "
        "change decisions."
    )


class PerformanceSummary(BaseModel):
    """Headline metrics for the run's subject model, quoted verbatim."""

    model_config = _CFG

    model_name: str
    n_test: int | None = None
    decision_threshold: float | None = None
    threshold_applied: bool | None = Field(
        default=None,
        description="False when the model exposes no predict_proba, so the threshold "
        "was not applied and the model's own decision rule produced the labels.",
    )
    positive_class: str | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    true_negatives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    true_positives: int | None = None
    roc_auc_unavailable_reason: str | None = Field(
        default=None,
        description="Why ROC-AUC is null. Present rather than a fabricated value.",
    )


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
class RegisteredArtifact(BaseModel):
    model_config = _CFG

    group: str
    path: str
    sha256: str
    size_bytes: int
    modified_utc: str


class AuditRunSummary(BaseModel):
    """Compact run record, for the list endpoint."""

    model_config = _CFG

    run_id: str
    run_type: RunType = Field(default="reference_case", description=RUN_TYPE_NOTE)
    status: RunStatus
    created_at: str
    refreshed_at: str
    refresh_count: int
    dataset_name: str
    dataset_version: str
    model_name: str
    model_version: str
    evidence_digest: str
    artifact_count: int
    coverage_complete: bool
    research_use: str
    real_world_deployment: str


class AuditRunDetail(BaseModel):
    """Full run record."""

    model_config = _CFG

    run_id: str
    run_type: RunType = Field(default="reference_case", description=RUN_TYPE_NOTE)
    schema_version: int
    status: RunStatus
    created_at: str
    refreshed_at: str
    refresh_count: int
    dataset_name: str
    dataset_version: str
    dataset_context: str
    model_name: str
    model_version: str
    model_run_identifier: str
    evidence_digest: str
    artifact_count: int
    performance_summary: PerformanceSummary | dict[str, Any] = Field(
        description="The reference shape for a reference-case run. An uploaded run "
        "whose performance artefact is absent reports why instead."
    )
    governance_decision: RecordedDecision
    blocking_risk_ids: list[str]
    audit_coverage: AuditCoverage | dict[str, Any] = Field(
        description="Reference-case runs use the five-phase shape; uploaded runs "
        "report their own capability flags."
    )
    artifacts: list[RegisteredArtifact]
    disclaimer: str = REGISTRY_DISCLAIMER


class RunListResponse(BaseModel):
    model_config = _CFG

    count: int
    active_run_id: str | None = Field(
        default=None,
        description="The active reference-case run. Scoped by run type on purpose: "
        "registering uploaded audits never moves it.",
    )
    active_reference_run_id: str | None = Field(
        default=None, description="Explicit alias for active_run_id."
    )
    uploaded_run_ids: list[str] = Field(
        default_factory=list,
        description="Uploaded-model runs, newest first. Listed separately so they are "
        "never mistaken for versions of the reference case.",
    )
    database: str = Field(description="Registry database path, repo-relative.")
    filters_applied: dict[str, str | None] = Field(default_factory=dict)
    run_type_note: str = RUN_TYPE_NOTE
    runs: list[AuditRunSummary]
    disclaimer: str = REGISTRY_DISCLAIMER


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
class ArtifactIntegrity(BaseModel):
    model_config = _CFG

    group: str
    path: str
    status: Literal["verified", "missing", "changed"]
    expected_sha256: str
    actual_sha256: str | None = None
    expected_size_bytes: int
    actual_size_bytes: int | None = None
    detail: str


class IntegrityResponse(BaseModel):
    """Result of recomputing every registered checksum, right now."""

    model_config = _CFG

    run_id: str
    run_status: RunStatus
    checked_at: str
    integrity_status: IntegrityStatus
    integrity_ok: bool
    artifacts_checked: int
    verified_count: int
    missing_count: int
    changed_count: int
    verified_files: list[str]
    missing_files: list[str]
    changed_files: list[str]
    registered_evidence_digest: str
    current_evidence_digest: str = Field(
        description="Digest recomputed over the registered artefact set as it is now. "
        "Differs from the registered digest when anything changed or went missing."
    )
    interpretation: list[str] = Field(
        description="What this result does and does not establish."
    )
    artifacts: list[ArtifactIntegrity]


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #
class TimelineEvent(BaseModel):
    model_config = _CFG

    event_time: str
    event_type: str
    source: Literal["registry", "evidence"] = Field(
        description="'registry' events are recorded actions; 'evidence' events are "
        "derived from artefact modification times."
    )
    detail: str


class TimelineResponse(BaseModel):
    model_config = _CFG

    run_id: str
    run_status: RunStatus
    count: int = Field(description="Events returned in this response.")
    total_events: int = Field(
        description="Full event count. Always reported, so a window is never a "
        "silent cap."
    )
    truncated: bool = Field(
        description="True when older events were omitted from this response."
    )
    events: list[TimelineEvent]
    note: str = (
        "Evidence events are derived from file modification times, which reflect "
        "when a file was last written on this machine -- not a signed provenance "
        "record. Registry events are an append-only log of registry actions."
    )


# --------------------------------------------------------------------------- #
# Registration result (used by the CLI)
# --------------------------------------------------------------------------- #
class RegistrationResult(BaseModel):
    model_config = _CFG

    run_id: str
    run_type: RunType = "reference_case"
    action: Literal["created", "refreshed", "superseded_previous"]
    created_at: str
    refreshed_at: str
    refresh_count: int
    artifact_count: int
    evidence_digest: str
    superseded_run_ids: list[str] = Field(default_factory=list)
    database: str
    message: str


class RegistryStatsResponse(BaseModel):
    """Small helper payload used by the dashboard header."""

    model_config = _CFG

    total_runs: int
    active: int
    superseded: int
    archived: int
    by_run_type: dict[str, int] = Field(
        default_factory=dict,
        description="Run counts split by run type, so the reference case and uploaded "
        "audits are countable separately.",
    )
    database: str
    initialised: bool
