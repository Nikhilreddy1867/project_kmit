"""
app/gates/schemas.py
====================
Pydantic v2 schemas for Governance-as-Code: policy profiles, gate evaluations,
waivers, the Conformity Bundle and clause-to-artefact traceability.

Four statuses, and the difference between two of them matters most
-----------------------------------------------------------------
``PASS`` / ``WAIVE`` / ``BLOCK`` / ``NOT_EVALUATED``.

``NOT_EVALUATED`` is not a soft pass. It means the evidence a control needs does
not exist -- no sensitive columns were selected, the model has no ``predict_proba``,
or the control (human release authorisation, monitoring, rollback) is simply outside
what this prototype can hold. Every consumer of these schemas is expected to render
it as *absent evidence*, and the governance state derived from an evaluation treats
it as ``insufficient_evidence`` rather than as progress.

``WAIVE`` is likewise not a pass. It records that a named human accepted a
documented risk until a stated expiry, with the requirement still unmet.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_CFG = ConfigDict(protected_namespaces=())

GateStatus = Literal["PASS", "WAIVE", "BLOCK", "NOT_EVALUATED"]
EvidenceStatus = Literal["verified", "changed", "missing", "not_applicable"]
WaiverStatus = Literal["active", "expired", "revoked"]

#: Required verbatim wherever a fairness screening result blocks a gate, so the
#: result can never be read as a legal finding.
FAIRNESS_GATE_NOTICE = (
    "This is a configured research-policy gate result, not a legal conclusion or "
    "proof of discrimination."
)

GATE_DECISION_NOTICE = (
    "Deterministic decision-support evidence for human governance review. Gate "
    "results are advisory: they are not statements of legal compliance, not findings "
    "of discrimination, not causal claims, and not authorisation to deploy."
)


# --------------------------------------------------------------------------- #
# Policy profiles
# --------------------------------------------------------------------------- #
class PolicyThreshold(BaseModel):
    model_config = _CFG
    name: str
    value: Any
    applies_when: str
    on_absent_evidence: str | None = Field(
        default=None,
        description="What the engine does when the evidence this threshold needs is "
        "absent. Never a silent pass.",
    )
    rationale: str


class PolicyControl(BaseModel):
    model_config = _CFG
    control_id: str
    gate: str
    title: str
    requirement: str
    evidence_artifact: str | None = Field(
        default=None,
        description="Run artefact this control is evidenced by; null when the "
        "evidence does not exist in this prototype.",
    )
    evidence_required: bool
    api_endpoint: str | None = None
    owner: str
    waiver_eligible: bool
    limitation: str


class PolicyGate(BaseModel):
    model_config = _CFG
    gate_code: str
    gate_name: str
    order: int
    owner: str
    question: str
    controls: list[str] = Field(
        default_factory=list, description="Control ids belonging to this gate."
    )
    never_auto_pass: bool = Field(
        default=False,
        description="True for the Release and Operations gates, which can only be "
        "BLOCK or NOT_EVALUATED.",
    )


class PolicyProfile(BaseModel):
    model_config = _CFG
    policy_id: str
    policy_name: str
    policy_version: str
    policy_status: str
    effective_from: str | None = None
    purpose: str
    applies_to: dict[str, Any]
    gates: list[PolicyGate]
    controls: list[PolicyControl]
    thresholds: list[PolicyThreshold]
    statuses: dict[str, str] = Field(description="Meaning of each gate status.")
    gate_result_rule: str
    never_auto_pass: dict[str, Any]
    decision_semantics: dict[str, str]
    waiver_rules: dict[str, Any]
    coverage_metrics: dict[str, Any]
    limitations: list[str]
    source_file: str = Field(
        description="Repo-relative path of the policy file this was loaded from, so a "
        "reader can diff the policy that produced a decision."
    )
    checksum: str = Field(description="SHA-256 of the policy file as loaded.")


class PolicyListResponse(BaseModel):
    model_config = _CFG
    count: int
    policies: list[PolicyProfile]
    notice: str = GATE_DECISION_NOTICE


# --------------------------------------------------------------------------- #
# Waivers
# --------------------------------------------------------------------------- #
class WaiverIn(BaseModel):
    """
    A human-created, time-bounded acceptance of one unmet control.

    Every field is required because a waiver without an owner, an expiry or a
    rationale is not an accountable decision -- it is an unattributed override, which
    is the failure mode waiver registers exist to prevent.
    """

    model_config = _CFG

    control_id: str = Field(min_length=2, description="The single control waived.")
    scope: str = Field(
        min_length=3,
        max_length=500,
        description="Exactly what is covered, e.g. 'research evaluation only, no "
        "real-world decisions'.",
    )
    owner: str = Field(
        min_length=2, max_length=120, description="Named human accepting the risk."
    )
    expires_at: str = Field(
        description="ISO-8601 expiry. Required: an open-ended waiver is a permanent "
        "exception with no review point."
    )
    rationale: str = Field(
        min_length=10,
        max_length=2000,
        description="Why the risk is acceptable for the stated scope.",
    )
    compensating_controls: list[str] = Field(
        min_length=1,
        description="What reduces the risk in the meantime. At least one is required.",
    )


class Waiver(BaseModel):
    model_config = _CFG
    waiver_id: str
    audit_run_id: str
    control_id: str
    gate: str | None = None
    scope: str
    owner: str
    created_at: str
    expires_at: str
    rationale: str
    compensating_controls: list[str]
    status: WaiverStatus = Field(
        description="'active' only while unexpired and not revoked. An expired waiver "
        "has no effect and its control reverts to BLOCK."
    )
    created_by_platform: Literal[False] = Field(
        default=False,
        description="Always false. The platform never creates or approves a waiver.",
    )
    notice: str = (
        "A waiver records an accepted risk with an expiry. It does not make the "
        "requirement met, and it can never satisfy the Release Gate."
    )


# --------------------------------------------------------------------------- #
# Gate evaluation
# --------------------------------------------------------------------------- #
class ControlFinding(BaseModel):
    """One control's result, with the evidence trail that produced it."""

    model_config = _CFG

    control_id: str
    gate: str
    title: str
    policy_requirement: str
    evidence_artifact_path: str | None = Field(
        default=None, description="Repo-relative path under runtime/."
    )
    source_api_endpoint: str | None = None
    expected_checksum: str | None = Field(
        default=None, description="SHA-256 recorded when the artefact was created."
    )
    actual_checksum: str | None = Field(
        default=None, description="SHA-256 recomputed at evaluation time."
    )
    evidence_status: EvidenceStatus
    gate_result: GateStatus
    observed: dict[str, Any] = Field(
        default_factory=dict,
        description="The measured values this result was decided from.",
    )
    reason: str = Field(description="Why this result, in one sentence.")
    limitation: str
    recommended_action: str
    waiver_id: str | None = None
    waiver_eligible: bool = True


class GateResult(BaseModel):
    model_config = _CFG
    gate_code: str
    gate_name: str
    order: int
    owner: str
    question: str
    status: GateStatus
    reason: str
    never_auto_pass: bool = False
    control_ids: list[str] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)


class GateEvaluation(BaseModel):
    model_config = _CFG
    audit_run_id: str
    policy_profile_id: str
    policy_version: str
    policy_checksum: str
    evaluated_at: str
    deterministic: Literal[True] = Field(
        default=True,
        description="Identical evidence and policy version always yield an identical "
        "evaluation.",
    )
    gates: list[GateResult]
    controls: list[ControlFinding]
    gate_summary: dict[str, str] = Field(
        description="Gate code -> status, for compact display."
    )
    status_counts: dict[str, int]
    blocking_controls: list[str] = Field(default_factory=list)
    evidence_coverage_score: float | None = Field(
        default=None,
        description="Verified required evidence / total required evidence. A "
        "governance coverage metric -- NOT a certified regulatory compliance score.",
    )
    control_coverage_score: float | None = Field(
        default=None,
        description="Evaluated applicable controls / total applicable controls. A "
        "governance coverage metric -- NOT a certified regulatory compliance score.",
    )
    coverage_metric_caveat: str = (
        "Coverage describes how much of the policy could be assessed with the evidence "
        "supplied. It is not a compliance percentage and confers no certification."
    )
    waivers_applied: list[Waiver] = Field(default_factory=list)
    fairness_gate_notice: str | None = Field(
        default=None,
        description="Present when a fairness screening result affected a gate.",
    )
    release_gate_note: str
    limitations: list[str] = Field(default_factory=list)
    notice: str = GATE_DECISION_NOTICE


# --------------------------------------------------------------------------- #
# Conformity Bundle and traceability
# --------------------------------------------------------------------------- #
class BundleEvidenceItem(BaseModel):
    model_config = _CFG
    artifact: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    status: EvidenceStatus
    source_api_endpoint: str | None = None


class ConformityBundle(BaseModel):
    """
    The assembled evidence package for one uploaded audit run.

    The bundle id is content-addressed: identical evidence under an identical policy
    version yields an identical id. That makes it a citable reference for a specific
    set of facts rather than a per-request identifier, and it is why re-evaluating an
    unchanged run does not produce a new bundle id.
    """

    model_config = _CFG

    bundle_id: str
    bundle_version: str = "1"
    audit_run_id: str
    run_type: str = "uploaded_model"
    created_at: str
    model_name: str
    model_version: str
    model_owner: str
    model_checksum: str
    dataset_identifier: str
    dataset_checksum: str
    dataset_row_count: int
    policy_profile_id: str
    policy_version: str
    policy_checksum: str
    policy_evaluated_at: str
    gate_sequence: list[str]
    gate_decisions: dict[str, str]
    control_findings: list[ControlFinding]
    evidence: list[BundleEvidenceItem]
    evidence_digest: str = Field(
        description="SHA-256 over the sorted path:sha256 lines of the evidence set. "
        "The input to the bundle id."
    )
    governance_summary: dict[str, Any]
    risk_summary: dict[str, Any]
    audit_coverage: dict[str, bool]
    evidence_coverage_score: float | None = None
    control_coverage_score: float | None = None
    coverage_metric_caveat: str = (
        "Governance coverage metrics, not certified regulatory compliance scores."
    )
    signature: None = Field(
        default=None,
        description="Always null. No digital signature scheme is implemented in this "
        "prototype -- integrity is SHA-256 change detection only, with no signing key "
        "and no non-repudiation.",
    )
    limitations: list[str]
    disclaimers: list[str]
    notice: str = GATE_DECISION_NOTICE


class TraceabilityRow(BaseModel):
    """One clause/control mapped to the artefact that evidences it."""

    model_config = _CFG

    control_id: str
    gate: str
    policy_requirement: str
    evidence_artifact_path: str | None = None
    source_api_endpoint: str | None = None
    expected_checksum: str | None = None
    actual_checksum: str | None = None
    evidence_status: EvidenceStatus
    gate_result: GateStatus
    limitation: str
    recommended_action: str


class TraceabilityMatrix(BaseModel):
    model_config = _CFG
    audit_run_id: str
    policy_profile_id: str
    policy_version: str
    generated_at: str
    rows: list[TraceabilityRow]
    evidence_coverage_score: float | None = None
    control_coverage_score: float | None = None
    coverage_metric_caveat: str = (
        "Governance coverage metrics, not certified regulatory compliance scores."
    )
    unresolved_evidence: list[str] = Field(
        default_factory=list,
        description="Controls whose evidence path is missing or whose checksum "
        "changed since creation.",
    )
    notice: str = GATE_DECISION_NOTICE


class GateEvaluationResponse(BaseModel):
    """Result of re-running the policy engine over an existing run's evidence."""

    model_config = _CFG

    audit_run_id: str
    evaluated_at: str
    policy_profile_id: str
    policy_version: str
    gate_summary: dict[str, str]
    governance_state: str
    conformity_bundle_id: str
    artifacts_rewritten: list[str] = Field(
        description="Generated artefacts recomputed by this call. Only ever files "
        "inside this run's own runtime/audits/<id>/ directory."
    )
    changed: bool = Field(
        description="False when the re-evaluation reproduced the previous result "
        "exactly, which is the expected outcome for unchanged evidence."
    )
    notice: str = GATE_DECISION_NOTICE
