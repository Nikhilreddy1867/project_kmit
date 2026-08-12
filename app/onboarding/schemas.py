"""
app/onboarding/schemas.py
=========================
Pydantic v2 schemas for the model-intake and uploaded-audit endpoints.

Conventions inherited from ``app/schemas/models.py``
---------------------------------------------------
* Metric fields are ``float | None``. ``None`` means **not computable** -- an
  undefined denominator, or a capability the uploaded model does not have. It
  never means zero, and the API never substitutes zero for it.
* ``model_config = ConfigDict(protected_namespaces=())`` because many fields
  legitimately start with ``model_``.
* Field descriptions carry the interpretation limits, so they appear in Swagger
  where a consumer will actually read them.

Every response that carries a governance conclusion also carries
:data:`DECISION_SUPPORT_NOTICE`, so no caller can receive a verdict from this API
without also receiving the statement that it is not a deployment authorisation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_CFG = ConfigDict(protected_namespaces=())

DECISION_SUPPORT_NOTICE = (
    "Deterministic decision-support evidence for human governance review. This is "
    "not a legal compliance assessment, not a finding of discrimination, not a "
    "causal claim, and not an authorisation to deploy."
)

FOUR_FIFTHS_NOTICE = (
    "The four-fifths threshold is a screening heuristic. It is not a legal "
    "conclusion and does not prove discrimination or causation."
)

#: Deterministic governance states an uploaded run may be in. None of the three is
#: an approval: the best available outcome is "a human must now review this".
GovernanceState = Literal["review_required", "insufficient_evidence", "blocked_by_policy"]

FairnessStatus = Literal["available", "not_provided_by_user", "unavailable"]
ExplainabilityStatus = Literal["available", "unavailable", "partial"]
RunType = Literal["reference_case", "uploaded_model"]


# --------------------------------------------------------------------------- #
# Intake configuration
# --------------------------------------------------------------------------- #
class ModelMetadataIn(BaseModel):
    """User-declared provenance for the uploaded model."""

    model_config = _CFG

    model_name: str = Field(min_length=1, max_length=120)
    model_version: str = Field(min_length=1, max_length=60)
    model_owner: str = Field(
        min_length=1, max_length=120, description="Accountable owner or team."
    )
    intended_use: str = Field(
        min_length=1, max_length=2000, description="What the model is intended to do."
    )
    decision_context: str = Field(
        min_length=1,
        max_length=2000,
        description="Where its output would be used, and about whom. Drives how "
        "severely the risk summary reads.",
    )
    description: str | None = Field(default=None, max_length=4000)


class AuditConfigIn(BaseModel):
    """How to interpret the uploaded dataset."""

    model_config = _CFG

    target_column: str = Field(min_length=1)
    positive_class: str = Field(
        min_length=1,
        description="The target value treated as the positive class, as a string. "
        "Compared against the string form of the column's values.",
    )
    decision_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Applied to predicted probabilities when the model exposes "
        "predict_proba. Ignored (and reported as such) for predict-only models.",
    )
    sensitive_columns: list[str] = Field(
        default_factory=list,
        description="Columns to group by for fairness. Empty is allowed and yields "
        "fairness status 'not_provided_by_user' -- which is not a pass.",
    )
    policy_profile_id: str | None = Field(
        default=None, description="Policy profile to evaluate against."
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class ValidationIssue(BaseModel):
    model_config = _CFG
    code: str = Field(description="Stable machine-readable identifier.")
    severity: Literal["error", "warning"] = Field(
        description="'error' blocks audit creation; 'warning' does not."
    )
    field: str | None = Field(default=None, description="Input this concerns.")
    message: str
    hint: str | None = None


class DatasetProfile(BaseModel):
    """Structural facts about the uploaded CSV, with no metrics attached."""

    model_config = _CFG
    row_count: int
    column_count: int
    columns: list[str]
    target_column: str | None = None
    target_classes: list[str] = Field(default_factory=list)
    target_class_counts: dict[str, int] = Field(default_factory=dict)
    positive_class: str | None = None
    positive_class_count: int | None = None
    sensitive_columns: list[str] = Field(default_factory=list)
    duplicate_columns: list[str] = Field(default_factory=list)


class ModelCapabilities(BaseModel):
    """What the uploaded estimator can actually do, established by inspection."""

    model_config = _CFG
    loaded: bool = Field(
        description="False when the model was never deserialised -- either the "
        "security warning was not acknowledged, or structural validation failed first."
    )
    estimator_type: str | None = Field(
        default=None, description="Python class name of the loaded object."
    )
    estimator_module: str | None = None
    is_pipeline: bool = False
    pipeline_steps: list[str] = Field(default_factory=list)
    final_estimator: str | None = None
    has_predict: bool = False
    has_predict_proba: bool = Field(
        default=False,
        description="When False, ROC-AUC is reported as null rather than estimated "
        "from anything else.",
    )
    is_binary_classifier: bool = False
    classes: list[str] = Field(default_factory=list)
    expected_features: list[str] | None = Field(
        default=None,
        description="From feature_names_in_ where the estimator records it; null "
        "when the estimator does not expose feature names.",
    )
    n_features_expected: int | None = None
    supports_permutation_importance: bool = False
    supports_treeshap: bool = Field(
        default=False,
        description="True only for XGBoost models where exact TreeSHAP additivity "
        "can be verified.",
    )


class FeatureCompatibility(BaseModel):
    """Result of matching dataset columns against the model's expected features."""

    model_config = _CFG
    checked: bool
    compatible: bool
    missing_features: list[str] = Field(
        description="Features the model requires that the dataset does not provide."
    )
    unexpected_features: list[str] = Field(
        description="Dataset columns the model does not expect. Not fatal: they are "
        "dropped for inference and retained for fairness reporting."
    )
    matched_feature_count: int | None = None
    sensitive_columns_retained: list[str] = Field(
        default_factory=list,
        description="Sensitive columns kept for fairness reporting even when they "
        "are not model inputs.",
    )
    method: str = Field(
        description="How compatibility was established (feature_names_in_, "
        "n_features_in_, or a trial inference)."
    )


class ValidationResponse(BaseModel):
    model_config = _CFG
    valid: bool = Field(description="True when no 'error'-severity issue was raised.")
    upload_id: str | None = Field(
        default=None,
        description="Set when the uploads were stored. Pass it to POST "
        "/api/onboarding/audits to audit the same files without re-uploading.",
    )
    security_acknowledged: bool
    security_warning: str
    production_hardening: str
    issues: list[ValidationIssue]
    dataset: DatasetProfile | None = None
    model_capabilities: ModelCapabilities | None = None
    feature_compatibility: FeatureCompatibility | None = None
    audit_capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Which audits can run for this model/dataset pair. Unavailable "
        "capabilities are reported honestly rather than approximated.",
    )
    next_step: str
    notice: str = DECISION_SUPPORT_NOTICE


# --------------------------------------------------------------------------- #
# Audit results
# --------------------------------------------------------------------------- #
class UploadedConfusionMatrix(BaseModel):
    model_config = _CFG
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


class UploadedPerformance(BaseModel):
    model_config = _CFG
    audit_run_id: str
    n_samples: int
    positive_class: str
    negative_classes: list[str] = Field(default_factory=list)
    decision_threshold: float
    threshold_applied: bool = Field(
        description="False for predict-only models, where the model's own decision "
        "rule is used and the threshold field is informational."
    )
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = Field(
        default=None,
        description="Null when the model exposes no predict_proba. Never estimated.",
    )
    roc_auc_unavailable_reason: str | None = None
    confusion_matrix: UploadedConfusionMatrix
    computed_from: str = Field(
        description="The artefact these numbers were computed from and stored in."
    )
    caveats: list[str]
    notice: str = DECISION_SUPPORT_NOTICE


class UploadedFairnessGroup(BaseModel):
    model_config = _CFG
    attribute: str
    group: str
    n: int
    is_reference: bool
    small_group: bool = Field(description="n below the small-group threshold.")
    actual_positive_rate: float | None = None
    selection_rate: float | None = None
    true_positive_rate: float | None = None
    false_positive_rate: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    demographic_parity_difference: float | None = None
    demographic_parity_ratio: float | None = None
    disparate_impact_ratio: float | None = None
    equal_opportunity_difference: float | None = None
    equalized_odds_tpr_difference: float | None = None
    equalized_odds_fpr_difference: float | None = None
    passes_four_fifths_screen: bool | None = Field(
        default=None,
        description="Null when the ratio is undefined. Screening only -- see the "
        "four-fifths notice.",
    )
    undefined_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics whose denominator was empty for this group. Reported as "
        "null, never as zero.",
    )


class UploadedFairnessAttribute(BaseModel):
    model_config = _CFG
    attribute: str
    reference_group: str
    reference_n: int
    n_groups: int
    min_disparate_impact_ratio: float | None = None
    worst_group: str | None = None
    groups_failing_four_fifths: list[str] = Field(default_factory=list)
    max_abs_equal_opportunity_difference: float | None = None
    max_abs_equalized_odds_fpr_difference: float | None = None
    small_groups_present: bool = False
    undefined_metric_count: int = 0


class UploadedFairness(BaseModel):
    model_config = _CFG
    audit_run_id: str
    status: FairnessStatus = Field(
        description="'not_provided_by_user' means no sensitive columns were "
        "selected. That is neither a pass nor a fairness claim."
    )
    status_detail: str
    sensitive_columns_requested: list[str] = Field(default_factory=list)
    reference_group_rule: str = "Largest group by sample count."
    small_group_threshold: int
    attributes: list[UploadedFairnessAttribute] = Field(default_factory=list)
    groups: list[UploadedFairnessGroup] = Field(default_factory=list)
    interpretation: list[str] = Field(default_factory=list)
    four_fifths_notice: str = FOUR_FIFTHS_NOTICE
    notice: str = DECISION_SUPPORT_NOTICE


class UploadedFeatureImportance(BaseModel):
    model_config = _CFG
    rank: int
    feature: str
    importance_mean: float
    importance_std: float | None = None
    is_selected_sensitive_column: bool = False
    possible_proxy_for: list[str] = Field(default_factory=list)


class UploadedLocalFactor(BaseModel):
    model_config = _CFG
    feature: str
    value: str
    contribution: float


class UploadedLocalExplanation(BaseModel):
    model_config = _CFG
    case_type: str
    row_index: int
    predicted_probability: float | None = None
    predicted_label: str
    actual_label: str
    base_value: float
    top_factors: list[UploadedLocalFactor]


class UploadedExplainability(BaseModel):
    model_config = _CFG
    audit_run_id: str
    status: ExplainabilityStatus
    status_detail: str
    method: str | None = None
    n_repeats: int | None = None
    scorer: str | None = None
    global_importance: list[UploadedFeatureImportance] = Field(default_factory=list)
    local_method: str | None = None
    local_explanations: list[UploadedLocalExplanation] = Field(default_factory=list)
    proxy_assessment: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    notice: str = DECISION_SUPPORT_NOTICE


class UploadedRiskItem(BaseModel):
    model_config = _CFG
    risk_id: str
    category: str
    severity: Literal["critical", "high", "medium", "low", "info"]
    statement: str
    evidence: str
    limitation: str
    recommended_action: str


class UploadedGovernance(BaseModel):
    model_config = _CFG
    audit_run_id: str
    governance_state: GovernanceState
    state_meaning: str
    state_grounds: list[str]
    human_review_required: Literal[True] = Field(
        default=True,
        description="Always true. This platform never issues a deployment approval "
        "for an uploaded model.",
    )
    deployment_authorisation: str = "not_granted"
    audit_coverage: dict[str, bool]
    unavailable_capabilities: list[str] = Field(default_factory=list)
    risks: list[UploadedRiskItem] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    reference_case_note: str = Field(
        default="The built-in Adult Income reference case has its own separate, "
        "unchanged governance decision. Nothing here affects it.",
    )
    limitations: list[str] = Field(default_factory=list)
    notice: str = DECISION_SUPPORT_NOTICE


# --------------------------------------------------------------------------- #
# Run records
# --------------------------------------------------------------------------- #
class UploadedAuditSummary(BaseModel):
    model_config = _CFG
    audit_run_id: str
    run_type: RunType = "uploaded_model"
    created_at: str
    model_name: str
    model_version: str
    model_owner: str
    dataset_row_count: int
    dataset_column_count: int
    target_column: str
    positive_class: str
    decision_threshold: float
    sensitive_columns: list[str]
    policy_profile_id: str
    policy_version: str
    governance_state: GovernanceState
    fairness_status: FairnessStatus
    explainability_status: ExplainabilityStatus
    gate_summary: dict[str, str] = Field(
        default_factory=dict, description="Gate code -> PASS/WAIVE/BLOCK/NOT_EVALUATED."
    )
    conformity_bundle_id: str | None = None
    artifact_count: int


class UploadedAuditListResponse(BaseModel):
    model_config = _CFG
    count: int
    runs: list[UploadedAuditSummary]
    reference_case_separate: bool = Field(
        default=True,
        description="Uploaded audits are listed separately from the built-in Adult "
        "Income reference case, which is served by /api/models and /api/governance.",
    )
    notice: str = DECISION_SUPPORT_NOTICE


class UploadedAuditDetail(BaseModel):
    model_config = _CFG
    audit_run_id: str
    run_type: RunType = "uploaded_model"
    created_at: str
    model_metadata: dict[str, Any]
    dataset_metadata: dict[str, Any]
    target_configuration: dict[str, Any]
    model_checksum: str
    dataset_checksum: str
    upload_id: str
    security_acknowledged: bool
    security_warning: str
    model_capabilities: ModelCapabilities
    feature_compatibility: FeatureCompatibility
    audit_coverage: dict[str, bool]
    governance_state: GovernanceState
    policy_profile_id: str
    policy_version: str
    conformity_bundle_id: str | None = None
    artifacts: list[dict[str, Any]] = Field(
        description="Generated artefacts with the SHA-256 recorded at creation time."
    )
    limitations: list[str] = Field(default_factory=list)
    notice: str = DECISION_SUPPORT_NOTICE


class UploadedArtifactIntegrity(BaseModel):
    model_config = _CFG
    artifact: str
    path: str
    group: str
    status: Literal["verified", "changed", "missing", "not_baselined"] = Field(
        description="'not_baselined' is used only for manifest.json, which cannot "
        "record its own checksum. It is reported, not verified."
    )
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    detail: str


class UploadedIntegrityResponse(BaseModel):
    """Recomputed checksums for one uploaded run's evidence."""

    model_config = _CFG
    audit_run_id: str
    checked_at: str
    integrity_status: Literal[
        "verified", "incomplete", "modified", "modified_and_incomplete"
    ]
    integrity_ok: bool
    artifacts_checked: int
    verified_count: int
    changed_count: int
    missing_count: int
    artifacts: list[UploadedArtifactIntegrity]
    method: str
    interpretation: list[str]
    notice: str = DECISION_SUPPORT_NOTICE


class UploadedTimelineEvent(BaseModel):
    model_config = _CFG
    event_time: str
    event_type: str
    source: Literal["run", "registry", "waiver"] = Field(
        description="'run' events are derived from timestamps the artefacts carry; "
        "'registry' and 'waiver' events come from append-only logs."
    )
    detail: str


class UploadedTimelineResponse(BaseModel):
    model_config = _CFG
    audit_run_id: str
    count: int
    events: list[UploadedTimelineEvent]
    note: str
    notice: str = DECISION_SUPPORT_NOTICE


class AuditCreatedResponse(BaseModel):
    """Result of POST /api/onboarding/audits -- the only intentional write path."""

    model_config = _CFG
    audit_run_id: str
    upload_id: str
    created_at: str
    run_type: RunType = "uploaded_model"
    governance_state: GovernanceState
    fairness_status: FairnessStatus
    explainability_status: ExplainabilityStatus
    gate_summary: dict[str, str]
    conformity_bundle_id: str
    artifact_count: int
    registry_run_id: str | None = Field(
        default=None,
        description="Registry row for this audit, when registration succeeded.",
    )
    written_under: str = Field(
        description="Directory the run was written to. Always under runtime/."
    )
    warnings: list[ValidationIssue] = Field(default_factory=list)
    next_step: str
    notice: str = DECISION_SUPPORT_NOTICE
