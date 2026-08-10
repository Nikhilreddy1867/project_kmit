"""
schemas/models.py
=================
Pydantic v2 response schemas for the AI Governance Platform API.

Conventions
-----------
* Metric fields are typed ``float | None`` / ``int | None``. ``None`` means the
  audit recorded no value (e.g. a rate whose denominator was empty) -- it never
  means zero. The distinction is preserved end to end.
* Field descriptions carry the interpretation caveats from the audits, so they
  surface in the Swagger UI where a consumer will actually read them. A dashboard
  built against this API cannot claim ignorance of the 0.5 threshold or of the
  association-not-causation limit.
* ``model_config = ConfigDict(protected_namespaces=())`` is required because
  several fields legitimately begin with ``model_`` (``model``, ``model_name``),
  which Pydantic otherwise reserves.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_CFG = ConfigDict(protected_namespaces=())


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class ArtifactStatus(BaseModel):
    model_config = _CFG
    key: str = Field(description="Internal artefact identifier")
    path: str = Field(description="Repo-relative path")
    present: bool
    size_bytes: int | None = None
    modified_utc: str | None = None


class HealthResponse(BaseModel):
    model_config = _CFG
    status: Literal["ok", "degraded"] = Field(
        description="'degraded' when one or more audit artefacts are missing; "
        "the API still serves whatever is available."
    )
    service: str
    version: str
    mode: Literal["read-only"] = Field(
        description="The API never writes to data/, models/, predictions/ or results/."
    )
    artifacts_present: int
    artifacts_expected: int
    missing_artifacts: list[str]
    artifacts: list[ArtifactStatus]


# --------------------------------------------------------------------------- #
# Phase 1 - performance
# --------------------------------------------------------------------------- #
class ConfusionMatrix(BaseModel):
    model_config = _CFG
    true_negatives: int | None
    false_positives: int | None
    false_negatives: int | None
    true_positives: int | None


class PerformanceMetrics(BaseModel):
    model_config = _CFG
    accuracy: float | None = Field(
        default=None,
        description="Majority-class floor on this test set is 0.7607, so accuracy "
        "alone is close to uninformative here.",
    )
    precision: float | None = None
    recall: float | None = Field(
        default=None, description="Identical to TPR for the positive class (>50K)."
    )
    f1: float | None = Field(default=None, description="Preferred headline metric.")
    roc_auc: float | None = Field(
        default=None, description="The only metric here unaffected by the 0.5 threshold."
    )


class ModelSummary(BaseModel):
    model_config = _CFG
    model_name: str
    is_primary: bool = Field(
        description="True for the Phase 1 selected baseline (xgboost)."
    )
    metrics: PerformanceMetrics
    confusion_matrix: ConfusionMatrix
    n_test: int | None
    fit_seconds: float | None = None
    has_fairness_audit: bool
    has_explainability_audit: bool


class ModelListResponse(BaseModel):
    model_config = _CFG
    positive_class: str = ">50K"
    decision_threshold: float = Field(
        default=0.5,
        description="Inherited from Phase 1; never tuned. All rates depend on it.",
    )
    test_set_rows: int | None
    source: str = Field(description="Artefact these values were read from, verbatim.")
    count: int
    models: list[ModelSummary]


class PerformanceResponse(BaseModel):
    model_config = _CFG
    model_name: str
    is_primary: bool
    positive_class: str = ">50K"
    decision_threshold: float = 0.5
    n_test: int | None
    metrics: PerformanceMetrics
    confusion_matrix: ConfusionMatrix
    error_analysis: dict[str, Any] = Field(
        description="Counts read verbatim from the audit, plus interpretation notes. "
        "No metric is recomputed by the API."
    )
    fit_seconds: float | None = None
    source: str
    caveats: list[str]


# --------------------------------------------------------------------------- #
# Phase 2 - fairness
# --------------------------------------------------------------------------- #
class FairnessGroupMetrics(BaseModel):
    model_config = _CFG
    attribute: str
    group: str
    is_reference: bool
    reference_group: str
    n_samples: int | None
    n_actual_positive: int | None
    actual_positive_rate: float | None
    selection_rate: float | None = Field(
        default=None, description="Predicted positive rate for this group."
    )
    tpr: float | None
    fpr: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    selection_rate_ci95: float | None = Field(
        default=None, description="95% Wilson interval half-width."
    )
    tpr_ci95: float | None = None
    small_group_flag: bool | None = Field(
        default=None,
        description="True when n < 200. Do not rank these groups on point estimates.",
    )
    demographic_parity_difference: float | None
    demographic_parity_ratio: float | None
    disparate_impact_ratio: float | None = Field(
        default=None,
        description="Same formula as demographic_parity_ratio -- one measurement "
        "under two vocabularies, not two independent checks.",
    )
    fails_four_fifths_rule: bool | None = Field(
        default=None,
        description="Screening trigger from US employment-law convention. Not a "
        "statistical test and not a legal verdict.",
    )
    equal_opportunity_difference: float | None
    equalized_odds_tpr_difference: float | None
    equalized_odds_fpr_difference: float | None
    equalized_odds_max_difference: float | None


class FairnessAttributeSummary(BaseModel):
    model_config = _CFG
    attribute: str
    reference_group: str
    reference_n: int | None
    n_groups: int | None
    demographic_parity_difference_vs_reference: float | None
    demographic_parity_difference_worst_group: str | None
    demographic_parity_difference_range: float | None
    disparate_impact_ratio_min: float | None
    disparate_impact_ratio_worst_group: str | None
    groups_failing_four_fifths: int | None
    equal_opportunity_difference_max_abs: float | None
    equal_opportunity_worst_group: str | None
    equalized_odds_fpr_difference_max_abs: float | None
    equalized_odds_difference: float | None
    small_groups_present: int | None


class FairnessResponse(BaseModel):
    model_config = _CFG
    model_name: str
    positive_class: str = ">50K"
    decision_threshold: float = 0.5
    sensitive_attributes: list[str]
    reference_group_rule: str = Field(
        description="How reference groups were chosen, and what that does not imply."
    )
    summary: list[FairnessAttributeSummary]
    groups: list[FairnessGroupMetrics]
    sources: list[str]
    interpretation: dict[str, list[str]] = Field(
        description="What these metrics establish and what they explicitly do not."
    )


# --------------------------------------------------------------------------- #
# Phase 3 - explainability
# --------------------------------------------------------------------------- #
class FeatureImportance(BaseModel):
    model_config = _CFG
    rank: int | None = Field(
        default=None, description="Rank by the primary model (xgboost)."
    )
    feature: str
    importance_mean: float | None = Field(
        default=None, description="Drop in held-out ROC-AUC when the column is shuffled."
    )
    importance_std: float | None = Field(
        default=None,
        description="Std over 10 permutation repeats. Where comparable to the mean, "
        "the feature is not distinguishable from unimportant.",
    )
    importance_accuracy_mean: float | None = None
    importance_f1_mean: float | None = None
    classification: Literal["protected", "likely_proxy", "other"] = Field(
        description="Governance flag: protected attribute, likely proxy, or neither."
    )


class LocalFactor(BaseModel):
    model_config = _CFG
    feature: str
    feature_value: Any
    shap_log_odds: float | None = Field(
        default=None,
        description="Log-odds contribution. Not percentage points.",
    )
    factor_rank: int | None
    direction: str | None


class LocalExplanation(BaseModel):
    model_config = _CFG
    case_id: str = Field(
        description="Synthetic id. Dataset row indices are deliberately withheld."
    )
    actual_income: int | None
    predicted_income: int | None
    predicted_probability: float | None
    base_log_odds: float | None
    top_factors: list[LocalFactor]


class ExplainabilityResponse(BaseModel):
    model_config = _CFG
    model_name: str
    is_primary: bool
    method: dict[str, str]
    scorer: str = "roc_auc"
    n_features: int
    global_importance: list[FeatureImportance]
    local_explanations: list[LocalExplanation] = Field(
        description="Present for the primary model only; empty otherwise."
    )
    proxy_assessment: dict[str, Any]
    sources: list[str]
    caveats: list[str]


# --------------------------------------------------------------------------- #
# Phase 4 - governance
# --------------------------------------------------------------------------- #
class RiskEntry(BaseModel):
    model_config = _CFG
    risk_id: str
    category: str
    risk_statement: str
    evidence: str
    affected_groups: str
    likelihood: str
    impact: str
    overall_risk: str
    recommended_control: str
    residual_risk: str
    owner: str
    status: str


class RiskRegisterResponse(BaseModel):
    model_config = _CFG
    count: int
    total_in_register: int
    filters_applied: dict[str, str | None]
    rating_scale: dict[str, str]
    assessment_framing: str
    counts_by_overall_risk: dict[str, int]
    counts_by_status: dict[str, int]
    risks: list[RiskEntry]
    source: str


class DecisionResponse(BaseModel):
    model_config = _CFG
    subject: str
    decision_date: str
    research_use: Literal["conditionally_approved", "not_approved", "undetermined"]
    real_world_deployment: Literal["blocked", "approved", "conditional", "undetermined"]
    headline: str
    grounds_for_deployment_block: list[str]
    conditions_on_research_use: list[str]
    revisit_requirements: str
    risk_profile: dict[str, Any]
    blocking_risk_ids: list[str]
    disclaimer: str = Field(
        description="No legal or causal claim is made by this decision."
    )
    sources: list[str]
    summary_markdown: str | None = Field(
        default=None,
        description="Full governance summary; included when include_markdown=true.",
    )


class ModelCardResponse(BaseModel):
    model_config = _CFG
    subject: str
    format: Literal["markdown"] = "markdown"
    sections: list[str]
    character_count: int
    source: str
    content: str | None = Field(
        default=None, description="Omitted when sections_only=true."
    )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ErrorDetail(BaseModel):
    model_config = _CFG
    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable explanation")
    hint: str | None = Field(default=None, description="How to resolve it")
    available: list[str] | None = Field(
        default=None, description="Valid alternatives, when the input was a bad name"
    )
