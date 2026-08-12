"""
app/onboarding/audit_service.py
===============================
Runs the actual audit for an uploaded model and writes the run to ``runtime/``.

Relationship to the built-in reference case
-------------------------------------------
This module computes **new** metrics for a **new** model on a **new** dataset. It
never reads, re-derives or overwrites the Adult Income evidence, and the reference
case's numbers continue to be served verbatim from ``results/`` by
``app/services/``. The two paths share only the metric *definitions* -- deliberately
identical to ``src/fairness_audit.py``, so a disparate impact ratio means the same
thing on both sides of the platform.

Metric semantics carried over from the reference audit
------------------------------------------------------
An undefined denominator yields ``None``, never ``0.0``. "No positives in this
group, so TPR is undefined" and "TPR is zero" are different statements, and
collapsing them would manufacture a fairness gap that the data does not show. Every
such value is also named in the group's ``undefined_metrics`` list, so a reader can
see *which* quantities were undefined rather than inferring it from nulls.

What is deliberately not done
-----------------------------
* No model is trained, refitted or substituted -- ever.
* No probability is synthesised for a model without ``predict_proba``; ROC-AUC is
  reported ``null`` with a reason.
* No importance score or local explanation is invented for a model type this
  prototype cannot explain; the status becomes ``unavailable`` with the reason.
* No governance state approves deployment. The three possible states are
  ``review_required``, ``insufficient_evidence`` and ``blocked_by_policy``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.onboarding import model_validator, runtime_store as store, security
from app.onboarding.schemas import (
    DECISION_SUPPORT_NOTICE,
    FOUR_FIFTHS_NOTICE,
    AuditConfigIn,
    DatasetProfile,
    FeatureCompatibility,
    ModelCapabilities,
    ModelMetadataIn,
    ValidationIssue,
)

#: Below this group size an observed rate carries so much sampling uncertainty
#: that a difference is not distinguishable from noise. Same value as the
#: reference fairness audit, so the two read consistently.
SMALL_GROUP_THRESHOLD = 200
FOUR_FIFTHS = 0.80

#: Permutation-importance settings. Five repeats is a deliberate compromise: enough
#: for a stable ranking on a local upload, small enough that an interactive request
#: returns promptly. The count is reported in the response.
PERMUTATION_REPEATS = 5
PERMUTATION_SEED = 42
TOP_FEATURES = 15

#: Cramer's V at or above this counts as "possible proxy" for screening purposes.
#: A screening heuristic, not a causal claim -- stated as such in the output.
PROXY_ASSOCIATION_THRESHOLD = 0.30
PROXY_MAX_BINS = 10


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _safe_div(numerator: float, denominator: float) -> float | None:
    """``num/den``, or ``None`` when the denominator is empty (never ``0.0``)."""
    return float(numerator) / float(denominator) if denominator else None


def _clean(value: Any) -> Any:
    """Convert NaN/inf to ``None`` so JSON carries 'not available', not 'NaN'."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _diff(group_value: float | None, reference_value: float | None) -> float | None:
    """Difference between two rates, or ``None`` if either side is undefined."""
    if group_value is None or reference_value is None:
        return None
    return float(group_value) - float(reference_value)


def _ratio(group_value: float | None, reference_value: float | None) -> float | None:
    """Ratio of two rates, ``None`` if either is undefined or the reference is 0."""
    if group_value is None or reference_value in (None, 0):
        return None
    return float(group_value) / float(reference_value)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_inference(
    model: Any,
    features: pd.DataFrame,
    capabilities: ModelCapabilities,
    threshold: float,
    positive_class_index: int,
) -> tuple[np.ndarray, np.ndarray | None, bool, str | None]:
    """
    Score the dataset.

    Returns ``(predicted_positive, probability_or_none, threshold_applied, reason)``.

    With ``predict_proba`` the supplied threshold decides the label, which is what
    makes the threshold auditable. Without it, the model's own ``predict`` decision
    rule is used unchanged and ``threshold_applied`` is ``False`` -- the threshold is
    then reported as informational rather than silently pretended to apply.
    """
    if capabilities.has_predict_proba:
        try:
            proba = np.asarray(model.predict_proba(features), dtype=float)
            positive_proba = proba[:, positive_class_index]
            return (positive_proba >= threshold), positive_proba, True, None
        except Exception as exc:
            reason = (
                f"predict_proba failed at inference time ({type(exc).__name__}: {exc}); "
                "fell back to the model's own predict() decision rule."
            )
            labels = np.asarray(model.predict(features))
            return _labels_to_positive(labels, capabilities, positive_class_index), None, False, reason

    labels = np.asarray(model.predict(features))
    return (
        _labels_to_positive(labels, capabilities, positive_class_index),
        None,
        False,
        "The uploaded model exposes no predict_proba method, so no probability "
        "estimate exists. ROC-AUC is therefore unavailable and the decision "
        "threshold was not applied -- the model's own decision rule was used.",
    )


def _labels_to_positive(
    labels: np.ndarray, capabilities: ModelCapabilities, positive_class_index: int
) -> np.ndarray:
    """Map predicted class labels onto a positive/negative boolean array."""
    positive_label = capabilities.classes[positive_class_index]
    return np.asarray([str(v).strip() == positive_label for v in labels], dtype=bool)


def resolve_positive_class_index(
    capabilities: ModelCapabilities, positive_class: str
) -> tuple[int, list[str]]:
    """
    Locate the user's positive class among the model's own ``classes_``.

    Matching on the model's classes rather than on the dataset's labels is what
    guarantees the probability column and the label mapping refer to the same class.
    A mismatch is an error rather than a guess: assuming ``classes_[1]`` is the
    positive class is right for the common case and silently wrong for the rest.
    """
    wanted = str(positive_class).strip()
    classes = list(capabilities.classes)
    if wanted in classes:
        index = classes.index(wanted)
        return index, [c for c in classes if c != wanted]
    raise model_validator.ModelValidationError(
        "positive_class_not_in_model",
        f"The positive class '{wanted}' is not one of the model's classes "
        f"({', '.join(classes)}).",
        hint="The positive class must match a label the model was fitted on, so the "
        "probability column and the predicted label refer to the same class.",
    )


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def compute_performance(
    actual_positive: np.ndarray,
    predicted_positive: np.ndarray,
    probability: np.ndarray | None,
    threshold: float,
    threshold_applied: bool,
    positive_class: str,
    negative_classes: list[str],
    proba_reason: str | None,
) -> dict[str, Any]:
    """
    Held-out metrics from the confusion matrix.

    Derived from the four cell counts with :func:`_safe_div` rather than from
    sklearn's scorers, so a degenerate case (no predicted positives, no actual
    positives) yields ``None`` instead of sklearn's ``0.0`` + warning. ROC-AUC is
    the exception: it is threshold-free and comes from sklearn, and it is omitted
    entirely when no probability exists.
    """
    true_positives = int(np.sum(actual_positive & predicted_positive))
    false_positives = int(np.sum(~actual_positive & predicted_positive))
    false_negatives = int(np.sum(actual_positive & ~predicted_positive))
    true_negatives = int(np.sum(~actual_positive & ~predicted_positive))
    total = true_positives + false_positives + false_negatives + true_negatives

    precision = _safe_div(true_positives, true_positives + false_positives)
    recall = _safe_div(true_positives, true_positives + false_negatives)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )

    roc_auc: float | None = None
    roc_reason: str | None = proba_reason
    if probability is not None:
        if len(np.unique(actual_positive)) < 2:
            roc_reason = (
                "ROC-AUC is undefined when the test set contains only one actual "
                "class; it was not estimated."
            )
        else:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(actual_positive.astype(int), probability))
            roc_reason = None

    caveats = [
        f"All rates except ROC-AUC depend on the decision threshold ({threshold:g}).",
        "Metrics describe this uploaded test set only. If it is not representative "
        "of the population the model would be used on, they do not transfer.",
        "A single held-out evaluation carries sampling uncertainty that is not "
        "quantified here.",
    ]
    if not threshold_applied:
        caveats.append(
            "The threshold was NOT applied: this model exposes no predict_proba, so "
            "its own decision rule produced the labels."
        )
    majority = max(
        int(np.sum(actual_positive)), int(np.sum(~actual_positive))
    )
    floor = _safe_div(majority, total)
    if floor is not None:
        caveats.append(
            f"Majority-class accuracy floor on this dataset is {floor:.4f}; accuracy "
            "above that is the only accuracy worth reading."
        )

    return {
        "n_samples": total,
        "positive_class": positive_class,
        "negative_classes": negative_classes,
        "decision_threshold": float(threshold),
        "threshold_applied": bool(threshold_applied),
        "accuracy": _clean(_safe_div(true_positives + true_negatives, total)),
        "precision": _clean(precision),
        "recall": _clean(recall),
        "f1": _clean(f1),
        "roc_auc": _clean(roc_auc),
        "roc_auc_unavailable_reason": roc_reason,
        "confusion_matrix": {
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
        },
        "computed_from": "runtime/audits/<audit_run_id>/predictions.csv",
        "caveats": caveats,
        "notice": DECISION_SUPPORT_NOTICE,
    }


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def _group_metrics(
    actual_positive: np.ndarray, predicted_positive: np.ndarray
) -> tuple[dict[str, float | None], list[str]]:
    """Per-group rates, plus the names of the ones whose denominator was empty."""
    true_positives = int(np.sum(actual_positive & predicted_positive))
    false_positives = int(np.sum(~actual_positive & predicted_positive))
    false_negatives = int(np.sum(actual_positive & ~predicted_positive))
    true_negatives = int(np.sum(~actual_positive & ~predicted_positive))
    n = len(actual_positive)

    true_positive_rate = _safe_div(true_positives, true_positives + false_negatives)
    false_positive_rate = _safe_div(false_positives, false_positives + true_negatives)
    precision = _safe_div(true_positives, true_positives + false_positives)
    f1 = (
        _safe_div(2 * precision * true_positive_rate, precision + true_positive_rate)
        if precision is not None and true_positive_rate is not None
        else None
    )

    metrics: dict[str, float | None] = {
        "actual_positive_rate": _safe_div(true_positives + false_negatives, n),
        "selection_rate": _safe_div(true_positives + false_positives, n),
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "recall": true_positive_rate,  # identical for the positive class
        "f1": f1,
    }
    undefined = [name for name, value in metrics.items() if value is None]
    return metrics, undefined


def compute_fairness(
    frame: pd.DataFrame,
    actual_positive: np.ndarray,
    predicted_positive: np.ndarray,
    sensitive_columns: list[str],
) -> dict[str, Any]:
    """
    Group metrics and disparity measures for the user's chosen columns only.

    No sensitive columns selected is **not** a pass: the status becomes
    ``not_provided_by_user`` and every downstream consumer (gates, governance
    summary, dashboard) treats it as absent evidence rather than as a clean result.
    """
    interpretation = [
        "These are outcome differences measured on the uploaded test set. They are "
        "not findings of discrimination and they identify no causal mechanism.",
        FOUR_FIFTHS_NOTICE,
        "Fairness criteria conflict mathematically: except in degenerate cases, no "
        "classifier can equalise selection rate, TPR and precision simultaneously. "
        "Which criterion matters is a normative choice this platform does not make.",
        f"Groups smaller than {SMALL_GROUP_THRESHOLD} rows are flagged: their rates "
        "carry wide sampling uncertainty, so a difference may be noise.",
        "An undefined rate (empty denominator) is reported as null. It is never "
        "reported as zero.",
    ]

    if not sensitive_columns:
        return {
            "status": "not_provided_by_user",
            "status_detail": (
                "No sensitive columns were selected, so no fairness assessment was "
                "performed. This is an absence of evidence -- not a pass, and not a "
                "claim that the model is fair."
            ),
            "sensitive_columns_requested": [],
            "small_group_threshold": SMALL_GROUP_THRESHOLD,
            "attributes": [],
            "groups": [],
            "interpretation": interpretation,
            "four_fifths_notice": FOUR_FIFTHS_NOTICE,
            "notice": DECISION_SUPPORT_NOTICE,
        }

    attributes: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []

    for column in sensitive_columns:
        values = frame[column].astype("string").fillna("(missing)").str.strip()
        counts = values.value_counts()
        if counts.empty:
            continue

        # Reference = largest group. Ties broken by name so the choice is stable.
        top = counts[counts == counts.max()].index.tolist()
        reference = sorted(str(t) for t in top)[0]
        reference_mask = (values == reference).to_numpy()
        reference_metrics, _ = _group_metrics(
            actual_positive[reference_mask], predicted_positive[reference_mask]
        )

        group_rows: list[dict[str, Any]] = []
        for group in [str(g) for g in counts.index.tolist()]:
            mask = (values == group).to_numpy()
            metrics, undefined = _group_metrics(
                actual_positive[mask], predicted_positive[mask]
            )
            ratio = _ratio(metrics["selection_rate"], reference_metrics["selection_rate"])
            row = {
                "attribute": column,
                "group": group,
                "n": int(mask.sum()),
                "is_reference": group == reference,
                "small_group": bool(mask.sum() < SMALL_GROUP_THRESHOLD),
                **{k: _clean(v) for k, v in metrics.items()},
                "demographic_parity_difference": _clean(
                    _diff(metrics["selection_rate"], reference_metrics["selection_rate"])
                ),
                "demographic_parity_ratio": _clean(ratio),
                # Identical quantity to the demographic parity ratio; named
                # separately because the four-fifths screen is stated in terms of
                # "disparate impact" and readers look for that label.
                "disparate_impact_ratio": _clean(ratio),
                "equal_opportunity_difference": _clean(
                    _diff(
                        metrics["true_positive_rate"],
                        reference_metrics["true_positive_rate"],
                    )
                ),
                "equalized_odds_tpr_difference": _clean(
                    _diff(
                        metrics["true_positive_rate"],
                        reference_metrics["true_positive_rate"],
                    )
                ),
                "equalized_odds_fpr_difference": _clean(
                    _diff(
                        metrics["false_positive_rate"],
                        reference_metrics["false_positive_rate"],
                    )
                ),
                "passes_four_fifths_screen": (
                    None if ratio is None else bool(ratio >= FOUR_FIFTHS)
                ),
                "undefined_metrics": undefined,
            }
            group_rows.append(row)

        ratios = [
            (r["group"], r["disparate_impact_ratio"])
            for r in group_rows
            if r["disparate_impact_ratio"] is not None
        ]
        worst_group, min_ratio = (min(ratios, key=lambda p: p[1]) if ratios else (None, None))
        equal_opportunity = [
            abs(r["equal_opportunity_difference"])
            for r in group_rows
            if r["equal_opportunity_difference"] is not None
        ]
        fpr_gaps = [
            abs(r["equalized_odds_fpr_difference"])
            for r in group_rows
            if r["equalized_odds_fpr_difference"] is not None
        ]

        attributes.append(
            {
                "attribute": column,
                "reference_group": reference,
                "reference_n": int(reference_mask.sum()),
                "n_groups": len(group_rows),
                "min_disparate_impact_ratio": _clean(min_ratio),
                "worst_group": worst_group,
                "groups_failing_four_fifths": [
                    r["group"]
                    for r in group_rows
                    if r["passes_four_fifths_screen"] is False
                ],
                "max_abs_equal_opportunity_difference": _clean(
                    max(equal_opportunity) if equal_opportunity else None
                ),
                "max_abs_equalized_odds_fpr_difference": _clean(
                    max(fpr_gaps) if fpr_gaps else None
                ),
                "small_groups_present": any(r["small_group"] for r in group_rows),
                "undefined_metric_count": sum(
                    len(r["undefined_metrics"]) for r in group_rows
                ),
            }
        )
        all_groups.extend(group_rows)

    return {
        "status": "available" if attributes else "unavailable",
        "status_detail": (
            f"Assessed {len(attributes)} sensitive attribute(s) selected by the user."
            if attributes
            else "The selected sensitive columns produced no usable groups."
        ),
        "sensitive_columns_requested": list(sensitive_columns),
        "reference_group_rule": "Largest group by sample count.",
        "small_group_threshold": SMALL_GROUP_THRESHOLD,
        "attributes": attributes,
        "groups": all_groups,
        "interpretation": interpretation,
        "four_fifths_notice": FOUR_FIFTHS_NOTICE,
        "notice": DECISION_SUPPORT_NOTICE,
    }


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    """
    Cramer's V between two categorical series -- a bounded [0, 1] association.

    Numeric columns are quantile-binned first. Used only as a **screening** measure
    for possible proxies: a high value says the two columns carry overlapping
    information, not that one causes or stands in for the other.
    """
    from scipy.stats import chi2_contingency  # provided by scikit-learn's dependency

    def _as_categorical(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            try:
                return pd.qcut(series, q=PROXY_MAX_BINS, duplicates="drop").astype("string")
            except (ValueError, TypeError):
                return series.astype("string")
        return series.astype("string")

    a = _as_categorical(left).fillna("(missing)")
    b = _as_categorical(right).fillna("(missing)")
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    try:
        chi2 = chi2_contingency(table.to_numpy())[0]
    except ValueError:
        return None
    n = int(table.to_numpy().sum())
    if n == 0:
        return None
    denominator = n * (min(table.shape) - 1)
    return float(math.sqrt(chi2 / denominator)) if denominator else None


def compute_explainability(
    model: Any,
    features: pd.DataFrame,
    actual_positive: np.ndarray,
    capabilities: ModelCapabilities,
    sensitive_columns: list[str],
    full_frame: pd.DataFrame,
    actual_labels: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Permutation importance over the **original** input columns, plus proxy screening.

    Importance is computed on the dataframe as handed to the model, so the features
    named are the user's own column names rather than one-hot fragments invented by
    a preprocessor inside the pipeline.

    Returns ``status: unavailable`` with a reason -- and no numbers at all -- for any
    model this prototype cannot explain. Nothing is estimated to fill the gap.
    """
    caveats = [
        "Permutation importance is associational. A high score means the model's "
        "predictions depend on that column, not that the column causes the outcome.",
        "Correlated columns share importance: permuting one while its correlate stays "
        "intact understates both. A low score is therefore not evidence that a column "
        "is unused.",
        "Importance is measured against this uploaded dataset only.",
        "A sensitive attribute scoring low is NOT evidence of fairness -- the "
        "information can reach the model through proxy columns.",
    ]

    if not capabilities.supports_permutation_importance:
        return {
            "status": "unavailable",
            "status_detail": (
                "Explainability not available for this model type in the current "
                f"local prototype: {capabilities.estimator_type} exposes no "
                "predict(X), so permutation importance cannot be computed."
            ),
            "global_importance": [],
            "local_explanations": [],
            "proxy_assessment": [],
            "caveats": caveats,
            "notice": DECISION_SUPPORT_NOTICE,
        }

    # The label vector has to be in the same space as the thing the scorer compares
    # it against. `roc_auc` scores predict_proba output, so it needs the 0/1
    # indicator; `accuracy` scores predict() output, which returns the estimator's
    # own class labels -- strings, for a model fitted on '>50K'/'<=50K'. Handing 0/1
    # to an accuracy scorer raises "Mix of label input types" and would leave every
    # predict-only model reported as unexplainable when in fact it is explainable.
    scorer = "roc_auc" if capabilities.has_predict_proba else "accuracy"
    if scorer == "accuracy" and actual_labels is not None:
        target = np.asarray(actual_labels)
    else:
        target = actual_positive.astype(int)

    try:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            features,
            target,
            n_repeats=PERMUTATION_REPEATS,
            random_state=PERMUTATION_SEED,
            scoring=scorer,
            n_jobs=1,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "status_detail": (
                "Explainability not available for this model type in the current "
                f"local prototype: permutation importance failed "
                f"({type(exc).__name__}: {exc}). No importance scores were estimated."
            ),
            "global_importance": [],
            "local_explanations": [],
            "proxy_assessment": [],
            "caveats": caveats,
            "notice": DECISION_SUPPORT_NOTICE,
        }

    order = np.argsort(result.importances_mean)[::-1]
    columns = list(features.columns)

    # -- proxy screening ---------------------------------------------------- #
    proxy_map: dict[str, list[str]] = {}
    proxy_notes: list[str] = []
    for column in columns:
        associations: list[str] = []
        for sensitive in sensitive_columns:
            if sensitive == column or sensitive not in full_frame.columns:
                continue
            value = _cramers_v(full_frame[column], full_frame[sensitive])
            if value is not None and value >= PROXY_ASSOCIATION_THRESHOLD:
                associations.append(f"{sensitive} (Cramer's V {value:.2f})")
        if associations:
            proxy_map[column] = associations

    importance_rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order[:TOP_FEATURES], start=1):
        name = columns[index]
        importance_rows.append(
            {
                "rank": rank,
                "feature": name,
                "importance_mean": float(result.importances_mean[index]),
                "importance_std": float(result.importances_std[index]),
                "is_selected_sensitive_column": name in sensitive_columns,
                "possible_proxy_for": proxy_map.get(name, []),
            }
        )

    if sensitive_columns:
        flagged = [row for row in importance_rows if row["possible_proxy_for"]]
        if flagged:
            proxy_notes.append(
                "Possible proxy columns among the top features (association screening, "
                f"Cramer's V >= {PROXY_ASSOCIATION_THRESHOLD:.2f}): "
                + "; ".join(
                    f"{row['feature']} -> {', '.join(row['possible_proxy_for'])}"
                    for row in flagged
                )
                + "."
            )
            proxy_notes.append(
                "Association is not proof that the model uses these columns as "
                "substitutes for the sensitive attribute, and it establishes no "
                "causal mechanism. It marks where to look."
            )
        else:
            proxy_notes.append(
                "No top-ranked feature reached the association threshold with a "
                "selected sensitive column. Absence of a flagged proxy is not "
                "evidence that no proxy exists -- this screen only detects pairwise "
                "association above the threshold, not combinations of columns."
            )
        present = [row for row in importance_rows if row["is_selected_sensitive_column"]]
        if present:
            proxy_notes.append(
                "Selected sensitive column(s) are themselves model inputs and appear "
                "in the ranking: "
                + ", ".join(f"{r['feature']} (rank {r['rank']})" for r in present)
                + ". Removing them would not by itself remove the information, which "
                "can still arrive through correlated columns."
            )
    else:
        proxy_notes.append(
            "No sensitive columns were selected, so no proxy screening was performed."
        )

    payload: dict[str, Any] = {
        "status": "available",
        "status_detail": (
            f"Permutation importance over {len(columns)} original input column(s), "
            f"{PERMUTATION_REPEATS} repeats, scored by {scorer}."
        ),
        "method": "permutation_importance (sklearn.inspection) on original input columns",
        "n_repeats": PERMUTATION_REPEATS,
        "scorer": scorer,
        "global_importance": importance_rows,
        "local_method": None,
        "local_explanations": [],
        "proxy_assessment": proxy_notes,
        "caveats": caveats,
        "notice": DECISION_SUPPORT_NOTICE,
    }

    local = _try_treeshap(model, features, capabilities)
    if local["explanations"]:
        payload["local_method"] = local["method"]
        payload["local_explanations"] = local["explanations"]
    elif local["reason"]:
        payload["caveats"] = caveats + [local["reason"]]
    return payload


def _try_treeshap(
    model: Any, features: pd.DataFrame, capabilities: ModelCapabilities
) -> dict[str, Any]:
    """
    Exact TreeSHAP for XGBoost pipelines, with the additivity check enforced.

    Attempted only when the final estimator is an XGBoost classifier. The additivity
    identity (contributions + base value == raw margin) is verified before anything
    is returned; if it does not hold, the explanation is discarded rather than
    reported, because a SHAP decomposition that does not reconstruct the prediction
    is not an explanation of it.

    Every failure path returns a stated reason and no numbers.
    """
    if not capabilities.supports_treeshap:
        return {
            "method": None,
            "explanations": [],
            "reason": (
                "Local TreeSHAP explanations are not available: exact TreeSHAP is "
                f"attempted only for XGBoost classifiers, and the final estimator is "
                f"{capabilities.final_estimator}. No local explanation was estimated "
                "by another method."
            ),
        }

    try:
        import xgboost as xgb

        booster_owner = model
        transformed = features
        steps = getattr(model, "steps", None)
        if steps:
            booster_owner = steps[-1][1]
            # Apply every step except the final estimator to reach the matrix the
            # booster actually scores.
            from sklearn.pipeline import Pipeline

            if len(steps) > 1:
                transformed = Pipeline(steps[:-1]).transform(features)

        matrix = xgb.DMatrix(transformed)
        booster = booster_owner.get_booster()
        contributions = np.asarray(
            booster.predict(matrix, pred_contribs=True), dtype=float
        )
        margins = np.asarray(
            booster.predict(matrix, output_margin=True), dtype=float
        )

        reconstructed = contributions.sum(axis=1)
        max_error = float(np.max(np.abs(reconstructed - margins)))
        if not math.isfinite(max_error) or max_error > 1e-3:
            return {
                "method": None,
                "explanations": [],
                "reason": (
                    "Local TreeSHAP explanations were discarded: the additivity check "
                    f"failed (max reconstruction error {max_error:.3e} exceeds 1e-3). "
                    "A decomposition that does not reconstruct the model's own output "
                    "is not reported as an explanation."
                ),
            }
    except Exception as exc:
        return {
            "method": None,
            "explanations": [],
            "reason": (
                f"Local TreeSHAP explanations are not available: {type(exc).__name__}: "
                f"{exc}. No local explanation was estimated by another method."
            ),
        }

    return {
        "method": (
            "exact TreeSHAP via xgboost pred_contribs, additivity verified "
            f"(max error {max_error:.2e}). Contributions are in log-odds, not "
            "percentage points."
        ),
        "explanations": [],
        "reason": None,
        "_contributions": contributions,
        "_base_values": contributions[:, -1],
    }


# --------------------------------------------------------------------------- #
# Risk summary
# --------------------------------------------------------------------------- #
def build_risk_summary(
    performance: dict[str, Any],
    fairness: dict[str, Any],
    explainability: dict[str, Any],
    capabilities: ModelCapabilities,
    metadata: ModelMetadataIn,
    config: AuditConfigIn,
    dataset: DatasetProfile,
) -> dict[str, Any]:
    """
    Deterministic risk items derived only from what the audit actually observed.

    Rule-based throughout: identical inputs produce an identical list, in the same
    order. Severity is a presentation-level triage aid, not a calibrated
    probability, and every item names its own limitation.
    """
    risks: list[dict[str, Any]] = []

    def add(
        risk_id: str,
        category: str,
        severity: str,
        statement: str,
        evidence: str,
        limitation: str,
        action: str,
    ) -> None:
        risks.append(
            {
                "risk_id": risk_id,
                "category": category,
                "severity": severity,
                "statement": statement,
                "evidence": evidence,
                "limitation": limitation,
                "recommended_action": action,
            }
        )

    # -- UR-01: deployment status ------------------------------------------- #
    add(
        "UR-01",
        "Governance",
        "critical",
        "This uploaded model has no deployment authorisation from this platform. "
        "The audit produces evidence for human review; it does not approve use.",
        "Platform design: uploaded runs can only reach review_required, "
        "insufficient_evidence or blocked_by_policy.",
        "Absence of authorisation is a statement about this platform, not a "
        "judgement about the model's legality or quality.",
        "Route the Conformity Bundle to the accountable owner "
        f"({metadata.model_owner}) for a documented human decision.",
    )

    # -- UR-02: probability availability ------------------------------------ #
    if not capabilities.has_predict_proba:
        add(
            "UR-02",
            "Evidence completeness",
            "high",
            "The model exposes no predict_proba, so ROC-AUC is unavailable and the "
            "decision threshold could not be applied or varied.",
            "Model inspection: predict_proba absent; performance.roc_auc is null.",
            "This limits the audit; it is not itself a defect in the model. Some "
            "estimators are deliberately hard-label only.",
            "Supply a probability-capable estimator if threshold-sensitivity or "
            "ranking quality needs to be assessed.",
        )

    # -- UR-03: fairness evidence ------------------------------------------- #
    if fairness.get("status") == "not_provided_by_user":
        add(
            "UR-03",
            "Fairness",
            "high",
            "No fairness evidence exists for this run because no sensitive columns "
            "were selected. Nothing here indicates the model is fair.",
            "Audit configuration: sensitive_columns was empty.",
            "This is an absence of evidence, not evidence of absence of disparity.",
            "Re-run the audit selecting the sensitive attributes relevant to the "
            "stated decision context, where such data is lawfully available.",
        )
    else:
        failing = [
            (attribute["attribute"], attribute["groups_failing_four_fifths"])
            for attribute in fairness.get("attributes", [])
            if attribute.get("groups_failing_four_fifths")
        ]
        if failing:
            add(
                "UR-03",
                "Fairness",
                "high",
                "One or more groups fall below the four-fifths screening threshold: "
                + "; ".join(
                    f"{attribute}: {', '.join(groups)}" for attribute, groups in failing
                )
                + ".",
                "fairness.json, disparate_impact_ratio per group against the largest "
                "group.",
                FOUR_FIFTHS_NOTICE,
                "Have a human reviewer examine these groups, the base rates behind "
                "them, and whether the threshold and features are appropriate for the "
                "stated decision context.",
            )
        small = [
            attribute["attribute"]
            for attribute in fairness.get("attributes", [])
            if attribute.get("small_groups_present")
        ]
        if small:
            add(
                "UR-04",
                "Statistical uncertainty",
                "medium",
                f"Group(s) below {SMALL_GROUP_THRESHOLD} rows are present for: "
                + ", ".join(small)
                + ". Their rates carry wide sampling uncertainty.",
                "fairness.json, per-group sample counts.",
                "Small-group differences may be sampling noise rather than a "
                "systematic pattern.",
                "Do not act on small-group differences without a larger sample or an "
                "explicit uncertainty analysis.",
            )
        undefined = sum(
            int(attribute.get("undefined_metric_count") or 0)
            for attribute in fairness.get("attributes", [])
        )
        if undefined:
            add(
                "UR-05",
                "Evidence completeness",
                "medium",
                f"{undefined} group metric value(s) were undefined (empty "
                "denominator) and are reported as null.",
                "fairness.json, per-group undefined_metrics.",
                "Null means not computable. It was not replaced with zero, which "
                "would have created an artificial disparity.",
                "Interpret those cells as 'no data', not as a measured zero.",
            )

    # -- UR-06: explainability ---------------------------------------------- #
    if explainability.get("status") != "available":
        add(
            "UR-06",
            "Transparency",
            "medium",
            "No explainability evidence is available for this model in this "
            "prototype, so the basis of its predictions is undocumented here.",
            f"explainability.json status = {explainability.get('status')}.",
            "The limitation is the prototype's, not necessarily the model's.",
            "Provide a model type this prototype can explain, or supply external "
            "explainability evidence to the reviewer.",
        )
    else:
        flagged = [
            row
            for row in explainability.get("global_importance", [])
            if row.get("possible_proxy_for")
        ]
        if flagged:
            add(
                "UR-07",
                "Fairness",
                "medium",
                f"{len(flagged)} top-ranked feature(s) are associated with a selected "
                "sensitive attribute and could carry that information into the model.",
                "explainability.json, possible_proxy_for (Cramer's V screening).",
                "Association is not proof of proxy use and establishes no causal "
                "mechanism. Removing the sensitive column would not remove this path.",
                "Review whether each flagged feature is justified for the stated "
                "decision context.",
            )

    # -- UR-08: representativeness ------------------------------------------ #
    add(
        "UR-08",
        "Validity",
        "medium",
        "All metrics describe the single uploaded dataset "
        f"({dataset.row_count} rows). Nothing establishes that it represents the "
        "population implied by the stated decision context.",
        f"uploaded_dataset_metadata.json; decision context: {config.target_column} "
        f"with positive class '{config.positive_class}'.",
        "Representativeness cannot be assessed from the file alone.",
        "Document the sampling frame and how it relates to the intended population.",
    )

    # -- UR-09: threshold ---------------------------------------------------- #
    if performance.get("threshold_applied"):
        add(
            "UR-09",
            "Validity",
            "low",
            f"Every rate except ROC-AUC depends on the "
            f"{performance.get('decision_threshold')} threshold, which this audit "
            "took as given rather than tuning.",
            "performance.json decision_threshold.",
            "A different threshold would produce different rates and different "
            "fairness gaps.",
            "Choose the threshold from the decision context's relative cost of false "
            "positives and false negatives, and record that reasoning.",
        )

    # -- UR-10: monitoring --------------------------------------------------- #
    add(
        "UR-10",
        "Operations",
        "medium",
        "No monitoring, drift-detection, incident-response or rollback evidence was "
        "supplied with this upload, so the Operations Gate cannot be evaluated.",
        "Intake scope: this prototype accepts a model and a dataset only.",
        "Not evaluated means not assessed -- it is neither a pass nor a failure.",
        "Supply operational readiness evidence before any real-world use is "
        "considered.",
    )

    counts: dict[str, int] = {}
    for risk in risks:
        counts[risk["severity"]] = counts.get(risk["severity"], 0) + 1

    return {
        "risks": risks,
        "severity_counts": counts,
        "total": len(risks),
        "framing": (
            "Deterministic, rule-based triage of what this audit observed. Severity is "
            "a qualitative reading aid, not a calibrated probability, and no item is a "
            "legal finding."
        ),
        "notice": DECISION_SUPPORT_NOTICE,
    }


# --------------------------------------------------------------------------- #
# Predictions artefact
# --------------------------------------------------------------------------- #
def build_predictions_frame(
    frame: pd.DataFrame,
    config: AuditConfigIn,
    actual_positive: np.ndarray,
    predicted_positive: np.ndarray,
    probability: np.ndarray | None,
    positive_class: str,
    negative_label: str,
) -> pd.DataFrame:
    """
    The per-row prediction record, mirroring the reference case's export.

    Selected sensitive columns are carried through so the fairness numbers can be
    re-derived from this file by a reviewer -- the same reason the Phase 1 exports
    keep ``sex`` and ``race``. No other raw column is copied, which keeps the
    exported record to what the audit actually needs.
    """
    output = pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "actual_label": [
                positive_class if flag else negative_label for flag in actual_positive
            ],
            "predicted_label": [
                positive_class if flag else negative_label for flag in predicted_positive
            ],
            "actual_positive": actual_positive.astype(int),
            "predicted_positive": predicted_positive.astype(int),
        }
    )
    output["predicted_probability"] = (
        probability if probability is not None else pd.NA
    )
    for column in config.sensitive_columns:
        if column in frame.columns:
            output[column] = frame[column].to_numpy()
    return output


# --------------------------------------------------------------------------- #
# Validation entry point (shared by both POST endpoints)
# --------------------------------------------------------------------------- #
def validate_submission(
    model_path: Path,
    frame: pd.DataFrame | None,
    dataset: DatasetProfile | None,
    config: AuditConfigIn,
    *,
    security_acknowledged: bool,
    issues: list[ValidationIssue],
) -> tuple[ModelCapabilities | None, FeatureCompatibility | None, Any, list[ValidationIssue]]:
    """
    Load and inspect the model, then check it against the dataset.

    Called only after the dataset has been validated, and it returns immediately
    without loading anything if the dataset failed -- which is what keeps an
    unauditable submission from ever being unpickled.
    """
    issues = list(issues)
    if frame is None or dataset is None:
        return None, None, None, issues

    try:
        security.require_acknowledgement(security_acknowledged)
    except security.AcknowledgementRequired as exc:
        issues.append(
            ValidationIssue(
                code=exc.code,
                severity="error",
                field="security_acknowledged",
                message=exc.message,
                hint=exc.hint,
            )
        )
        return None, None, None, issues

    try:
        model = model_validator.load_model(
            model_path, security_acknowledged=security_acknowledged
        )
    except model_validator.ModelValidationError as exc:
        issues.append(
            ValidationIssue(
                code=exc.code, severity="error", field="model_file",
                message=exc.message, hint=exc.hint,
            )
        )
        return None, None, None, issues

    capabilities = model_validator.inspect_model(model)
    for code, message, hint in model_validator.validate_capabilities(capabilities):
        issues.append(
            ValidationIssue(
                code=code, severity="error", field="model_file",
                message=message, hint=hint,
            )
        )

    compatibility = model_validator.check_feature_compatibility(
        capabilities, frame, config.target_column, config.sensitive_columns
    )
    if not compatibility.compatible:
        issues.append(
            ValidationIssue(
                code="feature_mismatch",
                severity="error",
                field="dataset",
                message=(
                    "The dataset does not provide every feature the model expects. "
                    "Missing: " + ", ".join(compatibility.missing_features[:25]) + "."
                ),
                hint=(
                    "Unexpected columns are not fatal (they are dropped for inference "
                    "and kept for fairness reporting), but missing ones are. "
                    f"Check method: {compatibility.method}."
                ),
            )
        )
    if compatibility.unexpected_features:
        issues.append(
            ValidationIssue(
                code="unexpected_features",
                severity="warning",
                field="dataset",
                message=(
                    f"{len(compatibility.unexpected_features)} dataset column(s) are "
                    "not model inputs and will be dropped for inference: "
                    + ", ".join(compatibility.unexpected_features[:15])
                    + "."
                ),
                hint="Selected sensitive columns are retained for fairness reporting "
                "regardless of whether the model uses them.",
            )
        )

    if capabilities.is_binary_classifier:
        try:
            resolve_positive_class_index(capabilities, config.positive_class)
        except model_validator.ModelValidationError as exc:
            issues.append(
                ValidationIssue(
                    code=exc.code, severity="error", field="positive_class",
                    message=exc.message, hint=exc.hint,
                )
            )

    # A capability the model does not have is stated as a warning as well as in the
    # structured capability map. The map is what a client should branch on, but a user
    # reading the issue list is the one who has to decide whether to proceed, and
    # "your Validation Gate will have an un-evaluated control" is not something they
    # should have to infer from a false flag two fields away. The absent-fairness case
    # is warned about in the same way when no sensitive column is selected.
    if not capabilities.has_predict_proba:
        issues.append(
            ValidationIssue(
                code="no_predict_proba",
                severity="warning",
                field="model_file",
                message=(
                    "The model does not implement predict_proba, so no probability "
                    "score exists. ROC-AUC cannot be computed and no probability will "
                    "be synthesised; the decision threshold is not applied, and "
                    "predict() is used directly."
                ),
                hint=(
                    "The Validation Gate's ROC-AUC control is recorded as "
                    "NOT_EVALUATED rather than passed. Local SHAP explanations are "
                    "also unavailable without a score."
                ),
            )
        )
    if not capabilities.supports_permutation_importance:
        issues.append(
            ValidationIssue(
                code="no_global_explainability",
                severity="warning",
                field="model_file",
                message=(
                    "Global feature importance cannot be computed for this model in "
                    "the current local prototype. No importance score will be "
                    "invented in its place."
                ),
                hint="Explainability is reported as unavailable, which is not a pass.",
            )
        )

    return capabilities, compatibility, model, issues


def audit_capability_map(
    capabilities: ModelCapabilities | None, config: AuditConfigIn
) -> dict[str, bool]:
    """Which audits can run, stated honestly rather than optimistically."""
    if capabilities is None:
        return {
            "performance": False,
            "roc_auc": False,
            "fairness": False,
            "explainability_global": False,
            "explainability_local_shap": False,
        }
    return {
        "performance": capabilities.has_predict,
        "roc_auc": capabilities.has_predict_proba,
        "fairness": bool(config.sensitive_columns),
        "explainability_global": capabilities.supports_permutation_importance,
        "explainability_local_shap": capabilities.supports_treeshap,
    }


def validate_upload(
    config: AuditConfigIn,
    upload_manifest: dict[str, Any],
    *,
    security_acknowledged: bool,
) -> dict[str, Any]:
    """
    Run every check an audit would run, and write nothing beyond the stored upload.

    This is a dry run of :func:`create_audit`'s validation phase, so a user can fix a
    mismatched target column or a missing feature before an audit run exists. It
    returns the same issue list the audit would raise, plus the honest capability map,
    so nothing about what *can* be audited comes as a surprise later.

    The model is still loaded here -- deserialising is unavoidable to inspect what it
    supports -- which is exactly why the acknowledgement is checked first and the load
    happens only after the dataset has passed.
    """
    from app.onboarding import upload_service as uploads

    model_path, dataset_path = uploads.upload_paths(upload_manifest)
    dataset, frame, issues = uploads.validate_dataset(
        dataset_path.read_bytes(),
        config.target_column,
        config.positive_class,
        config.sensitive_columns,
    )
    capabilities, compatibility, _model, issues = validate_submission(
        model_path,
        frame,
        dataset,
        config,
        security_acknowledged=security_acknowledged,
        issues=issues,
    )

    valid = not any(issue.severity == "error" for issue in issues)
    if valid:
        next_step = (
            "POST /api/onboarding/audits with the same upload_id and configuration to "
            "create the audit run. No audit run exists yet."
        )
    else:
        next_step = (
            "Fix the reported errors and validate again. Nothing was audited and no "
            "audit run was created."
        )

    return {
        "valid": valid,
        "upload_id": str(upload_manifest["upload_id"]),
        "security_acknowledged": bool(security_acknowledged),
        "security_warning": security.JOBLIB_SECURITY_WARNING,
        "production_hardening": security.PRODUCTION_HARDENING_NOTE,
        "issues": issues,
        "dataset": dataset,
        "model_capabilities": capabilities,
        "feature_compatibility": compatibility,
        "audit_capabilities": audit_capability_map(capabilities, config),
        "next_step": next_step,
    }


# --------------------------------------------------------------------------- #
# Audit run creation
# --------------------------------------------------------------------------- #
class AuditRejected(Exception):
    """
    Validation failed, so no audit run was created and nothing was written.

    Carries every issue rather than only the first, so one round trip tells the user
    everything they need to fix.
    """

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__(
            "The submission was rejected: "
            + "; ".join(i.message for i in issues if i.severity == "error")
        )


def audit_run_identifier(
    upload_id: str, model_sha: str, dataset_sha: str, config: AuditConfigIn, policy_id: str
) -> str:
    """
    Derive the audit-run id from the submission's identity.

    Content-addressed over the upload id, both file checksums and the full audit
    configuration. Because ``upload_id`` is a fresh UUID per submission, each audit
    gets its own id -- but the *configuration* is part of the hash too, which means the
    id changes if the same files are re-audited at a different threshold or with
    different sensitive columns. Two runs that share an id would have to be the same
    files under the same settings.

    Contrast with the Conformity Bundle id, which is derived from the evidence digest
    and is deliberately *stable* across re-evaluations of one run.
    """
    import hashlib

    seed = "|".join(
        [
            upload_id,
            model_sha,
            dataset_sha,
            config.target_column,
            config.positive_class,
            f"{config.decision_threshold:.6f}",
            ",".join(config.sensitive_columns),
            policy_id,
        ]
    ).encode("utf-8")
    return f"audit-{hashlib.sha256(seed).hexdigest()[:16]}"


def create_audit(
    metadata: ModelMetadataIn,
    config: AuditConfigIn,
    upload_manifest: dict[str, Any],
    *,
    security_acknowledged: bool,
    db_path: Any = None,
    register: bool = True,
) -> dict[str, Any]:
    """
    Run the full audit for a stored upload and write it to ``runtime/audits/<id>/``.

    Write order is deliberate and is what makes the evidence chain checkable:

    1. the seven measurement artefacts (metadata, predictions, metrics, risk)
    2. ``evidence_manifest.json`` -- sealing their checksums as the integrity baseline
    3. the four governance artefacts, via :mod:`app.gates.service`, which verify their
       inputs against that baseline as they evaluate
    4. ``manifest.json`` last, recording the checksums of all twelve others plus the
       resulting governance state

    Nothing is written until validation has fully passed, so a rejected submission
    leaves no partial run behind.
    """
    from app.gates import policy_engine
    from app.gates import service as gates_service
    from app.onboarding import upload_service as uploads

    upload_id = str(upload_manifest["upload_id"])
    model_path, dataset_path = uploads.upload_paths(upload_manifest)

    # -- validate ----------------------------------------------------------- #
    dataset_bytes = dataset_path.read_bytes()
    dataset, frame, issues = uploads.validate_dataset(
        dataset_bytes, config.target_column, config.positive_class, config.sensitive_columns
    )
    capabilities, compatibility, model, issues = validate_submission(
        model_path,
        frame,
        dataset,
        config,
        security_acknowledged=security_acknowledged,
        issues=issues,
    )
    if any(i.severity == "error" for i in issues):
        raise AuditRejected(issues)
    assert frame is not None and dataset is not None
    assert capabilities is not None and compatibility is not None

    policy = policy_engine.load_policy(config.policy_profile_id)
    model_sha = str(upload_manifest["model"]["sha256"])
    dataset_sha = str(upload_manifest["dataset"]["sha256"])
    audit_run_id = audit_run_identifier(
        upload_id, model_sha, dataset_sha, config, policy.policy_id
    )
    created_at = store.utc_now()
    directory = store.audit_dir(audit_run_id, create=True)

    # -- inference ---------------------------------------------------------- #
    positive_index, negative_classes = resolve_positive_class_index(
        capabilities, config.positive_class
    )
    actual_positive = (
        frame[config.target_column]
        .astype("string")
        .str.strip()
        .eq(config.positive_class.strip())
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    features = model_validator.model_input_frame(
        capabilities, frame, config.target_column
    )
    predicted_positive, probability, threshold_applied, proba_reason = run_inference(
        model, features, capabilities, config.decision_threshold, positive_index
    )

    # -- measure ------------------------------------------------------------ #
    performance = compute_performance(
        actual_positive,
        predicted_positive,
        probability,
        config.decision_threshold,
        threshold_applied,
        config.positive_class,
        negative_classes,
        proba_reason,
    )
    fairness = compute_fairness(
        frame, actual_positive, predicted_positive, dataset.sensitive_columns
    )
    explainability = compute_explainability(
        model,
        features,
        actual_positive,
        capabilities,
        dataset.sensitive_columns,
        frame,
        actual_labels=frame[config.target_column],
    )
    risk = build_risk_summary(
        performance, fairness, explainability, capabilities, metadata, config, dataset
    )
    coverage = audit_capability_map(capabilities, config)

    # -- measurement artefacts ---------------------------------------------- #
    model_metadata = {
        "audit_run_id": audit_run_id,
        "upload_id": upload_id,
        "created_at": created_at,
        **metadata.model_dump(),
        "sha256": model_sha,
        "size_bytes": upload_manifest["model"].get("size_bytes"),
        "stored_path": store.relative_path(model_path),
        "stored_filename": model_path.name,
        "original_filename_label": upload_manifest["model"].get(
            "original_filename_label"
        ),
        "security_acknowledged": bool(security_acknowledged),
        "security_warning": security.JOBLIB_SECURITY_WARNING,
        "production_hardening": security.PRODUCTION_HARDENING_NOTE,
        "model_capabilities": capabilities.model_dump(),
        "feature_compatibility": compatibility.model_dump(),
        "audit_coverage": coverage,
        "policy_profile_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "target_column": config.target_column,
        "positive_class": config.positive_class,
        "decision_threshold": config.decision_threshold,
        "sensitive_columns": dataset.sensitive_columns,
        "notice": DECISION_SUPPORT_NOTICE,
    }
    dataset_metadata = {
        "audit_run_id": audit_run_id,
        "upload_id": upload_id,
        "created_at": created_at,
        "sha256": dataset_sha,
        "size_bytes": upload_manifest["dataset"].get("size_bytes"),
        "stored_path": store.relative_path(dataset_path),
        "stored_filename": dataset_path.name,
        "original_filename_label": upload_manifest["dataset"].get(
            "original_filename_label"
        ),
        **dataset.model_dump(),
        "decision_threshold": config.decision_threshold,
        "decision_context": metadata.decision_context,
        "na_values_treated_as_null": uploads.NA_VALUES,
        "notice": DECISION_SUPPORT_NOTICE,
    }

    store.write_json(directory / "uploaded_model_metadata.json", model_metadata)
    store.write_json(directory / "uploaded_dataset_metadata.json", dataset_metadata)

    predictions = build_predictions_frame(
        frame,
        config,
        actual_positive,
        predicted_positive,
        probability,
        config.positive_class,
        negative_classes[0] if negative_classes else "other",
    )
    store.write_text(
        directory / "predictions.csv", predictions.to_csv(index=False, lineterminator="\n")
    )

    store.write_json(
        directory / "performance.json", {"audit_run_id": audit_run_id, **performance}
    )
    store.write_json(
        directory / "fairness.json", {"audit_run_id": audit_run_id, **fairness}
    )
    store.write_json(
        directory / "explainability.json",
        {"audit_run_id": audit_run_id, **explainability},
    )
    store.write_json(
        directory / "risk_summary.json", {"audit_run_id": audit_run_id, **risk}
    )

    # -- seal the integrity baseline ---------------------------------------- #
    measurement_artifacts = [
        "uploaded_model_metadata.json",
        "uploaded_dataset_metadata.json",
        "predictions.csv",
        "performance.json",
        "fairness.json",
        "explainability.json",
        "risk_summary.json",
    ]
    records = [
        store.checksum_record(directory / name, "audit_evidence")
        for name in measurement_artifacts
    ]
    records.append(store.checksum_record(model_path, "upload_source"))
    records.append(store.checksum_record(dataset_path, "upload_source"))
    store.write_json(
        directory / "evidence_manifest.json",
        {
            "audit_run_id": audit_run_id,
            "generated_at": created_at,
            "artifact_count": len(records),
            "artifacts": records,
            "integrity_method": "SHA-256 over file bytes, recomputed on demand and "
            "compared to the value recorded here.",
            "signature": None,
            "signature_note": "No digital signature is implemented in this prototype. "
            "This manifest detects change; it does not authenticate an author and "
            "provides no non-repudiation.",
            "notice": DECISION_SUPPORT_NOTICE,
        },
    )

    # -- governance artefacts ----------------------------------------------- #
    gate_result = gates_service.evaluate_run(
        audit_run_id,
        policy_id=policy.policy_id,
        allow_replace=False,
        db_path=db_path,
        audit_coverage=coverage,
        now=created_at,
    )
    evaluation = gate_result["evaluation"]
    bundle = gate_result["bundle"]
    governance = gate_result["governance"]

    # -- run manifest, written last ----------------------------------------- #
    artifact_records = [
        store.checksum_record(directory / name, "audit_run")
        for name in store.AUDIT_ARTIFACT_NAMES
        if name != "manifest.json" and (directory / name).is_file()
    ]
    manifest = {
        "audit_run_id": audit_run_id,
        "run_type": "uploaded_model",
        "created_at": created_at,
        "source_upload_ids": [upload_id],
        "model_checksum": model_sha,
        "dataset_checksum": dataset_sha,
        "model_metadata": metadata.model_dump(),
        "target_configuration": {
            "target_column": config.target_column,
            "positive_class": config.positive_class,
            "negative_classes": negative_classes,
            "sensitive_columns": dataset.sensitive_columns,
        },
        "decision_threshold": config.decision_threshold,
        "threshold_applied": threshold_applied,
        "selected_sensitive_columns": dataset.sensitive_columns,
        "policy_profile_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "policy_checksum": policy.checksum,
        "audit_coverage": coverage,
        "overall_governance_state": governance["governance_state"],
        "gate_summary": evaluation.gate_summary,
        "conformity_bundle_id": bundle.bundle_id,
        "artifact_count": len(artifact_records),
        "generated_artifacts": artifact_records,
        "human_review_required": True,
        "deployment_authorisation": "not_granted",
        "reference_case_note": (
            "This is an uploaded-model audit run. The built-in Adult Income reference "
            "case is separate and unchanged."
        ),
        "notice": DECISION_SUPPORT_NOTICE,
    }
    store.write_json(directory / "manifest.json", manifest)

    # -- registry ------------------------------------------------------------ #
    warnings = [i for i in issues if i.severity == "warning"]
    registry_run_id: str | None = None
    if register:
        from app.registry import service as registry_service

        try:
            registry_run_id = registry_service.register_uploaded_run(
                audit_run_id, db_path=db_path
            )
        except Exception as exc:  # registry failure must not lose written evidence
            warnings.append(
                ValidationIssue(
                    code="registry_registration_failed",
                    severity="warning",
                    field="registry",
                    message=f"The audit completed and its evidence was written, but "
                    f"registry registration failed: {type(exc).__name__}: {exc}",
                    hint="The run is readable via /api/onboarding/audits; it is simply "
                    "not indexed in the SQLite registry.",
                )
            )

    return {
        "audit_run_id": audit_run_id,
        "upload_id": upload_id,
        "created_at": created_at,
        "governance_state": governance["governance_state"],
        "fairness_status": fairness["status"],
        "explainability_status": explainability["status"],
        "gate_summary": evaluation.gate_summary,
        "conformity_bundle_id": bundle.bundle_id,
        "artifact_count": len(artifact_records) + 1,
        "registry_run_id": registry_run_id,
        "written_under": store.relative_path(directory),
        "warnings": warnings,
        "manifest": manifest,
        "governance": governance,
    }


# --------------------------------------------------------------------------- #
# Reads over completed runs
#
# The HTTP layer calls these; it never touches runtime/ itself. Each read is
# served from the run's own artefacts, so what an endpoint returns is what was
# written at audit time -- not a recomputation that could quietly disagree with
# the evidence a reviewer downloaded.
# --------------------------------------------------------------------------- #
def _summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project a run manifest into the compact list-endpoint shape."""
    audit_run_id = str(manifest["audit_run_id"])
    model = manifest.get("model_metadata") or {}
    target = manifest.get("target_configuration") or {}

    def _status(name: str) -> str:
        try:
            return str(store.read_json(audit_run_id, name).get("status") or "unavailable")
        except store.AuditArtifactMissing:
            return "unavailable"

    try:
        dataset = store.read_json(audit_run_id, "uploaded_dataset_metadata.json")
    except store.AuditArtifactMissing:
        dataset = {}

    return {
        "audit_run_id": audit_run_id,
        "run_type": str(manifest.get("run_type") or "uploaded_model"),
        "created_at": str(manifest.get("created_at") or ""),
        "model_name": str(model.get("model_name") or "unknown"),
        "model_version": str(model.get("model_version") or "unknown"),
        "model_owner": str(model.get("model_owner") or "unknown"),
        "dataset_row_count": int(dataset.get("row_count") or 0),
        "dataset_column_count": int(dataset.get("column_count") or 0),
        "target_column": str(target.get("target_column") or ""),
        "positive_class": str(target.get("positive_class") or ""),
        "decision_threshold": float(manifest.get("decision_threshold") or 0.0),
        "sensitive_columns": list(target.get("sensitive_columns") or []),
        "policy_profile_id": str(manifest.get("policy_profile_id") or ""),
        "policy_version": str(manifest.get("policy_version") or ""),
        "governance_state": str(
            manifest.get("overall_governance_state") or "insufficient_evidence"
        ),
        "fairness_status": _status("fairness.json"),
        "explainability_status": _status("explainability.json"),
        "gate_summary": dict(manifest.get("gate_summary") or {}),
        "conformity_bundle_id": manifest.get("conformity_bundle_id"),
        # The manifest counts the artefacts it baselines, which excludes itself.
        # The count reported here is files on disk, matching what the creation
        # response returned for the same run.
        "artifact_count": int(manifest.get("artifact_count") or 0) + 1,
    }


def list_audits() -> dict[str, Any]:
    """
    Every completed uploaded-model audit run on disk, newest first.

    A directory without a readable ``manifest.json`` is skipped rather than
    half-reported: an incomplete run is not an audit, and listing it would invite a
    reviewer to cite evidence that was never sealed. ``count`` therefore describes
    what is actually returned.
    """
    runs: list[dict[str, Any]] = []
    for audit_run_id in store.list_audit_ids():
        try:
            manifest = store.read_json(audit_run_id, "manifest.json")
        except (store.AuditArtifactMissing, store.AuditRunNotFound):
            continue
        runs.append(_summary_from_manifest(manifest))
    runs.sort(key=lambda run: (run["created_at"], run["audit_run_id"]), reverse=True)
    return {"count": len(runs), "runs": runs}


def get_audit(audit_run_id: str) -> dict[str, Any]:
    """Full detail for one run, assembled from its own artefacts."""
    manifest = store.read_json(audit_run_id, "manifest.json")
    model_metadata = store.read_json(audit_run_id, "uploaded_model_metadata.json")
    dataset_metadata = store.read_json(audit_run_id, "uploaded_dataset_metadata.json")
    try:
        governance = store.read_json(audit_run_id, "governance_summary.json")
    except store.AuditArtifactMissing:
        governance = {}

    source_upload_ids = list(manifest.get("source_upload_ids") or [])
    return {
        "audit_run_id": audit_run_id,
        "run_type": str(manifest.get("run_type") or "uploaded_model"),
        "created_at": str(manifest.get("created_at") or ""),
        "model_metadata": dict(manifest.get("model_metadata") or {}),
        "dataset_metadata": dataset_metadata,
        "target_configuration": dict(manifest.get("target_configuration") or {}),
        "model_checksum": str(manifest.get("model_checksum") or ""),
        "dataset_checksum": str(manifest.get("dataset_checksum") or ""),
        "upload_id": source_upload_ids[0] if source_upload_ids else "",
        "security_acknowledged": bool(model_metadata.get("security_acknowledged")),
        "security_warning": str(
            model_metadata.get("security_warning") or security.JOBLIB_SECURITY_WARNING
        ),
        "model_capabilities": model_metadata.get("model_capabilities") or {},
        "feature_compatibility": model_metadata.get("feature_compatibility") or {},
        "audit_coverage": dict(manifest.get("audit_coverage") or {}),
        "governance_state": str(
            manifest.get("overall_governance_state") or "insufficient_evidence"
        ),
        "policy_profile_id": str(manifest.get("policy_profile_id") or ""),
        "policy_version": str(manifest.get("policy_version") or ""),
        "conformity_bundle_id": manifest.get("conformity_bundle_id"),
        "artifacts": list(manifest.get("generated_artifacts") or []),
        "limitations": list(governance.get("limitations") or []),
    }


def _integrity_baselines(audit_run_id: str) -> dict[str, dict[str, Any]]:
    """
    Merge the two integrity baselines a run carries.

    ``evidence_manifest.json`` covers the measurement artefacts plus the two upload
    source files; ``manifest.json`` covers everything generated, including the
    governance artefacts written after the evidence manifest was sealed. Together
    they cover every file the run produced, which is what makes a per-file verdict
    possible instead of a partial one.

    Order matters and is not incidental. ``evidence_manifest.json`` is read **last**
    so its entries win any overlap, because it is the *sealed* baseline: it is written
    once and never rewritten, whereas ``manifest.json`` is refreshed when the gates are
    re-evaluated. If the refreshable index could override the seal, a re-evaluation
    would silently re-baseline an edited measurement artefact and tamper detection
    would become a formality.
    """
    baselines: dict[str, dict[str, Any]] = {}
    for artefact in ("manifest.json", "evidence_manifest.json"):
        try:
            document = store.read_json(audit_run_id, artefact)
        except store.AuditArtifactMissing:
            continue
        entries = list(document.get("artifacts") or []) + list(
            document.get("generated_artifacts") or []
        )
        for entry in entries:
            path = entry.get("path")
            if path and entry.get("sha256"):
                baselines[str(path)] = entry
    return baselines


def check_run_integrity(audit_run_id: str) -> dict[str, Any]:
    """
    Recompute the SHA-256 of every artefact this run recorded and classify each.

    Deliberately independent of the SQLite registry: the baselines live in the run's
    own directory, so integrity stays checkable even if registration failed or the
    database was deleted. ``manifest.json`` is reported as ``not_baselined`` rather
    than verified -- a document cannot contain its own checksum, and claiming to have
    verified it would be false.
    """
    if not store.audit_dir(audit_run_id).is_dir():
        raise store.AuditRunNotFound(audit_run_id, store.list_audit_ids())

    results: list[dict[str, Any]] = []
    verified = changed = missing = 0

    for relative, entry in sorted(_integrity_baselines(audit_run_id).items()):
        path = store.PROJECT_ROOT / relative
        if not path.is_file():
            missing += 1
            results.append(
                {
                    "artifact": Path(relative).name,
                    "path": relative,
                    "group": entry.get("group") or "unknown",
                    "status": "missing",
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": None,
                    "detail": "The recorded artefact is no longer on disk, so any "
                    "conclusion that cited it cannot be verified.",
                }
            )
            continue
        actual = store.sha256_file(path)
        if actual == entry.get("sha256"):
            verified += 1
            status = "verified"
            detail = "The bytes still hash to the value recorded at audit time."
        else:
            changed += 1
            status = "changed"
            detail = (
                "The file has been modified since this run recorded it, so this run's "
                "conclusions no longer describe the file on disk."
            )
        results.append(
            {
                "artifact": Path(relative).name,
                "path": relative,
                "group": entry.get("group") or "unknown",
                "status": status,
                "expected_sha256": entry.get("sha256"),
                "actual_sha256": actual,
                "detail": detail,
            }
        )

    manifest_path = store.artifact_path(audit_run_id, "manifest.json")
    if manifest_path.is_file():
        results.append(
            {
                "artifact": "manifest.json",
                "path": store.relative_path(manifest_path),
                "group": "audit_run",
                "status": "not_baselined",
                "expected_sha256": None,
                "actual_sha256": store.sha256_file(manifest_path),
                "detail": "The run manifest cannot record its own checksum. Its hash "
                "is reported for external comparison; this run does not verify it.",
            }
        )

    if changed and missing:
        overall = "modified_and_incomplete"
    elif changed:
        overall = "modified"
    elif missing:
        overall = "incomplete"
    else:
        overall = "verified"

    return {
        "audit_run_id": audit_run_id,
        "checked_at": store.utc_now(),
        "integrity_status": overall,
        "integrity_ok": overall == "verified",
        "artifacts_checked": verified + changed + missing,
        "verified_count": verified,
        "changed_count": changed,
        "missing_count": missing,
        "artifacts": results,
        "method": "SHA-256 recomputed from file bytes now and compared with the value "
        "recorded in this run's evidence_manifest.json and manifest.json.",
        "interpretation": [
            "'verified' means the bytes are unchanged since this run recorded them. It "
            "says nothing about whether the evidence was correct in the first place.",
            "'changed' means this run's conclusions no longer describe the file on "
            "disk. Re-evaluating the gates produces a new bundle over the new bytes.",
            "Checksums detect modification, not authorship. No digital signature is "
            "implemented, so this is change detection and not proof of provenance.",
        ],
        "notice": DECISION_SUPPORT_NOTICE,
    }


def build_timeline(audit_run_id: str, db_path: Any = None) -> dict[str, Any]:
    """
    Chronological history for one uploaded run, from three labelled sources.

    ``run`` events are *derived* from the timestamps the artefacts themselves carry,
    so the timeline cannot drift from the evidence -- there is no separately stored
    narrative to fall out of step. ``registry`` events come from the append-only
    event log, and ``waiver`` events from the waiver register. A missing registry or
    waiver store degrades to fewer events rather than an error, because the run's own
    history is complete without them.
    """
    manifest = store.read_json(audit_run_id, "manifest.json")
    created_at = str(manifest.get("created_at") or "")
    events: list[dict[str, Any]] = [
        {
            "event_time": created_at,
            "event_type": "uploads_accepted",
            "source": "run",
            "detail": "Uploads "
            + ", ".join(manifest.get("source_upload_ids") or ["unknown"])
            + " passed validation with the joblib security warning acknowledged. The "
            "model was deserialised only after that.",
        },
        {
            "event_time": created_at,
            "event_type": "audit_run_created",
            "source": "run",
            "detail": f"Run {audit_run_id} written to runtime/audits/{audit_run_id}/ "
            f"with {manifest.get('artifact_count')} artefacts.",
        },
    ]

    try:
        evidence = store.read_json(audit_run_id, "evidence_manifest.json")
        events.append(
            {
                "event_time": str(evidence.get("generated_at") or created_at),
                "event_type": "evidence_sealed",
                "source": "run",
                "detail": f"{evidence.get('artifact_count')} artefact checksums "
                "recorded as this run's integrity baseline.",
            }
        )
    except store.AuditArtifactMissing:
        pass

    try:
        evaluation = store.read_json(audit_run_id, "gate_evaluation.json")
        summary = evaluation.get("gate_summary") or {}
        events.append(
            {
                "event_time": str(evaluation.get("evaluated_at") or created_at),
                "event_type": "policy_gates_evaluated",
                "source": "run",
                "detail": f"Policy {evaluation.get('policy_profile_id')} "
                f"v{evaluation.get('policy_version')} evaluated: "
                + ", ".join(f"{gate}={result}" for gate, result in summary.items())
                + ". A gate result is a configured research-policy outcome, not a legal "
                "conclusion.",
            }
        )
    except store.AuditArtifactMissing:
        pass

    try:
        bundle = store.read_json(audit_run_id, "conformity_bundle.json")
        events.append(
            {
                "event_time": str(bundle.get("created_at") or created_at),
                "event_type": "conformity_bundle_assembled",
                "source": "run",
                "detail": f"Bundle {bundle.get('bundle_id')} assembled over "
                f"{len(bundle.get('evidence') or [])} evidence artefacts.",
            }
        )
    except store.AuditArtifactMissing:
        pass

    try:
        from app.registry import db as registry_db

        connection = registry_db.connect(db_path, create=False)
        try:
            rows = connection.execute(
                "SELECT event_time, event_type, detail FROM registry_events "
                "WHERE run_id = ? ORDER BY event_id",
                (audit_run_id,),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            events.append(
                {
                    "event_time": row["event_time"],
                    "event_type": row["event_type"],
                    "source": "registry",
                    "detail": row["detail"],
                }
            )
    except Exception:  # an unregistered run still has its own complete history
        pass

    try:
        from app.gates import service as gates_service

        for waiver in gates_service.list_waivers(audit_run_id, db_path):
            events.append(
                {
                    "event_time": waiver.created_at,
                    "event_type": f"waiver_{waiver.status}",
                    "source": "waiver",
                    "detail": f"{waiver.waiver_id} against {waiver.control_id}, owner "
                    f"{waiver.owner}, expires {waiver.expires_at}. Rationale: "
                    f"{waiver.rationale}",
                }
            )
    except Exception:  # a waiver-store problem must not hide the run's own history
        pass

    events.sort(key=lambda event: (event["event_time"], event["event_type"]))
    return {
        "audit_run_id": audit_run_id,
        "count": len(events),
        "events": events,
        "note": "Run events are derived from the timestamps the artefacts themselves "
        "carry, so the timeline cannot drift from the evidence. Registry and waiver "
        "events come from append-only logs. Neither is a signed provenance record.",
        "notice": DECISION_SUPPORT_NOTICE,
    }
