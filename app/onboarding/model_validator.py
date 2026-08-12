"""
app/onboarding/model_validator.py
=================================
Deserialises an uploaded ``.joblib`` and establishes what it can actually do.

This is the **only** module that calls ``joblib.load`` on user-supplied bytes, and
it refuses to do so until :func:`~app.onboarding.security.require_acknowledgement`
has passed. The order is load-last by design: the dataset has already been parsed
and checked by then, so a submission that could never be audited is rejected
without executing the pickle.

Honest capability reporting
---------------------------
Capabilities are established by **inspection of the loaded object**, never assumed
from the filename or the user's metadata:

* ``predict`` missing            -> the model cannot be audited at all.
* ``predict_proba`` missing      -> ROC-AUC is reported ``null`` with a stated
  reason, the decision threshold is reported as *not applied*, and the model's own
  decision rule is used. No probability is ever synthesised from ``decision_function``,
  because a margin is not a calibrated probability and treating it as one would
  make ROC-AUC look available when it is not.
* more than two classes          -> rejected; this prototype audits binary
  classification only.
* no ``feature_names_in_``       -> compatibility falls back to the expected feature
  *count*, and the method used is reported so the reader knows which check ran.

Nothing here substitutes, wraps, refits or trains anything. A model that fails
validation is reported as failing; it is never quietly replaced by one that works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.onboarding import security
from app.onboarding.schemas import FeatureCompatibility, ModelCapabilities

#: Names of final-estimator classes for which exact TreeSHAP is attempted.
_XGBOOST_ESTIMATORS = {"XGBClassifier", "XGBRFClassifier"}


class ModelValidationError(Exception):
    """The uploaded file could not be loaded as an auditable estimator."""

    def __init__(self, code: str, message: str, hint: str | None = None):
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)


def unloaded_capabilities(detail: str) -> ModelCapabilities:
    """
    Capability record for a model that was deliberately **not** loaded.

    Used when the acknowledgement is absent or earlier validation failed, so the
    response can distinguish "we did not look" from "we looked and it lacks this".
    """
    return ModelCapabilities(loaded=False, estimator_type=None, pipeline_steps=[detail][:0])


def load_model(path: Path, *, security_acknowledged: bool) -> Any:
    """
    Deserialise the uploaded model, gated on the security acknowledgement.

    The gate is the first statement in the function so there is no ordering in
    which bytes are unpickled without it.
    """
    security.require_acknowledgement(security_acknowledged)

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - joblib is a hard dependency
        raise ModelValidationError(
            "joblib_unavailable", "joblib is not installed in this environment."
        ) from exc

    try:
        return joblib.load(path)
    except Exception as exc:
        # Any exception here is the unpickling of an untrusted file failing. Report
        # the type and message, but do not re-raise the original traceback to the
        # client: it can quote arbitrary attacker-controlled content.
        raise ModelValidationError(
            "model_load_failed",
            f"The uploaded .joblib could not be deserialised: "
            f"{type(exc).__name__}: {exc}",
            hint="It must be a joblib-serialised, fitted scikit-learn estimator or "
            "Pipeline. Files saved by a different library version may not load.",
        ) from exc


# --------------------------------------------------------------------------- #
# Capability inspection
# --------------------------------------------------------------------------- #
def _final_estimator(model: Any) -> Any:
    """The estimator that actually predicts, unwrapping a Pipeline if present."""
    steps = getattr(model, "steps", None)
    if steps:
        return steps[-1][1]
    return model


def _expected_features(model: Any) -> tuple[list[str] | None, int | None]:
    """
    The input feature names and count the fitted estimator expects.

    ``feature_names_in_`` is checked on the outermost object first: for a Pipeline
    that is the names seen by the *first* step, which is what the caller must
    supply. Falling through to the final estimator would report the post-transform
    feature space instead, which the user's CSV is not expected to match.
    """
    names = getattr(model, "feature_names_in_", None)
    count = getattr(model, "n_features_in_", None)
    if names is None:
        steps = getattr(model, "steps", None)
        if steps:
            first = steps[0][1]
            names = getattr(first, "feature_names_in_", None)
            if count is None:
                count = getattr(first, "n_features_in_", None)
    resolved = [str(n) for n in names] if names is not None else None
    return resolved, (int(count) if count is not None else None)


def inspect_model(model: Any) -> ModelCapabilities:
    """Build the capability record for a loaded estimator."""
    final = _final_estimator(model)
    steps = getattr(model, "steps", None)
    names, count = _expected_features(model)

    classes_raw = getattr(final, "classes_", None)
    classes = [str(c) for c in classes_raw] if classes_raw is not None else []

    has_predict = callable(getattr(model, "predict", None))
    has_proba = callable(getattr(model, "predict_proba", None))
    final_name = type(final).__name__

    capabilities = ModelCapabilities(
        loaded=True,
        estimator_type=type(model).__name__,
        estimator_module=type(model).__module__,
        is_pipeline=bool(steps),
        pipeline_steps=[str(name) for name, _ in steps] if steps else [],
        final_estimator=final_name,
        has_predict=has_predict,
        has_predict_proba=has_proba,
        is_binary_classifier=len(classes) == 2,
        classes=classes,
        expected_features=names,
        n_features_expected=count,
        # Permutation importance needs only a fitted predictor and a scorer, so it
        # is available for any estimator that can predict.
        supports_permutation_importance=has_predict,
        supports_treeshap=final_name in _XGBOOST_ESTIMATORS,
    )
    return capabilities


def validate_capabilities(capabilities: ModelCapabilities) -> list[tuple[str, str, str]]:
    """
    Check the capability record against what this prototype can audit.

    Returns ``(code, message, hint)`` triples rather than raising, so the caller can
    report every problem at once instead of only the first.
    """
    problems: list[tuple[str, str, str]] = []

    if not capabilities.has_predict:
        problems.append(
            (
                "model_missing_predict",
                f"The uploaded object ({capabilities.estimator_type}) has no "
                "predict(X) method, so it cannot be audited.",
                "Upload a fitted scikit-learn estimator or Pipeline.",
            )
        )
    if not capabilities.classes:
        problems.append(
            (
                "model_not_a_classifier",
                f"The uploaded object ({capabilities.estimator_type}) exposes no "
                "classes_ attribute, so it does not appear to be a fitted classifier.",
                "This prototype audits fitted binary classifiers only. A regressor "
                "or an unfitted estimator cannot be audited.",
            )
        )
    elif len(capabilities.classes) != 2:
        problems.append(
            (
                "model_not_binary",
                f"The model predicts {len(capabilities.classes)} classes "
                f"({', '.join(capabilities.classes[:6])}). This prototype audits "
                "binary classification only.",
                "No substitute model is trained and no classes are collapsed.",
            )
        )
    return problems


# --------------------------------------------------------------------------- #
# Feature compatibility
# --------------------------------------------------------------------------- #
def check_feature_compatibility(
    capabilities: ModelCapabilities,
    frame: pd.DataFrame,
    target_column: str,
    sensitive_columns: list[str],
) -> FeatureCompatibility:
    """
    Match the dataset's columns against the model's expected input features.

    Two checks, in preference order:

    * ``feature_names_in_`` -- exact set comparison, giving precise *missing* and
      *unexpected* lists.
    * ``n_features_in_``    -- count comparison only, when names are unavailable.

    Missing features are fatal; unexpected ones are not. Unexpected columns are
    dropped for inference and **retained for fairness reporting**, which is what
    lets a user group by a sensitive attribute the model never sees -- the case that
    matters most, since a model can be unfair through proxies without ever
    receiving the attribute itself.
    """
    candidates = [c for c in frame.columns if c != target_column]
    retained = [c for c in sensitive_columns if c in frame.columns]

    expected = capabilities.expected_features
    if expected:
        available = set(candidates)
        missing = [f for f in expected if f not in available]
        unexpected = [c for c in candidates if c not in set(expected)]
        return FeatureCompatibility(
            checked=True,
            compatible=not missing,
            missing_features=missing,
            unexpected_features=unexpected,
            matched_feature_count=len(expected) - len(missing),
            sensitive_columns_retained=retained,
            method="feature_names_in_ (exact name match)",
        )

    if capabilities.n_features_expected is not None:
        supplied = len(candidates)
        expected_count = capabilities.n_features_expected
        compatible = supplied >= expected_count
        missing = (
            []
            if compatible
            else [f"<{expected_count - supplied} unnamed feature(s)>"]
        )
        return FeatureCompatibility(
            checked=True,
            compatible=compatible,
            missing_features=missing,
            unexpected_features=[],
            matched_feature_count=min(supplied, expected_count),
            sensitive_columns_retained=retained,
            method=(
                f"n_features_in_ (count only: model expects {expected_count}, dataset "
                f"provides {supplied} non-target columns). The estimator records no "
                "feature names, so names could not be matched."
            ),
        )

    return FeatureCompatibility(
        checked=False,
        compatible=True,
        missing_features=[],
        unexpected_features=[],
        matched_feature_count=None,
        sensitive_columns_retained=retained,
        method=(
            "not checked: the estimator exposes neither feature_names_in_ nor "
            "n_features_in_. Compatibility is established only by the trial "
            "inference below, so a mismatch would surface as an inference error."
        ),
    )


def model_input_frame(
    capabilities: ModelCapabilities, frame: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """
    The exact columns handed to ``predict``, in the order the model expects.

    Column order matters: a fitted ColumnTransformer selects by position after
    validating names, and sklearn raises if the order differs from training. When
    the model records no names, the non-target columns are passed through in file
    order -- the only defensible choice, and the reason the compatibility method is
    reported to the user.
    """
    expected = capabilities.expected_features
    if expected:
        return frame.loc[:, list(expected)]
    return frame.drop(columns=[target_column], errors="ignore")
