"""
preprocessing.py
================
Leakage-safe feature engineering + the single canonical train/test split.

Why this file exists
--------------------
Every transformation that *learns a parameter from data* (median for imputation,
the set of one-hot categories, the mean/std used for scaling) is wrapped in a
``ColumnTransformer`` and returned **unfitted**. `train.py` puts it as step 1 of
a ``Pipeline`` whose final step is the estimator, then calls ``fit`` on the
TRAINING FOLD ONLY.

That gives three guarantees:

1. **No data leakage.** Test-set statistics never influence the fitted
   transformers. Fitting an imputer or scaler on the full dataset before
   splitting (a very common mistake) leaks test information into training and
   inflates the reported score.
2. **No train/serve skew.** The saved ``.joblib`` contains preprocessing *and*
   model, so inference applies the exact same transformations. You cannot forget
   to scale at prediction time.
3. **Unknown categories are safe.** ``OneHotEncoder(handle_unknown="ignore")``
   means a category seen only in the test set (or in future production data)
   becomes an all-zero block instead of raising.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Split configuration -- kept as module constants so `train.py` and
# `evaluate.py` reproduce *bit-identical* folds.
TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Column typing
# --------------------------------------------------------------------------- #
def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric_cols, categorical_cols)`` inferred from dtypes."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _make_ohe() -> OneHotEncoder:
    """
    OneHotEncoder with dense output, compatible across scikit-learn versions.

    ``handle_unknown="ignore"`` is the leakage/robustness requirement: categories
    absent from the training fold are encoded as all zeros rather than crashing.
    """
    try:  # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build the (unfitted) preprocessing ``ColumnTransformer``.

    Parameters
    ----------
    scale_numeric :
        ``True``  -> numeric branch = median impute + StandardScaler.
                     Required for Logistic Regression: it is a distance/gradient
                     based model with regularisation, so unscaled features like
                     ``fnlwgt`` (~1e5) would dominate ``age`` (~1e1) and the
                     L2 penalty would be applied unevenly. It also lets lbfgs
                     converge in a sane number of iterations.
        ``False`` -> numeric branch = median impute only.
                     Tree ensembles (Random Forest, XGBoost) split on ordered
                     thresholds, so any monotonic rescaling is a no-op for them;
                     skipping it saves work and keeps features interpretable.

    Notes
    -----
    * Numeric missing values -> **median** (robust to the heavy right skew in
      ``capital-gain`` / ``fnlwgt``; the mean would be dragged by outliers).
    * Categorical missing values -> **most frequent** category.
    * Both imputers learn their fill value from the training fold only.
    """
    numeric_cols, categorical_cols = split_columns(X)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",       # be explicit: nothing sneaks through unprocessed
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# The canonical split
# --------------------------------------------------------------------------- #
def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    80/20 train/test split, stratified on the target.

    * ``stratify=y`` preserves the class ratio (~24% earn >50K) in BOTH folds.
      Adult is imbalanced, so an unstratified random split can shift the positive
      rate between folds and make recall / ROC-AUC noisy and non-comparable
      across models.
    * ``random_state=42`` makes the split deterministic, so every model sees the
      exact same rows and `evaluate.py` can rebuild the identical test set later.
    * ``X`` stays a DataFrame (not a NumPy array) so the ColumnTransformer can
      select columns by name and so ``sex`` / ``race`` remain addressable for the
      fairness columns in the prediction outputs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"[preprocessing] Split -> train={X_train.shape[0]:,} rows, "
        f"test={X_test.shape[0]:,} rows "
        f"(positive rate: train={y_train.mean():.4f}, test={y_test.mean():.4f})"
    )
    return X_train, X_test, y_train, y_test
