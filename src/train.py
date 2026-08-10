"""
train.py
========
Phase-1 training entry point for the AI Governance Platform.

Run:  python src/train.py

Flow
----
raw snapshot -> clean -> **split first** -> per-model Pipeline(preprocessor, model)
-> fit on TRAIN only -> evaluate on TEST only -> persist artefacts.

Leakage-prevention checklist enforced here
------------------------------------------
* ``make_split`` is called on the *cleaned but untransformed* data, so the split
  happens BEFORE any fitted transformation exists.
* Each model gets its OWN preprocessor instance inside its OWN Pipeline, so
  imputer medians / one-hot vocabularies / scaler statistics are learned from
  that model's training fold and nothing else.
* ``pipeline.fit(X_train, y_train)`` is the only ``fit`` call in the file. The
  test set is touched exclusively through ``predict`` / ``predict_proba``.
* The whole fitted Pipeline (preprocessing + estimator) is what gets saved to
  ``.joblib`` -- no risk of a future caller applying different preprocessing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Allow `python src/train.py` from the repo root as well as from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_dataset  # noqa: E402
from evaluate import (  # noqa: E402
    evaluate_model,
    plot_metric_comparison,
    save_metrics_table,
)
from preprocessing import (  # noqa: E402
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    make_split,
    split_columns,
)

# XGBoost is a hard requirement of Phase 1 -- fail loudly rather than skipping it.
try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is required but not installed.\n"
        "Activate the venv and run:  pip install -r requirements.txt\n"
        f"(original error: {exc})"
    ) from exc


def build_models(X_train) -> dict[str, Pipeline]:
    """
    Return ``{model_name: unfitted Pipeline}``.

    Design notes
    ------------
    * Logistic Regression gets ``scale_numeric=True`` (see preprocessing.py);
      the tree models get ``scale_numeric=False`` because scaling cannot change
      a threshold split. Each therefore needs its own preprocessor instance --
      which is also the correct thing to do for leakage reasons.
    * ``class_weight="balanced"`` is deliberately NOT used, and no resampling is
      applied: the three models are compared under identical, untouched class
      priors so the metric table is apples-to-apples. Class re-weighting /
      threshold tuning is a Phase-2 fairness lever, tracked as future work.
    * Hyperparameters are sensible, fast, fixed defaults -- no tuning, no CV
      search -- so this baseline is reproducible on a laptop CPU. Any tuning
      must use CV *inside the training fold* to stay leakage-free.
    """
    _, categorical_cols = split_columns(X_train)
    print(f"[train] {len(categorical_cols)} categorical / "
          f"{X_train.shape[1] - len(categorical_cols)} numeric input features")

    models: dict[str, Pipeline] = {
        # --- 1. Logistic Regression: linear, interpretable governance baseline
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,      # headroom for lbfgs on ~100 one-hot columns
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        # --- 2. Random Forest: bagged trees, strong non-linear baseline
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,   # mild regularisation vs. fully grown trees
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        # --- 3. XGBoost: gradient boosting, usually the accuracy leader on Adult
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",   # fast histogram algorithm on CPU
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase-1 income-classification models.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-fetch the dataset from UCI even if data/raw/adult_raw.csv exists.",
    )
    args = parser.parse_args()

    ensure_dirs()

    # ---------------------------------------------------------------- 1. data
    print("=" * 78)
    print("STEP 1/5  Load + clean raw data")
    print("=" * 78)
    X, y = load_dataset(force_download=args.force_download)
    print(f"[train] Cleaned dataset: X={X.shape}, y={y.shape}, "
          f"positive rate={y.mean():.4f}")
    n_missing = int(X.isna().sum().sum())
    print(f"[train] Missing feature cells after '?'-> null conversion: {n_missing:,}")

    # --------------------------------------------------------------- 2. split
    print("\n" + "=" * 78)
    print(f"STEP 2/5  Stratified split (test_size={TEST_SIZE}, "
          f"random_state={RANDOM_STATE}) -- BEFORE any fitting")
    print("=" * 78)
    X_train, X_test, y_train, y_test = make_split(X, y)

    # ------------------------------------------------------- 3. build + train
    print("\n" + "=" * 78)
    print("STEP 3/5  Train models (fit on training fold only)")
    print("=" * 78)
    models = build_models(X_train)

    fitted: dict[str, Pipeline] = {}
    fit_seconds: dict[str, float] = {}
    for name, pipe in models.items():
        print(f"\n[train] Fitting {name} ...")
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)          # <-- the ONLY fit; test never seen
        elapsed = time.perf_counter() - t0
        fit_seconds[name] = elapsed
        fitted[name] = pipe

        path = MODELS_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipe, path)             # full pipeline: preprocessing + model
        size_mb = path.stat().st_size / 1e6
        print(f"[train] Fitted in {elapsed:,.1f}s -> saved {path} ({size_mb:.1f} MB)")

    # ------------------------------------------------------------ 4. evaluate
    print("\n" + "=" * 78)
    print("STEP 4/5  Evaluate on the held-out test set")
    print("=" * 78)
    rows = []
    for name, pipe in fitted.items():
        m = evaluate_model(name, pipe, X_test, y_test)
        m["fit_seconds"] = round(fit_seconds[name], 2)
        rows.append(m)

    # ------------------------------------------------------------- 5. outputs
    print("\n" + "=" * 78)
    print("STEP 5/5  Write comparison artefacts")
    print("=" * 78)
    metrics_df = save_metrics_table(rows)
    cmp_path = plot_metric_comparison(metrics_df)
    print(f"[train] Comparison chart -> {cmp_path}")

    print("\n=== FINAL MODEL COMPARISON (held-out test set, positive class = >50K) ===")
    show = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fit_seconds"]
    print(metrics_df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest F1: {metrics_df.iloc[0]['model']}")
    print(f"\nAll artefacts under: {RESULTS_DIR.parent}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
