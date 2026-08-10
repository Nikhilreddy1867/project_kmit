"""
evaluate.py
===========
Metric computation, prediction export and confusion-matrix plots.

Used two ways:

* imported by ``train.py`` right after each model is fitted;
* run standalone (``python src/evaluate.py``) to re-score the saved ``.joblib``
  pipelines without retraining. Because the split is deterministic
  (``stratify=y``, ``random_state=42``), the rebuilt test set is identical to the
  one held out during training, so the numbers match exactly.

All metrics are computed on the HELD-OUT TEST SET only -- never on training data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SENSITIVE_COLS,
    ensure_dirs,
    load_dataset,
)
from preprocessing import make_split

# Human-readable label for the positive class (encoded as 1).
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
CLASS_NAMES = [f"{NEGATIVE_LABEL} (0)", f"{POSITIVE_LABEL} (1)"]

METRICS_CSV = RESULTS_DIR / "model_metrics.csv"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Classification metrics for the positive class ``>50K`` (label 1).

    Why each one, on an imbalanced target (~24% positives):
    * accuracy  -- headline number, but a "predict everyone <=50K" model already
                   scores ~0.76, so it must never be read alone.
    * precision -- of those flagged as high earners, how many really are.
    * recall    -- of the true high earners, how many we caught.
    * f1        -- harmonic mean of the two; the single best summary here.
    * roc_auc   -- threshold-free ranking quality, computed from PROBABILITIES
                   (not hard labels), so it reflects calibration-independent
                   discriminative power.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "n_test": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# Confusion matrix figure
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Save an annotated confusion-matrix PNG (raw counts + row-normalised %)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)  # per-true-class recall view

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted income")
    ax.set_ylabel("Actual income")
    ax.set_title(f"Confusion matrix - {model_name}\n(held-out test set)")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{cm_pct[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if cm_pct[i, j] > 0.5 else "black",
                fontsize=11,
            )

    fig.colorbar(im, ax=ax, label="Share of actual class")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_comparison(metrics_df: pd.DataFrame, out_dir: Path = RESULTS_DIR) -> Path:
    """Grouped bar chart comparing all models across the headline metrics."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df = metrics_df.set_index("model")[cols]

    x = np.arange(len(cols))
    width = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, row) in enumerate(df.iterrows()):
        offset = (i - (len(df) - 1) / 2) * width
        bars = ax.bar(x + offset, row.values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x, labels=cols)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set (positive class = >50K)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Prediction export
# --------------------------------------------------------------------------- #
def save_predictions(
    model_name: str,
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path = PREDICTIONS_DIR,
) -> Path:
    """
    Write one row per test record.

    Columns (exactly as required by the governance spec):
      actual_income, predicted_income, predicted_probability, sex, race

    ``sex`` and ``race`` are copied from the RAW (pre-transform) test features,
    so they are the original human-readable values -- not one-hot columns. This
    is what makes group-wise fairness analysis (demographic parity, equal
    opportunity, ...) possible in Phase 2 without re-running the model.
    """
    out = pd.DataFrame(
        {
            "actual_income": np.asarray(y_true).astype(int),
            "predicted_income": np.asarray(y_pred).astype(int),
            "predicted_probability": np.asarray(y_prob, dtype=float),
        },
        index=X_test.index,
    )
    for col in SENSITIVE_COLS:
        out[col] = X_test[col].values

    # Extra convenience columns: the string labels behind the 0/1 encoding.
    out["actual_income_label"] = np.where(out["actual_income"] == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    out["predicted_income_label"] = np.where(
        out["predicted_income"] == 1, POSITIVE_LABEL, NEGATIVE_LABEL
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_test_predictions.csv"
    out.to_csv(path, index=False)
    return path


def save_classification_report(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path = RESULTS_DIR
) -> Path:
    """Persist sklearn's per-class text report for the record."""
    text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"classification_report_{model_name}.txt"
    path.write_text(f"Model: {model_name}\n\n{text}\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Full evaluation of one fitted pipeline
# --------------------------------------------------------------------------- #
def evaluate_model(
    model_name: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Score a fitted Pipeline on the test set and write all per-model artefacts.

    ``pipeline.predict`` / ``predict_proba`` run the *fitted* preprocessing from
    the training fold, so the test data is transformed but never fitted on.
    """
    y_true = np.asarray(y_test)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # P(income > 50K)

    metrics = compute_metrics(model_name, y_true, y_pred, y_prob)
    pred_path = save_predictions(model_name, X_test, y_true, y_pred, y_prob)
    cm_path = plot_confusion_matrix(model_name, y_true, y_pred)
    rep_path = save_classification_report(model_name, y_true, y_pred)

    print(f"[evaluate] {model_name}: " + "  ".join(
        f"{k}={metrics[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
    ))
    print(f"           predictions -> {pred_path}")
    print(f"           confusion   -> {cm_path}")
    print(f"           report      -> {rep_path}")
    return metrics


def save_metrics_table(rows: list[dict], path: Path = METRICS_CSV) -> pd.DataFrame:
    """Write the model-comparison table, best F1 first."""
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[evaluate] Metrics table -> {path}")
    return df


# --------------------------------------------------------------------------- #
# Standalone re-evaluation of saved models
# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_dirs()
    X, y = load_dataset()
    # Identical arguments as in training => identical test fold.
    _, X_test, _, y_test = make_split(X, y)

    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        raise SystemExit(
            f"No .joblib models in {MODELS_DIR}. Run `python src/train.py` first."
        )

    rows = []
    for f in model_files:
        name = f.stem.replace("_pipeline", "")
        print(f"\n[evaluate] Loading {f.name}")
        rows.append(evaluate_model(name, joblib.load(f), X_test, y_test))

    df = save_metrics_table(rows)
    plot_metric_comparison(df)
    print("\n=== Model comparison (test set) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
