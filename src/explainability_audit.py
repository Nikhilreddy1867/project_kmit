"""
explainability_audit.py
=======================
Phase 3 -- explainability audit of the Phase-1 income-classification baseline.

READ-ONLY with respect to Phases 1 and 2. This module loads the saved pipelines
to *interrogate* them, never to refit or re-save them, and writes only into
``results/explainability/``.

Primary model: **XGBoost** (best held-out performance in Phase 1). Logistic
Regression is included as a concise comparison.

Run:  python src/explainability_audit.py

Method choices, and why
-----------------------
* **Permutation importance is computed on the RAW DataFrame through the whole
  pipeline.** Permuting `occupation` shuffles one human-readable column and the
  fitted preprocessor re-encodes it, so the importance lands on the original
  feature instead of being scattered across 41 one-hot fragments. Reading
  importance off the transformed matrix would have produced exactly the
  fragmented output the spec asks us to avoid.
* **Scored on the held-out test set**, so importance reflects generalisation, not
  how hard the model leaned on a feature while memorising the training fold.
* **ROC-AUC is the primary scorer** (threshold-free), with accuracy and F1
  alongside: Phase 2 established that every 0.5-threshold quantity is a policy
  artefact, and importance rankings can genuinely reorder between scorers.
* **Local attributions use XGBoost's built-in exact TreeSHAP**
  (``pred_contribs=True``), which needs no extra dependency. SHAP values are
  additive in log-odds, so contributions from the one-hot columns of a single
  source feature **sum exactly** back to that feature -- aggregation loses
  nothing. The additivity is asserted numerically, not assumed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import SENSITIVE_COLS, load_dataset  # noqa: E402
from preprocessing import RANDOM_STATE, TEST_SIZE, make_split  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FAIRNESS_DIR = PROJECT_ROOT / "results" / "fairness"
OUT_DIR = PROJECT_ROOT / "results" / "explainability"

PRIMARY_MODEL = "xgboost"
COMPARISON_MODEL = "logistic_regression"

TOP_N = 15  # the dataset has 14 original features, so this is the full ranked set
N_REPEATS = 10
SCORERS = ["roc_auc", "accuracy", "f1"]
PRIMARY_SCORER = "roc_auc"

# Features whose presence among the top factors is governance-relevant: the
# protected attributes themselves, plus the features most likely to act as their
# proxies. Flagged automatically so the finding cannot depend on us eyeballing it.
PROTECTED = ["sex", "race"]
LIKELY_PROXIES = [
    "marital-status",
    "relationship",
    "occupation",
    "education",
    "education-num",
    "hours-per-week",
    "native-country",
]

# --- chart style: validated categorical palette, slots 1-2 ------------------ #
C_XGB = "#2a78d6"   # blue
C_LR = "#eb6834"    # orange
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
    }
)


# --------------------------------------------------------------------------- #
# Transformed-column -> original-feature map
# --------------------------------------------------------------------------- #
def build_feature_map(preprocessor) -> list[str]:
    """
    Return, for each of the 105 transformed columns, the ORIGINAL feature it came
    from.

    Built from the fitted transformers' own metadata (column lists and
    ``OneHotEncoder.categories_``) rather than by parsing ``"workclass_Private"``
    strings -- category values in this dataset contain hyphens and dots, so string
    splitting on the separator would mis-assign columns.
    """
    mapping: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            mapping.extend(columns)  # one transformed column per numeric column
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            for col, cats in zip(columns, ohe.categories_):
                mapping.extend([col] * len(cats))  # one column per category
    expected = len(preprocessor.get_feature_names_out())
    if len(mapping) != expected:
        raise RuntimeError(f"Feature map has {len(mapping)} entries, expected {expected}.")
    return mapping


# --------------------------------------------------------------------------- #
# 1. Global: permutation importance
# --------------------------------------------------------------------------- #
def global_permutation_importance(pipeline, X_test, y_test, label: str) -> pd.DataFrame:
    """
    Permutation importance over the ORIGINAL features, on the held-out test set.

    Importance = drop in score when one column is randomly shuffled. A large drop
    means the fitted model *relies* on that column; it says nothing about whether
    the column causes income.
    """
    print(f"[explain] Permutation importance for {label} "
          f"({X_test.shape[1]} features x {N_REPEATS} repeats, scorers={SCORERS}) ...")
    res = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=SCORERS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,  # reproducible shuffles
        n_jobs=1,                   # avoid nesting on top of XGBoost's own threads
    )
    out = pd.DataFrame({"feature": X_test.columns})
    for scorer in SCORERS:
        out[f"{label}_perm_{scorer}_mean"] = res[scorer].importances_mean
        out[f"{label}_perm_{scorer}_std"] = res[scorer].importances_std
    return out


def xgb_native_gain(pipeline, feature_map: list[str]) -> pd.DataFrame:
    """
    XGBoost's own gain-based importance, aggregated from one-hot columns back to
    original features and normalised to shares.

    Included as a cross-check with a completely different definition of
    "important": gain is measured on the TRAINING data during tree construction,
    whereas permutation importance is measured on held-out predictions. Where the
    two disagree, neither is wrong -- they answer different questions.
    """
    booster = pipeline.named_steps["model"].get_booster()
    scores = booster.get_score(importance_type="gain")
    agg: dict[str, float] = {}
    for key, gain in scores.items():
        idx = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else None
        if idx is None:
            continue
        agg[feature_map[idx]] = agg.get(feature_map[idx], 0.0) + float(gain)
    total = sum(agg.values()) or 1.0
    return pd.DataFrame(
        {"feature": list(agg), "xgb_native_gain_share": [v / total for v in agg.values()]}
    )


def lr_coefficients(pipeline, feature_map: list[str]) -> pd.DataFrame:
    """
    Logistic Regression coefficients in log-odds, with their original feature.

    Two interpretation caveats, both stated in the report:
    * numeric coefficients are per **1 standard deviation** (the numeric branch is
      standardised), not per natural unit;
    * no one-hot baseline category was dropped, so a single dummy coefficient is
      only meaningful *relative to the other categories of the same feature*, not
      as an absolute effect. That is why the per-feature summary below uses the
      **spread** (max - min) across a feature's categories.
    """
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.coef_[0]
    return pd.DataFrame(
        {
            "transformed_column": names,
            "original_feature": feature_map,
            "coefficient_log_odds": coefs,
            "abs_coefficient": np.abs(coefs),
            "direction": np.where(coefs >= 0, "increases P(>50K)", "decreases P(>50K)"),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def lr_feature_summary(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse LR coefficients to one row per original feature."""
    rows = []
    for feat, g in coef_df.groupby("original_feature"):
        c = g["coefficient_log_odds"]
        top = g.loc[g["abs_coefficient"].idxmax()]
        rows.append(
            {
                "feature": feat,
                "lr_coef_spread": float(c.max() - c.min()) if len(c) > 1 else float(abs(c.iloc[0])),
                "lr_max_abs_coef": float(g["abs_coefficient"].max()),
                "lr_strongest_column": top["transformed_column"],
                "lr_strongest_coef": float(top["coefficient_log_odds"]),
            }
        )
    return pd.DataFrame(rows)


def plot_global_importance(imp: pd.DataFrame, out_path: Path) -> Path:
    """
    Horizontal grouped bars: permutation importance per original feature, both
    models, ranked by the primary model.

    Both series are the same unit (drop in test ROC-AUC), so they share ONE axis.
    Error bars are +/- 1 std over the permutation repeats -- essential here,
    because several low-ranked features have importance smaller than their own
    run-to-run variation and are therefore not distinguishable from zero.
    """
    d = imp.head(TOP_N).iloc[::-1]  # best at top
    y = np.arange(len(d))
    height = 0.38

    fig, ax = plt.subplots(figsize=(10.4, 7.4))
    for i, (label, colour) in enumerate(
        [(PRIMARY_MODEL, C_XGB), (COMPARISON_MODEL, C_LR)]
    ):
        vals = d[f"{label}_perm_{PRIMARY_SCORER}_mean"].to_numpy(dtype=float)
        errs = d[f"{label}_perm_{PRIMARY_SCORER}_std"].to_numpy(dtype=float)
        offset = (0.5 - i) * height
        ax.barh(
            y + offset,
            vals,
            height * 0.9,
            xerr=errs,
            label=label,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.0,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 0.9, "capsize": 2.5},
        )
        if label == PRIMARY_MODEL:
            pad = max(vals) * 0.02
            for yy, v, e in zip(y + offset, vals, errs):
                # Offset past the error-bar cap, not just the bar end, so the
                # label never sits on top of the whisker.
                ax.text(
                    v + e + pad,
                    yy,
                    f"{v:.4f}",
                    va="center",
                    fontsize=7.5,
                    color=INK_SECONDARY,
                )

    ax.set_yticks(y, labels=d["feature"], fontsize=9.5)
    ax.set_xlabel(f"Permutation importance  (drop in test {PRIMARY_SCORER.upper()} when shuffled)")
    ax.set_title(
        "Global feature importance on original features - held-out test set\n"
        f"Permutation importance, {N_REPEATS} repeats, error bars = ±1 std "
        f"(primary model: {PRIMARY_MODEL})",
        fontsize=11.5,
        color=INK_PRIMARY,
    )
    ax.axvline(0, color=BASELINE, linewidth=1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 2. Local explanations
# --------------------------------------------------------------------------- #
def select_cases(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, int]:
    """
    Pick 5 positional indices into the test set: TP, TN, FP, FN, borderline.

    Within each error quadrant we take the record whose predicted probability is
    **closest to that quadrant's median** -- a deliberately typical case rather
        than the most confident one, which would flatter the model. Selection is
    fully deterministic (no sampling), so the same 5 records reappear on re-run.
    The borderline case is the record nearest p=0.50 among those not already
    chosen, so all five are distinct records.
    """
    quadrants = {
        "true_positive": (y_true == 1) & (y_pred == 1),
        "true_negative": (y_true == 0) & (y_pred == 0),
        "false_positive": (y_true == 0) & (y_pred == 1),
        "false_negative": (y_true == 1) & (y_pred == 0),
    }
    chosen: dict[str, int] = {}
    for name, mask in quadrants.items():
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            print(f"[explain] WARNING: no {name} cases in the test set; skipping.")
            continue
        median = float(np.median(y_prob[idx]))
        chosen[name] = int(idx[np.argmin(np.abs(y_prob[idx] - median))])

    remaining = np.setdiff1d(np.arange(len(y_prob)), np.fromiter(chosen.values(), dtype=int))
    chosen["borderline_near_0.5"] = int(remaining[np.argmin(np.abs(y_prob[remaining] - 0.5))])
    return chosen


def local_explanations(
    pipeline, X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
    y_prob: np.ndarray, feature_map: list[str], cases: dict[str, int]
) -> pd.DataFrame:
    """
    Exact TreeSHAP attributions for the selected cases, aggregated to original
    features.

    Contributions are in **log-odds**: a value of +1.2 means that feature pushed
    the log-odds of `>50K` up by 1.2 relative to the model's base rate. They are
    additive, so ``base + sum(contributions) = model log-odds`` -- checked below.

    Privacy note: outputs carry only a synthetic ``case_id``. The underlying
    dataset row index is deliberately NOT written out, so published rows cannot be
    joined back to a specific census record. Feature values are included because
    an attribution is meaningless without the value it attaches to; `sex` and
    `race` appear only because they are model inputs under audit.
    """
    positions = list(cases.values())
    rows_X = X_test.iloc[positions]

    # Transform with the ALREADY-FITTED preprocessor (transform only, never fit).
    X_trans = pipeline.named_steps["preprocessor"].transform(rows_X)
    booster = pipeline.named_steps["model"].get_booster()
    contribs = booster.predict(xgb.DMatrix(X_trans), pred_contribs=True)
    # contribs[:, -1] is the bias / base log-odds; the rest align with X_trans cols.
    base = contribs[:, -1]
    per_column = contribs[:, :-1]

    fmap = np.asarray(feature_map)
    features = list(X_test.columns)

    records = []
    for i, (case_id, pos) in enumerate(cases.items()):
        # Aggregate one-hot contributions back to their source feature (exact:
        # SHAP values are additive, so a sum is the correct aggregation).
        agg = {f: float(per_column[i, fmap == f].sum()) for f in features}

        total = base[i] + sum(agg.values())
        recon = 1.0 / (1.0 + math.exp(-total))
        if abs(recon - y_prob[pos]) > 1e-4:
            raise RuntimeError(
                f"SHAP additivity check failed for {case_id}: "
                f"reconstructed {recon:.6f} vs pipeline {y_prob[pos]:.6f}"
            )

        order = sorted(features, key=lambda f: abs(agg[f]), reverse=True)
        for rank, feat in enumerate(order, start=1):
            records.append(
                {
                    "case_id": case_id,
                    "actual_income": int(y_true[pos]),
                    "predicted_income": int(y_pred[pos]),
                    "predicted_probability": float(y_prob[pos]),
                    "base_log_odds": float(base[i]),
                    "reconstructed_probability": recon,
                    "feature": feat,
                    "feature_value": rows_X.iloc[i][feat],
                    "shap_log_odds": agg[feat],
                    "abs_shap_log_odds": abs(agg[feat]),
                    "factor_rank": rank,
                    "direction": "pushes toward >50K" if agg[feat] > 0
                    else ("pushes toward <=50K" if agg[feat] < 0 else "no effect"),
                }
            )
    print(f"[explain] SHAP additivity verified for all {len(cases)} cases "
          f"(reconstructed probability matches pipeline output to <1e-4).")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# 3. Report
# --------------------------------------------------------------------------- #
def _f(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def write_report(
    imp: pd.DataFrame,
    local: pd.DataFrame,
    coef_df: pd.DataFrame,
    baseline_scores: dict,
    cases: dict[str, int],
) -> Path:
    """Generate ``explainability_report.md`` from the computed tables."""
    L: list[str] = []
    A = L.append
    pm, cm = PRIMARY_MODEL, COMPARISON_MODEL
    pcol = f"{pm}_perm_{PRIMARY_SCORER}_mean"

    A("# Phase 3 — Explainability Audit: Adult Income Baseline\n")
    A(
        f"Explainability audit of the saved **{pm}** pipeline (best Phase-1 held-out "
        f"performance), with **{cm}** as a comparison. Scored on the same "
        f"deterministic held-out test set as Phases 1–2 "
        f"(`test_size={TEST_SIZE}`, `stratify=y`, `random_state={RANDOM_STATE}`).\n"
    )
    A(
        "**Read-only:** models, predictions, Phase-1 metrics and Phase-2 fairness "
        "outputs were loaded but never modified. All artefacts here are written under "
        "`results/explainability/`.\n"
    )

    # ---------------- method
    A("## 1. Method\n")
    A(
        f"- **Permutation importance** on the held-out test set, {N_REPEATS} repeats, "
        f"`random_state={RANDOM_STATE}`. Importance = the drop in score when a single "
        "column is randomly shuffled.\n"
        "- Crucially, permutation is applied to the **raw input columns through the "
        "whole pipeline**, so a shuffled `occupation` is re-encoded by the fitted "
        "preprocessor and the importance is attributed to `occupation` itself — not "
        "spread across its 41 one-hot fragments. All importances below are therefore "
        "**original, human-readable features**.\n"
        f"- Primary scorer **{PRIMARY_SCORER.upper()}** (threshold-free), reported "
        f"alongside accuracy and F1. Baseline test scores: "
        + ", ".join(f"{k} = {_f(v)}" for k, v in baseline_scores.items())
        + ".\n"
        "- **Local attributions** use XGBoost's built-in exact **TreeSHAP** "
        "(`pred_contribs=True`) — no approximation and no extra dependency. Values are "
        "in log-odds and additive, so the one-hot contributions of a feature sum "
        "exactly back to it. Additivity was asserted numerically: "
        "`base + Σ contributions` reproduces each pipeline probability to < 1e-4.\n"
        f"- The dataset has **14 original features**, so \"top {TOP_N}\" is the "
        "complete ranked set rather than a truncation.\n"
    )

    # ---------------- global
    A("## 2. Global feature importance\n")
    A(f"![Global feature importance](global_feature_importance.png)\n")
    A(
        f"Ranked by {pm} permutation importance on test {PRIMARY_SCORER.upper()}. "
        "`± std` is the standard deviation across permutation repeats; where it is "
        "comparable to the mean, the feature is **not distinguishable from "
        "unimportant**.\n"
    )
    A(
        f"| # | feature | {pm} Δ{PRIMARY_SCORER} | ± std | {pm} Δacc | "
        f"{cm} Δ{PRIMARY_SCORER} | XGB native gain share | LR coef spread |"
    )
    A("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, r in imp.head(TOP_N).reset_index(drop=True).iterrows():
        flag = ""
        if r["feature"] in PROTECTED:
            flag = " 🔴"
        elif r["feature"] in LIKELY_PROXIES:
            flag = " 🟡"
        A(
            f"| {i + 1} | `{r['feature']}`{flag} | {_f(r[pcol])} | "
            f"{_f(r[f'{pm}_perm_{PRIMARY_SCORER}_std'])} | "
            f"{_f(r[f'{pm}_perm_accuracy_mean'])} | "
            f"{_f(r[f'{cm}_perm_{PRIMARY_SCORER}_mean'])} | "
            f"{_f(r.get('xgb_native_gain_share'), 3)} | "
            f"{_f(r.get('lr_coef_spread'), 2)} |"
        )
    A("\n🔴 = protected attribute · 🟡 = likely proxy for a protected attribute\n")

    A("### 2.1 What feature importance does and does not prove\n")
    A("**It does establish:**\n")
    A(
        "- **Reliance.** A large drop when `marital-status` is shuffled means *this "
        "fitted model's* test performance depends on that column. That is a fact about "
        "the model, and it is actionable: it tells you what breaks if the feature "
        "degrades in production.\n"
        "- **Redundancy, when importance is low.** A near-zero score means the model "
        "can be denied that column without losing measurable performance.\n"
    )
    A("**It does not establish:**\n")
    A(
        "- **Causation.** Nothing here says being married *causes* higher income. "
        "Importance is a measure of statistical association exploited by one fitted "
        "model on one dataset. Marriage, earnings, age and sex are entangled in the "
        "1994 labour market; permutation importance cannot separate them and is not a "
        "causal method.\n"
        "- **That a low-importance feature is unused or harmless.** Permutation "
        "importance is notoriously **diluted by correlated features**: when two columns "
        "carry the same signal, shuffling either one alone barely moves the score, "
        "because the model recovers the information from its partner. Both then look "
        "unimportant while the *information* remains fully in use. `education` and "
        "`education-num` are a literal duplicate pair here, and `relationship` and "
        "`marital-status` overlap heavily — so their individual scores are **lower "
        "bounds**, not measures of the underlying signal's importance.\n"
        "- **That importance is a fairness measure.** Ranking says nothing about *how* "
        "a feature is used, for whom, or in which direction. A feature can rank low "
        "globally and still drive the decision for a specific subgroup.\n"
        "- **Stability.** Rankings shift with the scorer, the split and the model — "
        "compare the two model columns above. There is no single true importance order.\n"
        "- **Permuting breaks the joint distribution.** Shuffling one column creates "
        "records that could not exist (e.g. `relationship = Wife` with `sex = Male`), "
        "so the model is scored partly off-manifold. This inflates apparent importance "
        "for features locked into strong dependencies with others.\n"
    )

    # ---------------- model comparison
    A(f"## 3. {pm} vs {cm}\n")
    top_lr = coef_df.head(10)
    A(
        "Permutation importance is directly comparable between the two models (same "
        "unit, same test set) and appears side by side in §2. Logistic Regression adds "
        "something the tree model cannot give directly: a **signed, additive** "
        "coefficient per encoded column.\n"
    )
    A("Strongest 10 LR coefficients (log-odds):\n")
    A("| transformed column | original feature | coefficient | direction |")
    A("|---|---|---:|---|")
    for _, r in top_lr.iterrows():
        A(
            f"| `{r['transformed_column']}` | `{r['original_feature']}` | "
            f"{r['coefficient_log_odds']:+.3f} | {r['direction']} |"
        )
    A("")
    A(
        "**Two interpretation caveats.** Numeric coefficients are per **1 standard "
        "deviation** (the numeric branch is standardised), not per natural unit — "
        "a coefficient on `age` is not \"per year\". And because no baseline one-hot "
        "category was dropped, a single dummy coefficient is meaningful only "
        "*relative to the other categories of the same feature*, never as an absolute "
        "effect; that is why the per-feature summary uses the **spread** "
        "(max − min) across a feature's categories. Coefficients are also not "
        "importances: a large coefficient on a rare category moves few predictions.\n"
    )
    agree = imp.head(5)["feature"].tolist()
    lr_rank = imp.sort_values(f"{cm}_perm_{PRIMARY_SCORER}_mean", ascending=False)
    A(
        f"The two models broadly agree on what matters: {pm}'s top 5 is "
        f"`{'`, `'.join(agree)}`, against `{'`, `'.join(lr_rank.head(5)['feature'].tolist())}` "
        f"for {cm}. Agreement between a linear model and a boosted ensemble is "
        "reassuring about the *signal*, but it is not independent corroboration — both "
        "learned from the same features and the same historical labels.\n"
    )

    # ---------------- local
    A("## 4. Local explanations (5 held-out cases)\n")
    A(
        "One representative record per error quadrant, plus one borderline case. "
        "Within each quadrant the record whose predicted probability is **closest to "
        "that quadrant's median** was chosen — a typical case, not the most confident "
        "one, which would flatter the model. Selection is deterministic.\n"
    )
    A(
        "**Privacy.** Only a synthetic `case_id` is published; the underlying dataset "
        "row index is deliberately withheld so these rows cannot be joined back to a "
        "specific census record. Feature values are shown because an attribution is "
        "meaningless without the value it attaches to, and `sex`/`race` appear only "
        "because they are model inputs under audit. This is a public academic benchmark "
        "of 1994 records; the same disclosure in a live system would need a formal "
        "privacy review.\n"
    )
    for case_id in cases:
        c = local[local["case_id"] == case_id]
        if c.empty:
            continue
        head = c.iloc[0]
        A(f"### {case_id}\n")
        A(
            f"- actual = **{int(head['actual_income'])}** "
            f"({'>50K' if head['actual_income'] == 1 else '<=50K'}) · "
            f"predicted = **{int(head['predicted_income'])}** "
            f"({'>50K' if head['predicted_income'] == 1 else '<=50K'}) · "
            f"P(>50K) = **{head['predicted_probability']:.4f}** · "
            f"base log-odds = {head['base_log_odds']:.4f}\n"
        )
        A("| rank | feature | value | SHAP (log-odds) | effect |")
        A("|---:|---|---|---:|---|")
        for _, r in c.head(6).iterrows():
            A(
                f"| {int(r['factor_rank'])} | `{r['feature']}` | {r['feature_value']} | "
                f"{r['shap_log_odds']:+.4f} | {r['direction']} |"
            )
        A("")
        others = c.iloc[6:]["shap_log_odds"].sum()
        A(f"Remaining {len(c) - 6} features together: {others:+.4f} log-odds.\n")
    A(
        "All 14 features are recorded for every case in "
        "`local_explanations.csv`, ranked by absolute contribution.\n"
    )

    # ---------------- governance concerns
    A("## 5. Governance-relevant concerns\n")
    watch = imp.reset_index(drop=True)
    watch["rank"] = watch.index + 1
    A("### 5.1 Are protected attributes and their proxies driving the model?\n")
    A(f"| feature | class | global rank | {pm} Δ{PRIMARY_SCORER} | in any case's top 3? |")
    A("|---|---|---:|---:|---|")
    for feat in PROTECTED + LIKELY_PROXIES:
        row = watch[watch["feature"] == feat]
        if row.empty:
            continue
        row = row.iloc[0]
        in_top3 = local[(local["feature"] == feat) & (local["factor_rank"] <= 3)]["case_id"].tolist()
        A(
            f"| `{feat}` | {'**protected**' if feat in PROTECTED else 'likely proxy'} | "
            f"{int(row['rank'])} | {_f(row[pcol])} | "
            f"{', '.join(in_top3) if in_top3 else '—'} |"
        )
    A("")

    sex_rank = int(watch[watch["feature"] == "sex"]["rank"].iloc[0])
    race_rank = int(watch[watch["feature"] == "race"]["rank"].iloc[0])
    sex_imp = float(watch[watch["feature"] == "sex"][pcol].iloc[0])
    race_imp = float(watch[watch["feature"] == "race"][pcol].iloc[0])
    n_features = len(watch)
    proxy_ranks = {
        f: int(watch[watch["feature"] == f]["rank"].iloc[0])
        for f in LIKELY_PROXIES
        if not watch[watch["feature"] == f].empty
    }
    proxies_above = {f: r for f, r in proxy_ranks.items() if r < min(sex_rank, race_rank)}
    non_proxy_top5 = [
        f for f in watch.head(5)["feature"]
        if f not in LIKELY_PROXIES and f not in PROTECTED
    ]
    A("**What the audit found.**\n")
    A(
        f"- `sex` ranks **{sex_rank}/{n_features}** (Δ{PRIMARY_SCORER} = {_f(sex_imp)}) "
        f"and `race` ranks **{race_rank}/{n_features}** "
        f"(Δ{PRIMARY_SCORER} = {_f(race_imp)}). Both are near the bottom: shuffling them "
        "barely changes test performance.\n"
        f"- **The single most important feature is a likely proxy** — `marital-status` at "
        f"rank 1 — and {len(proxies_above)} of the {len(proxy_ranks)} watch-listed "
        "proxies rank above both protected attributes ("
        + ", ".join(f"`{f}` #{r}" for f, r in sorted(proxies_above.items(), key=lambda kv: kv[1]))
        + ").\n"
        f"- Precision matters here: the ranking is **not** wholly proxy-driven. "
        f"`{'`, `'.join(non_proxy_top5)}` are also top-5 and are financial/age "
        "variables rather than watch-listed proxies. The finding is that proxies "
        "outrank the protected attributes, not that proxies are the only thing the "
        "model uses.\n"
        f"- `relationship` (rank {proxy_ranks.get('relationship', 0)}) deserves specific "
        "attention regardless of its rank: its categories `Husband` and `Wife` are "
        "**sex-coded by construction**, so this feature carries `sex` explicitly, under "
        "another name. That it outranks `sex` itself is the clearest single illustration "
        "of proxy encoding in this model.\n"
        f"- `education` ranks last ({proxy_ranks.get('education', 0)}/{n_features}) with a "
        "slightly **negative** score — shuffling it marginally *helped*. This is not "
        "evidence that education is irrelevant: it is a duplicate of `education-num` "
        f"(rank {proxy_ranks.get('education-num', 0)}), so the model recovers the signal "
        "from the twin column. It is the dilution effect of §2.1 caught in the act, and a "
        "worked example of why a low importance score must never be read as \"unused\".\n"
    )

    A("### 5.2 Association is not causation\n")
    A(
        "Every number in this report is **associational**. Permutation importance and "
        "SHAP both describe how a *fitted function* responds to its inputs on a "
        "*particular dataset*; neither is a causal estimand. \"`marital-status` is the "
        "most important feature\" means the model's accuracy depends on that column — "
        "not that marriage raises income, and not that changing someone's marital "
        "status would change their earnings. A SHAP value of `+1.2` for "
        "`marital-status = Married-civ-spouse` is a statement about the model's "
        "arithmetic, not about the world. Causal claims would require an explicit "
        "causal model and assumptions that this audit neither states nor tests.\n"
    )

    A("### 5.3 Absence from the model does not prove absence of proxy discrimination\n")
    A(
        "This is the central governance point, and the data above demonstrates it "
        "rather than merely asserting it.\n\n"
        f"`sex` ranks {sex_rank}/14 and `race` ranks {race_rank}/14 — so a naive reading "
        "would be \"the model barely uses them; it is fair.\" Phase 2 measured the "
        "opposite: **selection-rate ratios of ~0.30–0.32 for women and three of four "
        "non-reference race groups below the four-fifths threshold**. Low importance and "
        "large disparity coexist without contradiction, for three reasons:\n\n"
        "1. **Redundant encoding.** `relationship` (`Husband`/`Wife`), `marital-status`, "
        "`occupation` and `hours-per-week` jointly reconstruct sex to a large degree. "
        "Once they are present, `sex` adds little *incremental* signal — so permutation "
        "importance, which measures exactly that increment, correctly reports a small "
        "number while the information is fully in use.\n"
        "2. **Deleting a feature does not delete its signal.** Dropping `sex` and `race` "
        "would leave the proxies, and very likely leave the disparities, while removing "
        "the ability to measure them. \"Fairness through unawareness\" fails here for "
        "reasons visible in this table.\n"
        "3. **Global rankings average over people.** A feature can be irrelevant on "
        "average and decisive for a subgroup. Only the disaggregated Phase-2 metrics "
        "and per-case attributions can see that.\n\n"
        "**Therefore: low importance for `sex`/`race` is not evidence of fairness, and "
        "removing them would not be a mitigation.** Proxy discrimination has to be "
        "tested for on outcomes, per group — which is what Phase 2 does — not inferred "
        "from a feature list.\n"
    )

    # ---------------- limitations
    A("## 6. Limitations\n")
    A(
        "- **Correlated-feature dilution.** As above, importances for `education` / "
        "`education-num` and `relationship` / `marital-status` are lower bounds. "
        "Grouped permutation (shuffling correlated features together) would give a "
        "truer picture and was not run.\n"
        "- **Off-manifold scoring.** Permutation creates impossible records "
        "(`relationship = Wife`, `sex = Male`), so part of the measured drop reflects "
        "the model being asked about inputs that cannot occur.\n"
        "- **Single split, single seed.** One 9,769-row test set, `random_state=42`. "
        "Importance ranks — especially adjacent ones with overlapping error bars — are "
        "not stable across resamples; no bootstrap was run.\n"
        "- **Five local cases are illustrative, not representative.** They show *how* "
        "the model reasons in five instances and cannot support any claim about "
        "population-level behaviour or about subgroups.\n"
        "- **SHAP is exact but not unique.** TreeSHAP gives the exact Shapley values "
        "for this model, yet Shapley attribution is one choice among several "
        "(LIME, counterfactuals, integrated gradients) that can rank factors "
        "differently. Exactness of computation is not uniqueness of explanation.\n"
        "- **Log-odds are not probabilities.** A `+1.2` contribution moves the "
        "prediction differently depending on where it starts; contributions are not "
        "percentage points and should not be read as such.\n"
        "- **No interaction analysis.** Only main-effect attributions are aggregated; "
        "SHAP interaction values were not computed, so \"feature A matters only when "
        "B holds\" is invisible here.\n"
        "- **Explaining a model is not validating it.** Nothing in this phase checks "
        "whether the model *should* be used, and a coherent explanation of a biased "
        "model is still an explanation of a biased model.\n"
    )

    # ---------------- governance findings
    A("## 7. Governance findings\n")
    fair_note = ""
    fpath = FAIRNESS_DIR / "fairness_summary.csv"
    if fpath.exists():
        fs = pd.read_csv(fpath)
        x = fs[(fs["model"] == PRIMARY_MODEL)]
        sx = x[x["attribute"] == "sex"].iloc[0]
        rc = x[x["attribute"] == "race"].iloc[0]
        fair_note = (
            f"For {pm}, Phase 2 measured a disparate impact ratio of "
            f"**{sx['disparate_impact_ratio_min']:.3f}** for "
            f"`{sx['disparate_impact_ratio_worst_group']}` vs "
            f"`{sx['reference_group']}`, and **{rc['disparate_impact_ratio_min']:.3f}** for "
            f"`{rc['disparate_impact_ratio_worst_group']}` vs `{rc['reference_group']}`, "
            f"with {int(rc['groups_failing_four_fifths'])} race groups below the "
            "four-fifths threshold. "
        )
    A(
        "Reading Phases 2 and 3 together, and stating each claim at the strength the "
        "evidence supports:\n\n"
        f"1. **Measured disparity and measured feature reliance point in different "
        f"directions, and both are correct.** {fair_note}Phase 3 finds `sex` ranked "
        f"{sex_rank}/14 and `race` ranked {race_rank}/14 in global importance. These are "
        "not in conflict: the disparity is carried by correlated features, not by the "
        "protected attributes as standalone inputs.\n"
        "2. **Features entangled with the protected attributes outrank the protected "
        "attributes themselves.** `marital-status` is the single most important feature, "
        f"and `hours-per-week` (#{proxy_ranks.get('hours-per-week', 0)}), "
        f"`occupation` (#{proxy_ranks.get('occupation', 0)}) and "
        f"`relationship` (#{proxy_ranks.get('relationship', 0)}) all rank above `sex` "
        f"(#{sex_rank}) and `race` (#{race_rank}) — with `relationship` sex-coded in its "
        "own category labels. (Financial and age variables are also top-5, so the "
        "ranking is not exclusively proxy-driven.) This is a coherent "
        "*mechanism-shaped* account of how a model with low `sex` importance can "
        "produce a ~0.32 selection-rate ratio by sex — but it is an association-level "
        "account, and it is **not** a demonstration that these features cause the "
        "disparity.\n"
        "3. **Neither phase demonstrates discrimination, and neither demonstrates "
        "causation.** Phase 2 measured outcome differences; Phase 3 measured model "
        "reliance and per-case attributions. Establishing discrimination would "
        "additionally require a deployment context, a legal or normative standard, and "
        "a causal analysis — none of which exists here. Establishing causation would "
        "require a causal model this audit does not have. The base rates in the labels "
        "themselves differ by group (30.4% vs 11.2% by sex), and no method in either "
        "phase can separate \"the model is unfair\" from \"the 1994 labour market "
        "recorded in the labels was unequal.\"\n"
        "4. **Two things are nonetheless established well enough to act on.** First, "
        "**dropping `sex` and `race` is not a mitigation** — their low importance is "
        "evidence of redundancy, not of irrelevance, and removing them would destroy "
        "measurement while leaving the proxies. Second, **explainability cannot "
        "substitute for disaggregated outcome testing**: an audit that had run only "
        "Phase 3 would have concluded the protected attributes were barely used and "
        "missed the disparity entirely. Both phases are required, and the fairness "
        "metrics remain mutually incompatible (Phase 2 §5.5), so the platform must "
        "still choose and document which definition it commits to.\n"
    )
    A("---\n")
    A(
        "Generated by `src/explainability_audit.py`. Phase 1 models/outputs and Phase 2 "
        "fairness files unmodified.\n"
    )

    path = OUT_DIR / "explainability_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 3 EXPLAINABILITY AUDIT (read-only: Phase 1/2 artefacts not modified)")
    print("=" * 78)

    # ---- 1. reproduce the exact held-out split -------------------------------
    X, y = load_dataset()
    _, X_test, _, y_test = make_split(X, y)  # test_size=0.2, stratify=y, random_state=42

    paths = {
        PRIMARY_MODEL: MODELS_DIR / f"{PRIMARY_MODEL}_pipeline.joblib",
        COMPARISON_MODEL: MODELS_DIR / f"{COMPARISON_MODEL}_pipeline.joblib",
    }
    for name, p in paths.items():
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run `python src/train.py` first.")
    pipes = {name: joblib.load(p) for name, p in paths.items()}
    print(f"[explain] Loaded {', '.join(paths)} (read-only)")

    primary = pipes[PRIMARY_MODEL]
    feature_map = build_feature_map(primary.named_steps["preprocessor"])
    print(f"[explain] Mapped {len(feature_map)} transformed columns -> "
          f"{X_test.shape[1]} original features")

    y_true = np.asarray(y_test)
    y_pred = primary.predict(X_test)
    y_prob = primary.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    baseline_scores = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    print(f"[explain] Baseline test scores: " +
          ", ".join(f"{k}={v:.4f}" for k, v in baseline_scores.items()))

    # ---- 2. global importance ------------------------------------------------
    imp = global_permutation_importance(primary, X_test, y_test, PRIMARY_MODEL)
    imp_lr = global_permutation_importance(
        pipes[COMPARISON_MODEL], X_test, y_test, COMPARISON_MODEL
    )
    imp = imp.merge(imp_lr, on="feature")
    imp = imp.merge(xgb_native_gain(primary, feature_map), on="feature", how="left")

    coef_df = lr_coefficients(pipes[COMPARISON_MODEL], feature_map)
    imp = imp.merge(lr_feature_summary(coef_df), on="feature", how="left")

    imp = imp.sort_values(
        f"{PRIMARY_MODEL}_perm_{PRIMARY_SCORER}_mean", ascending=False
    ).reset_index(drop=True)
    imp.insert(0, "rank", imp.index + 1)

    imp_path = OUT_DIR / "global_feature_importance.csv"
    imp.to_csv(imp_path, index=False)
    print(f"[explain] Global importance -> {imp_path}  ({len(imp)} features)")

    coef_path = OUT_DIR / "logistic_regression_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"[explain] LR coefficients   -> {coef_path}  ({len(coef_df)} columns)")

    chart = plot_global_importance(imp, OUT_DIR / "global_feature_importance.png")
    print(f"[explain] Importance chart  -> {chart}")

    # ---- 3. local explanations ----------------------------------------------
    cases = select_cases(y_true, y_pred, y_prob)
    local = local_explanations(
        primary, X_test, y_true, y_pred, y_prob, feature_map, cases
    )
    local_path = OUT_DIR / "local_explanations.csv"
    local.to_csv(local_path, index=False)
    print(f"[explain] Local explanations -> {local_path}  ({len(local)} rows, "
          f"{len(cases)} cases)")

    # ---- 4. report -----------------------------------------------------------
    report = write_report(imp, local, coef_df, baseline_scores, cases)
    print(f"[explain] Report             -> {report}")

    # ---- console digest ------------------------------------------------------
    print(f"\n=== TOP {TOP_N} ORIGINAL FEATURES ({PRIMARY_MODEL}, "
          f"permutation importance on test {PRIMARY_SCORER.upper()}) ===")
    show = imp.head(TOP_N)[
        [
            "rank", "feature",
            f"{PRIMARY_MODEL}_perm_{PRIMARY_SCORER}_mean",
            f"{PRIMARY_MODEL}_perm_{PRIMARY_SCORER}_std",
            f"{COMPARISON_MODEL}_perm_{PRIMARY_SCORER}_mean",
        ]
    ].rename(
        columns={
            f"{PRIMARY_MODEL}_perm_{PRIMARY_SCORER}_mean": "xgb_dROC_AUC",
            f"{PRIMARY_MODEL}_perm_{PRIMARY_SCORER}_std": "xgb_std",
            f"{COMPARISON_MODEL}_perm_{PRIMARY_SCORER}_mean": "logreg_dROC_AUC",
        }
    )
    print(show.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    print("\n=== LOCAL CASES: top factor per case ===")
    top1 = local[local["factor_rank"] == 1][
        ["case_id", "actual_income", "predicted_income", "predicted_probability",
         "feature", "feature_value", "shap_log_odds"]
    ]
    print(top1.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    print("\nAudit completed successfully. Phase 1/2 artefacts untouched.")


if __name__ == "__main__":
    main()
