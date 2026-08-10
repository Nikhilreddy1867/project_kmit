"""
fairness_audit.py
=================
Phase 2 -- group fairness audit of the Phase-1 income-classification models.

READ-ONLY with respect to Phase 1. This module never trains, never loads a
``.joblib``, and never writes outside ``results/fairness/``. Its only inputs are
the prediction CSVs in ``predictions/``, which already carry ``sex`` and ``race``
alongside the actual label, the predicted label and the predicted probability.

Run:  python src/fairness_audit.py

Outputs (all under results/fairness/)
-------------------------------------
* ``fairness_metrics_by_group.csv`` -- one row per (model, attribute, group):
  absolute performance metrics plus differences/ratios vs. the reference group.
* ``fairness_summary.csv``          -- one row per (model, attribute): the
  headline fairness gaps.
* ``chart_<model>_<attribute>.png`` -- selection rate / TPR / FPR by group.
* ``chart_disparate_impact.png``    -- disparate-impact overview vs the 0.8 rule.
* ``fairness_report.md``            -- narrative report, generated from the data
  above so the numbers can never drift from the tables.

A NOTE ON WHAT THESE NUMBERS ARE
--------------------------------
Everything here is a *measurement of model behaviour on this test set*. A gap is
evidence of a disparity, not proof of discrimination: it may originate in the
model, in the sampling, in the 1994 census labels, or in genuine differences in
the underlying population. The report keeps that distinction explicit, and the
metrics are deliberately reported side by side because they are mathematically
incompatible -- you cannot equalise selection rate and error rates at once when
base rates differ.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
FAIRNESS_DIR = PROJECT_ROOT / "results" / "fairness"

SENSITIVE_ATTRS = ["sex", "race"]

# Groups smaller than this are reported but flagged: their rates have confidence
# intervals wide enough that an apparent gap may be sampling noise.
SMALL_GROUP_THRESHOLD = 200

# The "four-fifths rule" (US EEOC, 29 CFR 1607.4(D)): a selection rate below 80%
# of the reference group's is a conventional screening trigger for adverse impact.
# It is a rule of thumb from employment law, NOT a statistical test and NOT a
# legal verdict on a model.
FOUR_FIFTHS = 0.80

# --------------------------------------------------------------------------- #
# Chart styling -- validated categorical palette (slots 1-3).
# Verified with the dataviz validator: all-pairs CVD deltaE 9.2 (deutan),
# normal-vision 24.0, both modes. Aqua sits below 3:1 on the light surface, so
# every bar carries a visible direct label (the "relief rule") and the same
# numbers exist in the CSVs as a table view.
# --------------------------------------------------------------------------- #
C_SELECTION = "#2a78d6"  # slot 1, blue
C_TPR = "#eb6834"        # slot 2, orange
C_FPR = "#1baf7a"        # slot 3, aqua

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
# Helpers
# --------------------------------------------------------------------------- #
def _safe_div(num: float, den: float) -> float:
    """Return ``num/den``, or NaN when the denominator is empty.

    NaN rather than 0 on purpose: "no positives in this group, so TPR is
    undefined" is a different statement from "TPR is zero", and collapsing the
    two would silently manufacture a fairness gap.
    """
    return float(num) / float(den) if den else math.nan


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """
    Half-width of the 95% Wilson score interval for a proportion.

    Included specifically to make the small-group limitation quantitative: with
    n=69 an observed rate carries roughly +/-10pp of uncertainty, so a 5pp
    "disparity" for such a group is not distinguishable from noise.
    """
    if not n or math.isnan(p):
        return math.nan
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(centre + margin - p, p - (centre - margin)))


def group_metrics(df: pd.DataFrame) -> dict:
    """
    Confusion-matrix-derived metrics for one demographic group.

    Positive class = ``>50K`` (encoded 1). ``recall`` and ``TPR`` are the same
    quantity by definition -- both are reported because the spec asks for both,
    and the report says so rather than implying they are independent checks.
    """
    y_true = df["actual_income"].to_numpy()
    y_pred = df["predicted_income"].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n = len(df)

    tpr = _safe_div(tp, tp + fn)          # recall / sensitivity
    fpr = _safe_div(fp, fp + tn)
    precision = _safe_div(tp, tp + fp)
    selection_rate = _safe_div(tp + fp, n)
    actual_positive_rate = _safe_div(tp + fn, n)
    f1 = _safe_div(2 * precision * tpr, precision + tpr) if not (
        math.isnan(precision) or math.isnan(tpr)
    ) else math.nan

    return {
        "n_samples": n,
        "n_actual_positive": tp + fn,
        "n_actual_negative": fp + tn,
        "actual_positive_rate": actual_positive_rate,
        "selection_rate": selection_rate,          # = predicted positive rate
        "predicted_positive_rate": selection_rate,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": tpr,                             # identical to TPR by definition
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": _safe_div(tp + tn, n),
        "selection_rate_ci95": _wilson_halfwidth(selection_rate, n),
        "tpr_ci95": _wilson_halfwidth(tpr, tp + fn),
        "small_group_flag": n < SMALL_GROUP_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Per-model / per-attribute audit
# --------------------------------------------------------------------------- #
def audit_attribute(df: pd.DataFrame, model: str, attr: str) -> pd.DataFrame:
    """
    Build the per-group table for one sensitive attribute.

    The **reference group is the one with the largest sample count** (per spec).
    That makes the majority group the yardstick -- conventional for disparate
    impact analysis, and it keeps the reference's own metrics the most precisely
    estimated. It is not a claim that the majority group's treatment is correct.
    """
    rows = []
    for value, sub in df.groupby(attr, dropna=False):
        rows.append({"model": model, "attribute": attr, "group": str(value), **group_metrics(sub)})

    out = pd.DataFrame(rows).sort_values("n_samples", ascending=False).reset_index(drop=True)

    ref = out.iloc[0]  # largest group
    out["is_reference"] = out["group"] == ref["group"]
    out["reference_group"] = ref["group"]

    # ---- fairness metrics relative to the reference group --------------------
    # Demographic parity / disparate impact concern SELECTION RATES only -- they
    # ignore the true label entirely, so they can be violated by a model that is
    # perfectly accurate whenever the groups' true base rates differ.
    out["demographic_parity_difference"] = out["selection_rate"] - ref["selection_rate"]
    out["demographic_parity_ratio"] = out["selection_rate"] / ref["selection_rate"]
    # The disparate impact ratio IS the demographic parity ratio -- same formula,
    # different vocabulary (legal vs. ML). Emitted under both names to match the
    # requested schema; it is one measurement, not two.
    out["disparate_impact_ratio"] = out["demographic_parity_ratio"]
    out["fails_four_fifths_rule"] = out["disparate_impact_ratio"] < FOUR_FIFTHS

    # Equal opportunity: TPR gap only -- "among people who really do earn >50K,
    # is the model equally likely to find them?"
    out["equal_opportunity_difference"] = out["tpr"] - ref["tpr"]

    # Equalized odds: BOTH error rates must match, so it is the stricter pair.
    out["equalized_odds_tpr_difference"] = out["tpr"] - ref["tpr"]
    out["equalized_odds_fpr_difference"] = out["fpr"] - ref["fpr"]
    out["equalized_odds_max_difference"] = np.nanmax(
        np.abs(
            np.column_stack(
                [out["equalized_odds_tpr_difference"], out["equalized_odds_fpr_difference"]]
            )
        ),
        axis=1,
    )
    return out


def summarise(by_group: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-group table to one headline row per (model, attribute).

    Two flavours of demographic parity difference are reported:
    * ``*_vs_reference`` -- largest absolute gap against the reference group
      (what the spec asked for);
    * ``*_range``        -- max minus min selection rate across all groups
      (the standard fairlearn definition, reference-free).
    They answer slightly different questions and can differ, so both are given
    rather than picking one and calling it "the" parity difference.
    """
    rows = []
    for (model, attr), g in by_group.groupby(["model", "attribute"], sort=False):
        non_ref = g[~g["is_reference"]]
        ref_row = g[g["is_reference"]].iloc[0]

        def _worst(col: str):
            """Group with the largest absolute value of `col`, plus that value."""
            s = non_ref[col].dropna()
            if s.empty:
                return math.nan, ""
            idx = s.abs().idxmax()
            return float(s.loc[idx]), str(non_ref.loc[idx, "group"])

        dp_diff, dp_group = _worst("demographic_parity_difference")
        eo_diff, eo_group = _worst("equal_opportunity_difference")
        fpr_diff, fpr_group = _worst("equalized_odds_fpr_difference")

        di = non_ref["disparate_impact_ratio"].dropna()
        di_min = float(di.min()) if not di.empty else math.nan
        di_group = str(non_ref.loc[di.idxmin(), "group"]) if not di.empty else ""

        rows.append(
            {
                "model": model,
                "attribute": attr,
                "reference_group": ref_row["group"],
                "reference_n": int(ref_row["n_samples"]),
                "n_groups": int(len(g)),
                "n_total": int(g["n_samples"].sum()),
                "overall_selection_rate": float(
                    (g["selection_rate"] * g["n_samples"]).sum() / g["n_samples"].sum()
                ),
                "demographic_parity_difference_vs_reference": dp_diff,
                "demographic_parity_difference_worst_group": dp_group,
                "demographic_parity_difference_range": float(
                    g["selection_rate"].max() - g["selection_rate"].min()
                ),
                "disparate_impact_ratio_min": di_min,
                "disparate_impact_ratio_worst_group": di_group,
                "groups_failing_four_fifths": int(non_ref["fails_four_fifths_rule"].sum()),
                "equal_opportunity_difference_max_abs": eo_diff,
                "equal_opportunity_worst_group": eo_group,
                "equalized_odds_fpr_difference_max_abs": fpr_diff,
                "equalized_odds_fpr_worst_group": fpr_group,
                "equalized_odds_difference": float(
                    non_ref["equalized_odds_max_difference"].max()
                ) if not non_ref.empty else math.nan,
                "small_groups_present": int(g["small_group_flag"].sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_group_chart(g: pd.DataFrame, model: str, attr: str) -> Path:
    """
    Grouped bars: selection rate, TPR and FPR per demographic group.

    All three series are rates on [0, 1], so they share ONE y-axis -- no second
    scale. Groups are ordered by sample size (reference group first) and each bar
    is directly labelled, which doubles as the contrast relief for the aqua slot.
    """
    series = [
        ("Selection rate", "selection_rate", C_SELECTION),
        ("TPR (recall)", "tpr", C_TPR),
        ("FPR", "fpr", C_FPR),
    ]
    groups = g["group"].tolist()
    x = np.arange(len(groups))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(7.2, 1.9 * len(groups) + 2.4), 5.0))

    for i, (label, col, colour) in enumerate(series):
        offset = (i - 1) * width
        vals = g[col].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            np.nan_to_num(vals),
            width * 0.92,  # 8% of the slot left as a surface gap between bars
            label=label,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.0,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                (0 if math.isnan(v) else v) + 0.012,
                "n/a" if math.isnan(v) else f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_SECONDARY,
            )

    # x labels carry the sample size, and mark the reference + small groups.
    labels = []
    for _, r in g.iterrows():
        tag = "  (ref)" if r["is_reference"] else ""
        warn = "\n! small n" if r["small_group_flag"] else ""
        labels.append(f"{r['group']}{tag}\nn = {int(r['n_samples']):,}{warn}")

    ax.set_xticks(x, labels=labels, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title(
        f"Fairness by {attr} - {model}\n"
        f"Selection rate / TPR / FPR on the held-out test set "
        f"(reference: {g.iloc[0]['group']})",
        fontsize=11,
        color=INK_PRIMARY,
    )
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    path = FAIRNESS_DIR / f"chart_{model}_{attr}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_disparate_impact(by_group: pd.DataFrame) -> Path:
    """Overview: disparate impact ratio per group, per model, against the 0.8 rule."""
    models = by_group["model"].unique().tolist()
    colours = [C_SELECTION, C_TPR, C_FPR]

    fig, axes = plt.subplots(
        1, len(SENSITIVE_ATTRS), figsize=(12.4, 4.9), sharey=True
    )
    for ax, attr in zip(np.atleast_1d(axes), SENSITIVE_ATTRS):
        sub = by_group[by_group["attribute"] == attr]
        groups = (
            sub[sub["model"] == models[0]].sort_values("n_samples", ascending=False)["group"].tolist()
        )
        x = np.arange(len(groups))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            m = sub[sub["model"] == model].set_index("group").reindex(groups)
            vals = m["disparate_impact_ratio"].to_numpy(dtype=float)
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                np.nan_to_num(vals),
                width * 0.9,
                label=model,
                color=colours[i % len(colours)],
                edgecolor=SURFACE,
                linewidth=1.0,
            )
            for rect, v in zip(bars, vals):
                if not math.isnan(v):
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        v + 0.02,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=INK_SECONDARY,
                    )

        ax.axhline(FOUR_FIFTHS, color="#d03b3b", linewidth=1.4, linestyle="--")
        ax.text(
            len(groups) - 0.5,
            FOUR_FIFTHS + 0.03,
            "four-fifths threshold (0.80)",
            ha="right",
            fontsize=7.5,
            color="#d03b3b",
        )
        ax.axhline(1.0, color=BASELINE, linewidth=1.0)
        ax.set_xticks(x, labels=[g.replace("-", "-\n") for g in groups], fontsize=8)
        ax.set_title(f"by {attr}", fontsize=10, color=INK_PRIMARY)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    np.atleast_1d(axes)[0].set_ylabel("Disparate impact ratio\n(selection rate / reference)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Disparate impact ratio vs. reference group (Male / White) - 1.0 = parity",
        fontsize=11.5,
        color=INK_PRIMARY,
    )
    fig.tight_layout()

    path = FAIRNESS_DIR / "chart_disparate_impact.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _f(v, nd: int = 3, pct: bool = False) -> str:
    """Format a float for markdown, keeping NaN visible as 'n/a'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:+.1f}pp" if pct else f"{v:.{nd}f}"


def write_report(by_group: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Generate ``fairness_report.md`` from the computed tables."""
    models = by_group["model"].unique().tolist()
    n_test = int(by_group[(by_group["model"] == models[0]) & (by_group["attribute"] == "sex")][
        "n_samples"
    ].sum())

    L: list[str] = []
    A = L.append

    A("# Phase 2 — Fairness Audit: Adult Income Models\n")
    A(
        f"Group fairness audit of the {len(models)} Phase-1 models "
        f"(`{'`, `'.join(models)}`) on the held-out test set of "
        f"**{n_test:,} records**, disaggregated by **sex** and **race**.\n"
    )
    A(
        "This audit is **read-only**: it consumes the prediction CSVs written in "
        "Phase 1 and does not retrain, reload or modify any model. Every number "
        "below is reproducible from `results/fairness/fairness_metrics_by_group.csv`.\n"
    )

    # ---------------- method
    A("## 1. Method\n")
    A(
        "- **Positive class** = `>50K` (encoded 1). A \"selection\" means the model "
        "predicted a person into the high-income class.\n"
        "- **Reference groups** are the largest by sample count, per spec: "
        f"**{summary[summary.attribute == 'sex'].iloc[0]['reference_group']}** for sex "
        f"(n = {int(summary[summary.attribute == 'sex'].iloc[0]['reference_n']):,}) and "
        f"**{summary[summary.attribute == 'race'].iloc[0]['reference_group']}** for race "
        f"(n = {int(summary[summary.attribute == 'race'].iloc[0]['reference_n']):,}). "
        "Choosing the majority group as the yardstick is conventional for disparate-impact "
        "analysis and gives the most precisely estimated reference. It is **not** an "
        "assertion that the majority group's treatment is correct or desirable.\n"
        "- **Decision threshold** = 0.5, inherited unchanged from Phase 1.\n"
        "- `recall` and `TPR` are the same quantity by definition; both appear because "
        "the schema asks for both. Likewise the **disparate impact ratio** and the "
        "**demographic parity ratio** are one formula "
        "(group selection rate ÷ reference selection rate) under two vocabularies — "
        "one measurement, reported under both names.\n"
        "- 95% **Wilson score intervals** accompany selection rate and TPR so that a "
        "gap can be compared against its own sampling uncertainty.\n"
    )

    # ---------------- per-model tables
    A("## 2. Observed group metrics\n")
    for attr in SENSITIVE_ATTRS:
        A(f"### 2.{SENSITIVE_ATTRS.index(attr) + 1} By `{attr}`\n")
        for model in models:
            g = by_group[(by_group["model"] == model) & (by_group["attribute"] == attr)]
            A(f"**{model}**\n")
            A(
                "| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | "
                "DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |"
            )
            A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in g.iterrows():
                name = f"**{r['group']}** (ref)" if r["is_reference"] else r["group"]
                if r["small_group_flag"]:
                    name += " ⚠"
                A(
                    f"| {name} | {int(r['n_samples']):,} | {_f(r['actual_positive_rate'])} | "
                    f"{_f(r['selection_rate'])} | {_f(r['tpr'])} | {_f(r['fpr'])} | "
                    f"{_f(r['precision'])} | {_f(r['f1'])} | "
                    f"{_f(r['demographic_parity_difference'], pct=True)} | "
                    f"{_f(r['disparate_impact_ratio'], 3)} | "
                    f"{_f(r['equal_opportunity_difference'], pct=True)} | "
                    f"{_f(r['equalized_odds_fpr_difference'], pct=True)} |"
                )
            A("")
        A("⚠ = group smaller than "
          f"{SMALL_GROUP_THRESHOLD} test records; see limitations.\n")

    # ---------------- summary
    A("## 3. Headline gaps\n")
    A(
        "| model | attribute | ref | worst DP diff (group) | min DI ratio (group) | "
        "worst EqOpp diff (group) | worst EqOdds FPR diff | EqOdds diff | # failing 4/5 |"
    )
    A("|---|---|---|---|---|---|---|---:|---:|")
    for _, r in summary.iterrows():
        A(
            f"| {r['model']} | {r['attribute']} | {r['reference_group']} | "
            f"{_f(r['demographic_parity_difference_vs_reference'], pct=True)} "
            f"({r['demographic_parity_difference_worst_group']}) | "
            f"{_f(r['disparate_impact_ratio_min'])} ({r['disparate_impact_ratio_worst_group']}) | "
            f"{_f(r['equal_opportunity_difference_max_abs'], pct=True)} "
            f"({r['equal_opportunity_worst_group']}) | "
            f"{_f(r['equalized_odds_fpr_difference_max_abs'], pct=True)} "
            f"({r['equalized_odds_fpr_worst_group']}) | "
            f"{_f(r['equalized_odds_difference'])} | "
            f"{int(r['groups_failing_four_fifths'])} |"
        )
    A("")

    # ---------------- observed vs conclusion
    A("## 4. Observed disparities vs. conclusions about discrimination\n")
    A("### 4.1 What the numbers *do* establish\n")
    sex_rows = summary[summary["attribute"] == "sex"]
    race_rows = summary[summary["attribute"] == "race"]
    A(
        "These are direct measurements of model output on this test set, and they are "
        "not in dispute:\n"
    )
    A(
        f"1. **Every model selects women at a far lower rate than men.** The disparate "
        f"impact ratio for `Female` ranges "
        f"{sex_rows['disparate_impact_ratio_min'].min():.3f}–"
        f"{sex_rows['disparate_impact_ratio_min'].max():.3f} across the three models — all "
        f"well below the 0.80 four-fifths screening threshold.\n"
        f"2. **Every model finds true high-earning women less often than true "
        f"high-earning men** (negative equal-opportunity difference in all "
        f"{len(sex_rows)} model/sex rows).\n"
        f"3. **Race-based selection-rate gaps are present but smaller**, and are not "
        f"uniform across groups: the worst race DI ratio per model is "
        f"{race_rows['disparate_impact_ratio_min'].min():.3f}–"
        f"{race_rows['disparate_impact_ratio_min'].max():.3f}.\n"
        "4. **False-positive rates are consistently lower for the disadvantaged groups.** "
        "The models are *less* likely to wrongly promote someone into `>50K` in those "
        "groups — the flip side of selecting them less often. A single-metric audit "
        "would have missed this.\n"
    )

    A("### 4.2 What the numbers do *not* establish\n")
    A(
        "**None of the above is, by itself, proof of discrimination.** Specifically:\n\n"
        "- **A disparity is not a mechanism.** These metrics say *what* differs, never "
        "*why*. Distinguishing \"the model treats similar people differently\" from "
        "\"the groups differ in the recorded features\" requires counterfactual or "
        "causal analysis that is out of scope here.\n"
        "- **The base rates genuinely differ in the data.** In this test set the actual "
        f"`>50K` rate is "
        f"{by_group[(by_group.model == models[0]) & (by_group.group == 'Male')]['actual_positive_rate'].iloc[0]:.3f} "
        f"for men and "
        f"{by_group[(by_group.model == models[0]) & (by_group.group == 'Female')]['actual_positive_rate'].iloc[0]:.3f} "
        "for women. A model that merely reproduces that gap will fail demographic "
        "parity while being, in a narrow accuracy sense, \"right\". Whether reproducing "
        "a historical gap is acceptable is a policy question, not a statistical one.\n"
        "- **The label is not ground truth about merit.** `income > 50K` in 1994 census "
        "data records the outcome of a labour market that was itself unequal. Learning "
        "it faithfully means inheriting that inequality — the model can be an accurate "
        "predictor of a biased world.\n"
        "- **Legal thresholds are screening tools, not verdicts.** Failing the "
        "four-fifths rule triggers scrutiny under US employment-law convention; it is "
        "not a statistical test, and it does not establish unlawful discrimination.\n"
        "- **This dataset is not a hiring or lending system.** Adult/Census Income is a "
        "benchmark. Real-world harm depends on deployment context, which does not exist "
        "here.\n"
    )

    # ---------------- limitations
    A("## 5. Limitations\n")
    small = by_group[by_group["small_group_flag"]][["attribute", "group", "n_samples"]].drop_duplicates()
    A("### 5.1 Small group sizes\n")
    A(
        "Group metrics are only as stable as the group is large. Flagged groups "
        f"(n < {SMALL_GROUP_THRESHOLD}):\n"
    )
    A("| attribute | group | n | 95% CI half-width on selection rate |")
    A("|---|---|---:|---:|")
    for _, r in small.iterrows():
        ci = by_group[
            (by_group["attribute"] == r["attribute"]) & (by_group["group"] == r["group"])
        ]["selection_rate_ci95"].mean()
        A(f"| {r['attribute']} | {r['group']} | {int(r['n_samples']):,} | ±{ci * 100:.1f}pp |")
    A("")
    A(
        "For these groups a difference of several percentage points is not "
        "distinguishable from sampling noise, and TPR is worse still because it is "
        "computed on the handful of actual positives only. **Do not rank these groups "
        "by their point estimates.** Any conclusion about them needs bootstrap "
        "intervals or a larger sample.\n"
    )

    A("### 5.2 Historical bias in census data\n")
    A(
        "The data is the 1994 US Census extract. The label encodes real historical "
        "wage inequality between demographic groups, so the target itself is "
        "value-laden. Correcting the model cannot correct the label, and reported "
        "\"accuracy\" measures fidelity to a biased world, not fairness in it. Nothing "
        "here transfers unexamined to the present day.\n"
    )

    A("### 5.3 Proxy variables\n")
    A(
        "Removing `sex` and `race` from the features would **not** make the model "
        "blind to them. `relationship` contains the sex-coded categories `Husband` and "
        "`Wife`; `marital-status`, `occupation`, `hours-per-week`, `education` and "
        "`native-country` all correlate with the protected attributes and jointly "
        "reconstruct them. Fairness-through-unawareness therefore removes the ability "
        "to *measure* disparity without removing the disparity — which is exactly why "
        "Phase 1 retained both attributes.\n"
    )

    A("### 5.4 The 0.5 decision threshold\n")
    A(
        "Every disparity above is measured at the default 0.5 cut-off, which was never "
        "chosen for fairness — or for anything else. Selection rates, TPR and FPR are "
        "all threshold-dependent, so these gaps would change, and could partly close, "
        "under a different or per-group threshold. The ranking-quality metric (ROC-AUC, "
        "Phase 1) is threshold-free and would not move. Treat threshold choice as an "
        "open governance decision, and note that per-group thresholds are themselves "
        "legally and ethically contested.\n"
    )

    A("### 5.5 Fairness metrics conflict — mathematically, not incidentally\n")
    A(
        "The metrics in this report cannot all be satisfied at once. When base rates "
        "differ between groups (they do here), it is a **proven impossibility** that a "
        "classifier equalises demographic parity, equalised odds and calibration "
        "simultaneously except in degenerate cases (Kleinberg et al. 2016; Chouldechova "
        "2017). This audit is visible evidence of the trade-off: the same groups that "
        "are selected less often also receive fewer false positives. Forcing selection "
        "rates to parity would raise their false-positive rates. **There is no "
        "configuration that is fair on every metric**, so the platform must state which "
        "definition it is committing to, and why, rather than optimising a single number.\n"
    )

    A("### 5.6 Further scope limits\n")
    A(
        "- **No intersectional analysis.** `sex` and `race` are audited "
        "one at a time; a Black-female subgroup gap could be larger than either "
        "marginal gap and would be invisible here. Cell sizes shrink fast, which is "
        "the reason it was not attempted rather than an argument that it does not matter.\n"
        "- **Point estimates only** for the headline gaps; no significance testing or "
        "bootstrap CIs on the *differences* themselves.\n"
        "- **`race` uses the dataset's own five fixed categories**, which are coarse, "
        "of their time, and not self-identified in any modern sense. `sex` is recorded "
        "as a binary, which erases non-binary people entirely — an artefact of the 1994 "
        "instrument, and a limitation of the audit as much as of the data.\n"
        "- **Single split.** All numbers come from one 9,769-row test set with "
        "`random_state=42`; they are not averaged over repeated splits.\n"
        "- **No calibration analysis** (per-group reliability of "
        "`predicted_probability`), which is the third leg of the impossibility result "
        "and a natural next step.\n"
    )

    A("## 6. Recommended next steps\n")
    A(
        "1. **Choose and document a fairness criterion** before mitigating anything — "
        "the metrics conflict, so this is a policy decision, not an optimisation.\n"
        "2. **Add per-group calibration curves** for `predicted_probability`.\n"
        "3. **Bootstrap the gaps** to get confidence intervals on the differences, "
        "which is what the small groups actually need.\n"
        "4. **Sweep the decision threshold** and plot the fairness/accuracy frontier "
        "instead of accepting 0.5.\n"
        "5. **Then** evaluate mitigations (reweighing, threshold optimisation, "
        "constrained training), re-auditing after each — and re-check that a mitigation "
        "does not simply move the harm to another metric or group.\n"
    )
    A("---\n")
    A(
        "Generated by `src/fairness_audit.py`. Phase 1 models and outputs unmodified.\n"
    )

    path = FAIRNESS_DIR / "fairness_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(PREDICTIONS_DIR.glob("*_test_predictions.csv"))
    if not pred_files:
        raise SystemExit(
            f"No prediction CSVs found in {PREDICTIONS_DIR}. Run `python src/train.py` first."
        )

    print("=" * 78)
    print("PHASE 2 FAIRNESS AUDIT (read-only: Phase 1 artefacts are not modified)")
    print("=" * 78)

    all_rows = []
    for f in pred_files:
        model = f.stem.replace("_test_predictions", "")
        df = pd.read_csv(f)
        missing = {"actual_income", "predicted_income", *SENSITIVE_ATTRS} - set(df.columns)
        if missing:
            raise SystemExit(f"{f.name} is missing required columns: {sorted(missing)}")

        print(f"\n[audit] {model}: {len(df):,} test predictions")
        for attr in SENSITIVE_ATTRS:
            g = audit_attribute(df, model, attr)
            all_rows.append(g)
            worst = g[~g["is_reference"]]
            if not worst.empty:
                di = worst["disparate_impact_ratio"].dropna()
                if not di.empty:
                    j = di.idxmin()
                    print(
                        f"         {attr}: ref={g.iloc[0]['group']} | "
                        f"worst DI ratio {di.min():.3f} ({worst.loc[j, 'group']}) | "
                        f"{int(worst['fails_four_fifths_rule'].sum())}/{len(worst)} "
                        f"group(s) below 0.80"
                    )
            chart = plot_group_chart(g, model, attr)
            print(f"         chart -> {chart}")

    by_group = pd.concat(all_rows, ignore_index=True)

    # ---- column order: identity, absolute metrics, then relative metrics -----
    cols = [
        "model", "attribute", "group", "is_reference", "reference_group",
        "n_samples", "n_actual_positive", "n_actual_negative",
        "actual_positive_rate", "selection_rate", "predicted_positive_rate",
        "tpr", "fpr", "precision", "recall", "f1", "accuracy",
        "true_positives", "false_positives", "false_negatives", "true_negatives",
        "selection_rate_ci95", "tpr_ci95", "small_group_flag",
        "demographic_parity_difference", "demographic_parity_ratio",
        "disparate_impact_ratio", "fails_four_fifths_rule",
        "equal_opportunity_difference",
        "equalized_odds_tpr_difference", "equalized_odds_fpr_difference",
        "equalized_odds_max_difference",
    ]
    by_group = by_group[cols]

    by_group_path = FAIRNESS_DIR / "fairness_metrics_by_group.csv"
    by_group.to_csv(by_group_path, index=False)
    print(f"\n[audit] Per-group metrics -> {by_group_path}  ({len(by_group)} rows)")

    summary = summarise(by_group)
    summary_path = FAIRNESS_DIR / "fairness_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[audit] Summary            -> {summary_path}  ({len(summary)} rows)")

    di_chart = plot_disparate_impact(by_group)
    print(f"[audit] Overview chart     -> {di_chart}")

    report = write_report(by_group, summary)
    print(f"[audit] Report             -> {report}")

    # ---- console digest -----------------------------------------------------
    print("\n=== HEADLINE FAIRNESS GAPS ===")
    show = [
        "model", "attribute", "reference_group",
        "demographic_parity_difference_vs_reference",
        "disparate_impact_ratio_min", "disparate_impact_ratio_worst_group",
        "equal_opportunity_difference_max_abs",
        "equalized_odds_fpr_difference_max_abs",
        "groups_failing_four_fifths",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary[show].to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    print("\nAudit completed successfully. Phase 1 models/outputs untouched.")


if __name__ == "__main__":
    main()
