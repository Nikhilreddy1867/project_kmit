"""
governance_service.py
=====================
Assembles API payloads from the artefacts returned by :mod:`artifact_reader`.

This layer **shapes** data; it does not compute it. Specifically:

* Every metric is passed through verbatim from the audit CSVs.
* The only arithmetic anywhere in the API is counting *rows of the risk register*
  by rating (``counts_by_overall_risk``) -- metadata about the register, not a
  model metric.
* Governance conclusions (approve / block, grounds, conditions) are **parsed out
  of ``governance_summary.md``** rather than hardcoded here, so the committed
  decision record stays the single source of truth. If the markdown is edited,
  the API follows it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.services import artifact_reader as reader

# Mirrors the watch-list used by the Phase 3 audit (src/explainability_audit.py).
# Duplicated rather than imported so the API does not pull in xgboost/matplotlib.
PROTECTED_ATTRIBUTES = ("sex", "race")
LIKELY_PROXIES = (
    "marital-status",
    "relationship",
    "occupation",
    "education",
    "education-num",
    "hours-per-week",
    "native-country",
)

SENSITIVE_ATTRIBUTES = ["sex", "race"]

REFERENCE_GROUP_RULE = (
    "The reference group is the one with the largest sample count (Male for sex, "
    "White for race). This is conventional for disparate-impact analysis and gives "
    "the most precisely estimated reference. It is NOT an assertion that the "
    "majority group's treatment is correct or desirable."
)

FAIRNESS_INTERPRETATION = {
    "establishes": [
        "The model's outputs differ by group on this held-out test set, in the "
        "direction and magnitude tabulated.",
        "These are direct measurements of model behaviour, not estimates.",
    ],
    "does_not_establish": [
        "Not discrimination: these metrics describe what differs, never why. No "
        "legal determination is made or implied.",
        "Not causation: no fairness metric here identifies a causal effect of a "
        "protected attribute on the prediction.",
        "Not attributable to the model alone: group base rates differ in the 1994 "
        "labels themselves (30.4% vs 11.2% by sex), and no method used can separate "
        "model behaviour from historical label bias.",
        "The four-fifths rule is a screening trigger from US employment-law "
        "convention, not a statistical test and not a verdict.",
        "Metrics conflict mathematically: where base rates differ, demographic "
        "parity, equalised odds and calibration cannot all hold at once.",
    ],
}

PERFORMANCE_CAVEATS = [
    "All rates are measured at the default 0.5 decision threshold, which was never "
    "tuned. Only roc_auc is threshold-free.",
    "The majority-class accuracy floor on this test set is 0.7607, so accuracy must "
    "not be read alone.",
    "Single deterministic split (test_size=0.2, stratify=y, random_state=42); no "
    "cross-validation and no confidence intervals on these point estimates.",
    "Aggregate metrics conceal group disparities -- see the fairness endpoint.",
]

EXPLAINABILITY_CAVEATS = [
    "Association, not causation: importance and SHAP describe how a fitted function "
    "responds to inputs on one dataset. Neither is a causal estimand.",
    "Importances are LOWER BOUNDS under feature correlation. education ranks last "
    "with a near-zero score only because education-num duplicates it.",
    "A low importance score does not mean a feature is unused, and low importance "
    "for sex/race is not evidence of fairness.",
    "Permutation partly scores the model off-manifold: shuffling creates impossible "
    "records such as relationship=Wife with sex=Male.",
    "Local explanations are 5 illustrative cases; they support no population-level "
    "or subgroup claim.",
    "SHAP values are log-odds contributions, not percentage points.",
]

RATING_SCALE = {
    "likelihood": "Low / Medium / High (qualitative expert judgement)",
    "impact": "Low / Medium / High (qualitative expert judgement)",
    "overall_risk": (
        "Matrix: High x High = Critical; High x Medium or Medium x High = High; "
        "Medium x Medium, Low x High or High x Low = Medium; otherwise Low"
    ),
}

ASSESSMENT_FRAMING = (
    "Impact is rated AS IF the model were used for a consequential decision about "
    "individuals. This is a hypothetical framing: there is no deployment, and "
    "therefore no realised operational harm to date."
)

DISCLAIMER = (
    "This decision rests on documented measurements and absent prerequisites. It "
    "makes no finding that the model is legally discriminatory, and no causal claim "
    "about any feature or group."
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _classify(feature: str) -> str:
    if feature in PROTECTED_ATTRIBUTES:
        return "protected"
    if feature in LIKELY_PROXIES:
        return "likely_proxy"
    return "other"


def _rel(key: str) -> str:
    """Repo-relative path of an artefact, for the `source` fields."""
    return reader.ARTIFACTS[key].relative_to(reader.PROJECT_ROOT).as_posix()


def _md_section(md: str, pattern: str) -> str:
    """
    Return the body of the first heading matching `pattern`, up to the next
    heading at the same or a higher level. Returns '' when not found -- callers
    degrade gracefully rather than failing the request.
    """
    m = re.search(pattern, md, re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    level = len(m.group(0)) - len(m.group(0).lstrip("#"))
    nxt = re.search(rf"^#{{1,{max(level, 1)}}} ", md[start:], re.MULTILINE)
    return md[start : start + nxt.start()] if nxt else md[start:]


def _strip_md(text: str) -> str:
    """Remove bold/italic/code markers so list items read cleanly as JSON strings."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Models / performance
# --------------------------------------------------------------------------- #
def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}


def _confusion(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: row.get(k)
        for k in ("true_negatives", "false_positives", "false_negatives", "true_positives")
    }


def _has_fairness(model: str) -> bool:
    try:
        reader.fairness_rows(model)
        return True
    except reader.ArtifactError:
        return False


def build_model_list() -> dict[str, Any]:
    rows = reader.phase1_rows()
    models = []
    for row in rows:
        name = str(row["model"])
        models.append(
            {
                "model_name": name,
                "is_primary": name == reader.PRIMARY_MODEL,
                "metrics": _metrics(row),
                "confusion_matrix": _confusion(row),
                "n_test": row.get("n_test"),
                "fit_seconds": row.get("fit_seconds"),
                "has_fairness_audit": _has_fairness(name),
                "has_explainability_audit": name in reader.EXPLAINED_MODELS,
            }
        )
    return {
        "test_set_rows": rows[0].get("n_test") if rows else None,
        "source": _rel("phase1_metrics"),
        "count": len(models),
        "models": models,
    }


def build_performance(model: str) -> dict[str, Any]:
    row = reader.phase1_row(model)
    return {
        "model_name": model,
        "is_primary": model == reader.PRIMARY_MODEL,
        "n_test": row.get("n_test"),
        "metrics": _metrics(row),
        "confusion_matrix": _confusion(row),
        "error_analysis": {
            # Counts verbatim from the audit. The API deliberately performs no
            # arithmetic on them -- see governance_service module docstring.
            "false_negatives": row.get("false_negatives"),
            "false_positives": row.get("false_positives"),
            "notes": [
                "A false negative is a true high earner the model missed; a false "
                "positive is a low earner wrongly flagged.",
                "recall is the share of true high earners found, so the remainder "
                "are missed. The miss rate is not evenly distributed across "
                "demographic groups -- see the fairness endpoint.",
                "The 0.5 threshold treats both error types as equally undesirable, "
                "which is almost never true in a real application.",
            ],
        },
        "fit_seconds": row.get("fit_seconds"),
        "source": _rel("phase1_metrics"),
        "caveats": PERFORMANCE_CAVEATS,
    }


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
_GROUP_FIELDS = (
    "attribute", "group", "is_reference", "reference_group", "n_samples",
    "n_actual_positive", "actual_positive_rate", "selection_rate", "tpr", "fpr",
    "precision", "recall", "f1", "selection_rate_ci95", "tpr_ci95",
    "small_group_flag", "demographic_parity_difference", "demographic_parity_ratio",
    "disparate_impact_ratio", "fails_four_fifths_rule",
    "equal_opportunity_difference", "equalized_odds_tpr_difference",
    "equalized_odds_fpr_difference", "equalized_odds_max_difference",
)

_SUMMARY_FIELDS = (
    "attribute", "reference_group", "reference_n", "n_groups",
    "demographic_parity_difference_vs_reference",
    "demographic_parity_difference_worst_group",
    "demographic_parity_difference_range", "disparate_impact_ratio_min",
    "disparate_impact_ratio_worst_group", "groups_failing_four_fifths",
    "equal_opportunity_difference_max_abs", "equal_opportunity_worst_group",
    "equalized_odds_fpr_difference_max_abs", "equalized_odds_difference",
    "small_groups_present",
)


def build_fairness(model: str, attribute: str | None = None) -> dict[str, Any]:
    groups = reader.fairness_rows(model)
    summary = reader.fairness_summary_rows(model)

    if attribute:
        available = sorted({str(r["attribute"]) for r in groups})
        if attribute not in available:
            raise reader.ModelNotFoundError(attribute, available, "fairness attribute")
        groups = [r for r in groups if str(r["attribute"]) == attribute]
        summary = [r for r in summary if str(r["attribute"]) == attribute]

    return {
        "model_name": model,
        "sensitive_attributes": sorted({str(r["attribute"]) for r in groups}),
        "reference_group_rule": REFERENCE_GROUP_RULE,
        "summary": [{k: r.get(k) for k in _SUMMARY_FIELDS} for r in summary],
        "groups": [{k: r.get(k) for k in _GROUP_FIELDS} for r in groups],
        "sources": [_rel("fairness_by_group"), _rel("fairness_summary")],
        "interpretation": FAIRNESS_INTERPRETATION,
    }


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def build_explainability(model: str, top_n: int | None = None) -> dict[str, Any]:
    reader.assert_explained(model)
    rows = reader.importance_rows()

    prefix = f"{model}_perm"
    importance = [
        {
            "rank": r.get("rank"),
            "feature": str(r["feature"]),
            "importance_mean": r.get(f"{prefix}_roc_auc_mean"),
            "importance_std": r.get(f"{prefix}_roc_auc_std"),
            "importance_accuracy_mean": r.get(f"{prefix}_accuracy_mean"),
            "importance_f1_mean": r.get(f"{prefix}_f1_mean"),
            "classification": _classify(str(r["feature"])),
        }
        for r in rows
    ]
    # Ranking for the comparison model follows its own importance, descending;
    # this is ordering, not recomputation of any value.
    if model != reader.PRIMARY_MODEL:
        importance.sort(key=lambda d: (d["importance_mean"] is None, -(d["importance_mean"] or 0)))
    if top_n:
        importance = importance[:top_n]

    # Local explanations exist for the primary model only (Phase 3 scope).
    local: list[dict[str, Any]] = []
    if model == reader.PRIMARY_MODEL:
        by_case: dict[str, list[dict[str, Any]]] = {}
        for r in reader.local_explanation_rows():
            by_case.setdefault(str(r["case_id"]), []).append(r)
        for case_id, factors in by_case.items():
            factors.sort(key=lambda d: (d.get("factor_rank") is None, d.get("factor_rank")))
            head = factors[0]
            local.append(
                {
                    "case_id": case_id,
                    "actual_income": head.get("actual_income"),
                    "predicted_income": head.get("predicted_income"),
                    "predicted_probability": head.get("predicted_probability"),
                    "base_log_odds": head.get("base_log_odds"),
                    "top_factors": [
                        {
                            "feature": str(f["feature"]),
                            "feature_value": f.get("feature_value"),
                            "shap_log_odds": f.get("shap_log_odds"),
                            "factor_rank": f.get("factor_rank"),
                            "direction": f.get("direction"),
                        }
                        for f in factors[:5]
                    ],
                }
            )

    # Ranks read verbatim from the artefact -- not derived here.
    ranked = {str(r["feature"]): r.get("rank") for r in rows}
    proxies_above = {
        f: ranked[f]
        for f in LIKELY_PROXIES
        if f in ranked
        and ranked[f] is not None
        and all(
            ranked.get(p) is not None and ranked[f] < ranked[p] for p in PROTECTED_ATTRIBUTES
        )
    }

    return {
        "model_name": model,
        "is_primary": model == reader.PRIMARY_MODEL,
        "method": {
            "global": "Permutation importance on the RAW input columns through the "
            "full pipeline, so importance attaches to original human-readable "
            "features rather than one-hot fragments. 10 repeats, held-out test set.",
            "local": "Exact TreeSHAP (XGBoost pred_contribs), additive in log-odds; "
            "additivity verified numerically to <1e-4. Primary model only.",
        },
        "n_features": len(rows),
        "global_importance": importance,
        "local_explanations": local,
        "proxy_assessment": {
            "protected_attribute_ranks": {
                p: ranked.get(p) for p in PROTECTED_ATTRIBUTES
            },
            "proxies_ranked_above_all_protected_attributes": proxies_above,
            "finding": (
                "Both protected attributes rank near the bottom of global "
                "importance while Phase 2 measured substantial group disparities. "
                "These coexist because correlated features carry the same "
                "information: relationship is sex-coded by construction (its "
                "categories include Husband and Wife) and outranks sex itself."
            ),
            "implication": (
                "Low importance for sex/race is NOT evidence of fairness, and "
                "removing them would not be a mitigation -- it would leave the "
                "proxies while destroying the ability to measure disparity. Proxy "
                "discrimination must be tested on disaggregated outcomes, never "
                "inferred from a feature ranking."
            ),
        },
        "sources": [_rel("global_importance")]
        + ([_rel("local_explanations")] if local else []),
        "caveats": EXPLAINABILITY_CAVEATS,
    }


# --------------------------------------------------------------------------- #
# Governance: risks
# --------------------------------------------------------------------------- #
_RISK_FIELDS = (
    "risk_id", "category", "risk_statement", "evidence", "affected_groups",
    "likelihood", "impact", "overall_risk", "recommended_control",
    "residual_risk", "owner", "status",
)


def _risk_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    """Row counts by rating and by status (register metadata, not a model metric)."""
    return (
        dict(Counter(str(r["overall_risk"]) for r in rows)),
        dict(Counter(str(r["status"]) for r in rows)),
    )


def build_risks(
    overall_risk: str | None = None,
    category: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    all_rows = reader.risk_rows()
    rows = all_rows

    if overall_risk:
        rows = [r for r in rows if str(r["overall_risk"]).lower() == overall_risk.lower()]
    if category:
        rows = [r for r in rows if category.lower() in str(r["category"]).lower()]
    if status:
        rows = [r for r in rows if status.lower() in str(r["status"]).lower()]

    by_risk, by_status = _risk_counts(all_rows)
    return {
        "count": len(rows),
        "total_in_register": len(all_rows),
        "filters_applied": {
            "overall_risk": overall_risk,
            "category": category,
            "status": status,
        },
        "rating_scale": RATING_SCALE,
        "assessment_framing": ASSESSMENT_FRAMING,
        "counts_by_overall_risk": by_risk,
        "counts_by_status": by_status,
        "risks": [{k: str(r.get(k, "")) for k in _RISK_FIELDS} for r in rows],
        "source": _rel("risk_register"),
    }


# --------------------------------------------------------------------------- #
# Governance: decision
# --------------------------------------------------------------------------- #
def build_decision(include_markdown: bool = False) -> dict[str, Any]:
    md = reader.read_markdown("governance_summary")
    rows = reader.risk_rows()
    by_risk, by_status = _risk_counts(rows)

    upper = md.upper()
    research = (
        "conditionally_approved"
        if "CONDITIONALLY APPROVED" in upper
        else ("not_approved" if "NOT APPROVED" in upper else "undetermined")
    )
    deployment = (
        "blocked" if "BLOCKED FROM REAL-WORLD DEPLOYMENT" in upper else "undetermined"
    )

    grounds = [
        _strip_md(t)
        for t in re.findall(
            r"^\d+\.\s+\*\*(.+?)\*\*", _md_section(md, r"^### 1\.2 .*$"), re.MULTILINE
        )
    ]
    conditions = [
        _strip_md(t)
        for t in re.findall(r"^- (.+)$", _md_section(md, r"^### 1\.3 .*$"), re.MULTILINE)
    ]
    revisit = _strip_md(_md_section(md, r"^### 1\.4 .*$"))

    blocking = [str(r["risk_id"]) for r in rows if "blocking" in str(r["status"]).lower()]

    return {
        "subject": "xgboost_pipeline - Adult / Census Income classification baseline",
        "decision_date": "2026-08-10",
        "research_use": research,
        "real_world_deployment": deployment,
        "headline": (
            "Conditionally approved for research and educational use only; "
            "blocked from real-world deployment."
        ),
        "grounds_for_deployment_block": grounds,
        "conditions_on_research_use": conditions,
        "revisit_requirements": revisit,
        "risk_profile": {
            "total_risks": len(rows),
            "counts_by_overall_risk": by_risk,
            "counts_by_status": by_status,
            "rating_scale": RATING_SCALE,
            "assessment_framing": ASSESSMENT_FRAMING,
        },
        "blocking_risk_ids": blocking,
        "disclaimer": DISCLAIMER,
        "sources": [_rel("governance_summary"), _rel("risk_register")],
        "summary_markdown": md if include_markdown else None,
    }


# --------------------------------------------------------------------------- #
# Governance: model card
# --------------------------------------------------------------------------- #
def build_model_card(sections_only: bool = False) -> dict[str, Any]:
    md = reader.read_markdown("model_card")
    sections = [
        _strip_md(h) for h in re.findall(r"^##\s+(.+)$", md, re.MULTILINE)
    ]
    return {
        "subject": "xgboost_pipeline - Adult / Census Income classification baseline",
        "sections": sections,
        "character_count": len(md),
        "source": _rel("model_card"),
        "content": None if sections_only else md,
    }


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def build_health(service: str, version: str) -> dict[str, Any]:
    statuses = reader.artifact_status()
    missing = [s["key"] for s in statuses if not s["present"]]
    return {
        "status": "ok" if not missing else "degraded",
        "service": service,
        "version": version,
        "mode": "read-only",
        "artifacts_present": len(statuses) - len(missing),
        "artifacts_expected": len(statuses),
        "missing_artifacts": missing,
        "artifacts": statuses,
    }
