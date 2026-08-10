"""
fairness_agent.py
=================
Reports measured group disparities by ``sex`` and ``race``, the four-fifths
screening context, small-group uncertainty and the metric-conflict constraint.

Evidence: ``GET /api/models/{model_name}/fairness`` (Phase 2 audit).

Language constraint -- enforced, not aspirational
-------------------------------------------------
This agent describes **measured output differences**. It must never characterise a
disparity as a legal violation, unlawful discrimination, or a causal effect. Every
finding therefore carries :data:`MANDATORY_LIMITATIONS`, and
``tests/test_agents.py`` asserts that forbidden phrasings never appear in this
agent's output.

Severity policy (fixed, documented, presentation-only)
------------------------------------------------------
* ``HIGH`` when the audit's own ``fails_four_fifths_rule`` flag is true for any
  group -- the conventional trigger to look harder.
* ``MEDIUM`` otherwise for a measured gap, and for small-group uncertainty and the
  metric-conflict constraint.

The agent reads the audit's **own boolean flags** (``fails_four_fifths_rule``,
``small_group_flag``) rather than re-deriving the comparisons, so it cannot
disagree with the audit even in principle.
"""

from __future__ import annotations

from typing import Any

from app.agents.schemas import BaseGovernanceAgent, AgentReport, Severity
from app.schemas.models import FairnessResponse
from app.services import artifact_reader as reader
from app.services import governance_service as svc

MANDATORY_LIMITATIONS = [
    "This is a measured difference in model output, NOT a finding of unlawful "
    "discrimination. No legal determination is made or implied.",
    "Association only: no causal effect of a protected attribute on the prediction "
    "is identified or claimed.",
    "The four-fifths (0.80) rule is a screening convention from US employment-law "
    "practice used to decide where to look further. It is not a statistical test and "
    "not a verdict.",
    "Group base rates differ in the 1994 labels themselves, so a disparity is jointly "
    "attributable to the model and to historical label bias; no method used here can "
    "separate the two.",
]


class FairnessAgent(BaseGovernanceAgent):
    agent_name = "fairness"
    agent_role = (
        "Reports measured group disparities, screening context, small-group "
        "uncertainty and metric conflicts."
    )
    reads = ["GET /api/models/{model_name}/fairness"]
    reports = [
        "Selection-rate, TPR and FPR disparities by sex and race",
        "Disparate-impact ratios with four-fifths screening context",
        "Small-group uncertainty",
        "Mutual incompatibility of fairness metrics",
    ]
    constraints = BaseGovernanceAgent.constraints + [
        "Never describes a disparity as a legal violation or as causal discrimination.",
    ]

    def run(self, model_name: str) -> AgentReport:
        endpoint = f"GET /api/models/{model_name}/fairness"
        try:
            # Read the API's own response model (see performance_agent for why).
            data = FairnessResponse(**svc.build_fairness(model_name)).model_dump()
        except reader.ModelNotFoundError as exc:
            return self._unavailable(
                "FAIR-NA",
                model_name,
                "fairness",
                endpoint,
                exc.available,
                "The Phase 2 fairness audit did not cover this model.",
            )

        groups: list[dict[str, Any]] = data.get("groups") or []
        summaries: list[dict[str, Any]] = data.get("summary") or []
        interpretation = data.get("interpretation") or {}
        findings = []

        # -- FAIR-01..n: one finding per sensitive attribute ----------------- #
        for index, summary in enumerate(summaries, start=1):
            attribute = str(summary.get("attribute"))
            attr_groups = [g for g in groups if str(g.get("attribute")) == attribute]

            # The audit's own flag decides this -- the agent does not re-derive it.
            failing = [
                str(g.get("group")) for g in attr_groups if g.get("fails_four_fifths_rule")
            ]
            severity = Severity.HIGH if failing else Severity.MEDIUM

            findings.append(
                self._finding(
                    f"FAIR-{index:02d}",
                    severity,
                    (
                        f"By {attribute}, relative to the reference group "
                        f"'{summary.get('reference_group')}', the lowest measured "
                        f"disparate-impact ratio is "
                        f"{summary.get('disparate_impact_ratio_min')} for "
                        f"'{summary.get('disparate_impact_ratio_worst_group')}'. "
                        f"{summary.get('groups_failing_four_fifths')} group(s) fall "
                        "below the 0.80 screening threshold"
                        + (f": {', '.join(failing)}." if failing else ".")
                        + " The largest equal-opportunity (TPR) difference is "
                        f"{summary.get('equal_opportunity_difference_max_abs')} for "
                        f"'{summary.get('equal_opportunity_worst_group')}'."
                    ),
                    endpoint,
                    {
                        "attribute": attribute,
                        "reference_group": summary.get("reference_group"),
                        "disparate_impact_ratio_min": summary.get(
                            "disparate_impact_ratio_min"
                        ),
                        "disparate_impact_ratio_worst_group": summary.get(
                            "disparate_impact_ratio_worst_group"
                        ),
                        "groups_failing_four_fifths": summary.get(
                            "groups_failing_four_fifths"
                        ),
                        "groups_below_screening_threshold": failing,
                        "equal_opportunity_difference_max_abs": summary.get(
                            "equal_opportunity_difference_max_abs"
                        ),
                        "equal_opportunity_worst_group": summary.get(
                            "equal_opportunity_worst_group"
                        ),
                        "equalized_odds_difference": summary.get(
                            "equalized_odds_difference"
                        ),
                    },
                    MANDATORY_LIMITATIONS
                    + [
                        "The reference group is the largest by sample count, which does "
                        "not imply its treatment is correct or desirable.",
                    ],
                    (
                        "Treat this as a trigger for further scrutiny, not a conclusion. "
                        "Select and document a single fairness criterion before "
                        "attempting any mitigation."
                    ),
                )
            )

        # -- FAIR-SMALL: small-group uncertainty ---------------------------- #
        small = [g for g in groups if g.get("small_group_flag")]
        if small:
            findings.append(
                self._finding(
                    "FAIR-SMALL",
                    Severity.MEDIUM,
                    (
                        "Some groups are too small for their metrics to be relied on: "
                        + "; ".join(
                            f"{g.get('group')} (n={g.get('n_samples')}, "
                            f"{g.get('n_actual_positive')} actual positives, "
                            f"TPR 95% CI half-width {g.get('tpr_ci95')})"
                            for g in small
                        )
                        + ". Apparent differences for these groups may be sampling noise."
                    ),
                    endpoint,
                    {
                        "small_groups": [
                            {
                                "attribute": g.get("attribute"),
                                "group": g.get("group"),
                                "n_samples": g.get("n_samples"),
                                "n_actual_positive": g.get("n_actual_positive"),
                                "selection_rate_ci95": g.get("selection_rate_ci95"),
                                "tpr_ci95": g.get("tpr_ci95"),
                            }
                            for g in small
                        ]
                    },
                    MANDATORY_LIMITATIONS
                    + [
                        "Do NOT rank these groups against each other on point estimates.",
                        "TPR for these groups rests on a handful of actual positives, so "
                        "its interval is far wider than the selection-rate interval.",
                        "Wide intervals make a real disparity harder to detect as well as "
                        "harder to confirm -- this is not reassurance.",
                    ],
                    (
                        "Report intervals rather than point estimates for these groups, "
                        "and treat their disparity status as unresolved pending more data."
                    ),
                )
            )

        # -- FAIR-CONFLICT: metric incompatibility -------------------------- #
        findings.append(
            self._finding(
                "FAIR-CONFLICT",
                Severity.MEDIUM,
                (
                    "The fairness metrics reported here cannot all be satisfied at "
                    "once. Because group base rates differ in the data, equalising "
                    "selection rates, equalising error rates and calibration are "
                    "mutually incompatible, so any mitigation will improve one measure "
                    "at the expense of another."
                ),
                endpoint,
                {
                    "does_not_establish": interpretation.get("does_not_establish") or [],
                    "establishes": interpretation.get("establishes") or [],
                },
                MANDATORY_LIMITATIONS
                + [
                    "This is a mathematical constraint, not a defect of this particular "
                    "model, and it cannot be engineered away.",
                    "Visible in this audit: the groups selected least often also receive "
                    "the fewest false positives.",
                ],
                (
                    "Require an explicit, signed-off choice of primary fairness criterion "
                    "before any mitigation, and report the other measures alongside it so "
                    "the trade-off stays visible."
                ),
            )
        )

        worst = min(
            (s for s in summaries if s.get("disparate_impact_ratio_min") is not None),
            key=lambda s: s["disparate_impact_ratio_min"],
            default=None,
        )
        summary_text = (
            f"Measured disparities are present for '{model_name}' across "
            f"{len(summaries)} sensitive attribute(s)."
        )
        if worst:
            summary_text += (
                f" The lowest disparate-impact ratio is "
                f"{worst.get('disparate_impact_ratio_min')} "
                f"({worst.get('disparate_impact_ratio_worst_group')}, by "
                f"{worst.get('attribute')}). These are output differences, not findings "
                "of discrimination."
            )

        return AgentReport(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            model_name=model_name,
            status="ok",
            summary=summary_text,
            findings=findings,
            evidence_sources=[endpoint],
            caveats=list(interpretation.get("does_not_establish") or []),
        )
