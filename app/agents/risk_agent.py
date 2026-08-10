"""
risk_agent.py
=============
Reports the blocking risks, the conditions attached to research use, and the
existing deployment recommendation.

Evidence: ``GET /api/governance/decision`` and ``GET /api/governance/risks``
(Phase 4 governance artefacts).

Decision preservation -- the hard constraint
--------------------------------------------
This agent **reads and restates** the committed decision. It does not evaluate,
re-derive, soften or override it. The decision fields (``research_use``,
``real_world_deployment``, ``headline``) are copied verbatim into the finding
evidence and into :class:`~app.agents.schemas.PreservedDecision`.

If the agent were ever to encounter a decision record that did not block
deployment, it would report exactly what the record says -- it has no authority to
substitute its own verdict in either direction. ``tests/test_agents.py`` asserts
that the restated decision equals the API's decision field for field.

Severity policy (fixed, documented, presentation-only)
------------------------------------------------------
* ``CRITICAL`` when the record blocks real-world deployment, and for the presence
  of Critical-rated entries in the register -- these are the items that must not be
  skimmed past.
* ``HIGH`` for the set of blocking risks.
* ``MEDIUM`` for research-use conditions and for any risk not yet assessed.
"""

from __future__ import annotations

from typing import Any

from app.agents.schemas import (
    AgentReport,
    BaseGovernanceAgent,
    PreservedDecision,
    Severity,
)
from app.schemas.models import DecisionResponse, RiskRegisterResponse
from app.services import governance_service as svc

DECISION_ENDPOINT = "GET /api/governance/decision"
RISKS_ENDPOINT = "GET /api/governance/risks"

NOT_YET_ASSESSED = "not yet assessed"


class RiskDecisionAgent(BaseGovernanceAgent):
    agent_name = "risk"
    agent_role = (
        "Reports blocking risks, research-use conditions and the existing "
        "deployment recommendation."
    )
    reads = [DECISION_ENDPOINT, RISKS_ENDPOINT]
    reports = [
        "The committed deployment decision, restated verbatim",
        "Blocking and Critical-rated risks",
        "Conditions attached to research use",
        "Risks not yet assessed",
    ]
    constraints = BaseGovernanceAgent.constraints + [
        "Never re-evaluates, softens or overrides the committed governance decision; "
        "it restates it verbatim.",
    ]

    # -- decision access shared with the orchestrator --------------------- #
    def preserved_decision(self) -> PreservedDecision:
        """Return the committed decision, copied field for field."""
        decision = DecisionResponse(**svc.build_decision()).model_dump()
        return PreservedDecision(
            research_use=str(decision.get("research_use")),
            real_world_deployment=str(decision.get("real_world_deployment")),
            headline=str(decision.get("headline")),
            blocking_risk_ids=[str(r) for r in decision.get("blocking_risk_ids") or []],
            source=DECISION_ENDPOINT,
        )

    def run(self, model_name: str) -> AgentReport:
        # Read the API's own response models (see performance_agent for why).
        decision = DecisionResponse(**svc.build_decision()).model_dump()
        register = RiskRegisterResponse(**svc.build_risks()).model_dump()

        risks: list[dict[str, Any]] = register.get("risks") or []
        counts: dict[str, int] = register.get("counts_by_overall_risk") or {}
        blocking_ids = [str(r) for r in decision.get("blocking_risk_ids") or []]
        conditions = list(decision.get("conditions_on_research_use") or [])
        grounds = list(decision.get("grounds_for_deployment_block") or [])
        findings = []

        # -- RISK-01: the decision, restated verbatim ----------------------- #
        blocked = str(decision.get("real_world_deployment")) == "blocked"
        findings.append(
            self._finding(
                "RISK-01",
                Severity.CRITICAL if blocked else Severity.HIGH,
                (
                    "The committed governance decision for this model is: "
                    f"research use = '{decision.get('research_use')}', real-world "
                    f"deployment = '{decision.get('real_world_deployment')}'. "
                    f"{decision.get('headline')} This agent restates that record and "
                    "has no authority to change it."
                ),
                DECISION_ENDPOINT,
                {
                    "research_use": decision.get("research_use"),
                    "real_world_deployment": decision.get("real_world_deployment"),
                    "headline": decision.get("headline"),
                    "decision_date": decision.get("decision_date"),
                    "subject": decision.get("subject"),
                    "grounds_for_deployment_block": grounds,
                },
                [
                    "The decision is a copy of the committed record, not an agent "
                    "judgement. Agents cannot approve, block or amend anything.",
                    str(decision.get("disclaimer") or ""),
                    "The decision rests on documented measurements and absent "
                    "prerequisites, not on any legal finding.",
                ],
                (
                    "Preserve the decision as recorded: research and education only, "
                    "with real-world deployment blocked. Any change requires a new "
                    "signed-off assessment, not an agent run."
                ),
            )
        )

        # -- RISK-02: Critical-rated entries -------------------------------- #
        critical = [r for r in risks if str(r.get("overall_risk")).lower() == "critical"]
        findings.append(
            self._finding(
                "RISK-02",
                Severity.CRITICAL if critical else Severity.MEDIUM,
                (
                    f"The register holds {register.get('total_in_register')} risks, of "
                    f"which {len(critical)} are rated Critical: "
                    + ", ".join(f"{r.get('risk_id')} ({r.get('category')})" for r in critical)
                    + "."
                    if critical
                    else f"The register holds {register.get('total_in_register')} risks, "
                    "none rated Critical."
                ),
                RISKS_ENDPOINT,
                {
                    "total_in_register": register.get("total_in_register"),
                    "counts_by_overall_risk": counts,
                    "critical_risk_ids": [str(r.get("risk_id")) for r in critical],
                    "critical_categories": [str(r.get("category")) for r in critical],
                },
                [
                    str(register.get("assessment_framing") or ""),
                    "Severity ratings are qualitative expert judgement against "
                    "documented evidence, not calibrated probabilities.",
                    "Impact is rated as if the model were used for a consequential "
                    "decision; there is no deployment, so no realised harm to date.",
                ],
                "Review every Critical entry and its recommended control before any "
                "proposal to change the model's permitted use.",
            )
        )

        # -- RISK-03: blocking risks ---------------------------------------- #
        blocking = [r for r in risks if str(r.get("risk_id")) in blocking_ids]
        findings.append(
            self._finding(
                "RISK-03",
                Severity.HIGH if blocking_ids else Severity.MEDIUM,
                (
                    f"{len(blocking_ids)} risk(s) are marked blocking for deployment: "
                    + ", ".join(blocking_ids)
                    + ". Each must be closed, not merely acknowledged, before "
                    "deployment could be reconsidered."
                ),
                RISKS_ENDPOINT,
                {
                    "blocking_risk_ids": blocking_ids,
                    "blocking_risks": [
                        {
                            "risk_id": r.get("risk_id"),
                            "category": r.get("category"),
                            "overall_risk": r.get("overall_risk"),
                            "residual_risk": r.get("residual_risk"),
                            "status": r.get("status"),
                        }
                        for r in blocking
                    ],
                },
                [
                    "Residual risk remains after the recommended control for several "
                    "entries; a control is not a closure.",
                    "At least one risk is documented as unmitigable within this dataset, "
                    "so it cannot be closed by model changes alone.",
                ],
                "Track each blocking risk to closure with a named owner; treat residual "
                "risk as still open.",
            )
        )

        # -- RISK-04: research-use conditions ------------------------------- #
        findings.append(
            self._finding(
                "RISK-04",
                Severity.MEDIUM,
                (
                    f"Research use is permitted subject to {len(conditions)} explicit "
                    "condition(s). Breach of any condition voids the approval."
                ),
                DECISION_ENDPOINT,
                {
                    "research_use": decision.get("research_use"),
                    "conditions_on_research_use": conditions,
                    "revisit_requirements": decision.get("revisit_requirements"),
                },
                [
                    "The conditions are part of the approval, not advisory notes.",
                    "Approval covers teaching, methodology development and governance "
                    "demonstration only -- never a decision about a person.",
                ],
                "Attach these conditions to any redistribution of the model or its "
                "outputs, and re-run the audits after any change to data, features, "
                "threshold or model.",
            )
        )

        # -- RISK-05: not-yet-assessed risks -------------------------------- #
        unassessed = [r for r in risks if NOT_YET_ASSESSED in str(r.get("status")).lower()]
        if unassessed:
            findings.append(
                self._finding(
                    "RISK-05",
                    Severity.MEDIUM,
                    (
                        f"{len(unassessed)} risk(s) are recorded as not yet assessed: "
                        + ", ".join(
                            f"{r.get('risk_id')} ({r.get('category')})" for r in unassessed
                        )
                        + ". These are open gaps in the assessment, not passes."
                    ),
                    RISKS_ENDPOINT,
                    {
                        "unassessed_risk_ids": [str(r.get("risk_id")) for r in unassessed],
                        "unassessed": [
                            {
                                "risk_id": r.get("risk_id"),
                                "category": r.get("category"),
                                "status": r.get("status"),
                            }
                            for r in unassessed
                        ],
                    },
                    [
                        "An unassessed risk must not be read as a low risk.",
                        "Where subgroups cannot be assessed because cells are too small, "
                        "the correct report is 'unknown', not 'no disparity'.",
                    ],
                    "Close the assessment gap or state explicitly that the subgroup "
                    "cannot be assessed with the available data.",
                )
            )

        return AgentReport(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            model_name=model_name,
            status="ok",
            summary=(
                f"Decision preserved: research use '{decision.get('research_use')}', "
                f"real-world deployment '{decision.get('real_world_deployment')}'. "
                f"{len(critical)} Critical risk(s) and {len(blocking_ids)} blocking "
                f"risk(s) recorded, with {len(conditions)} condition(s) on research use."
            ),
            findings=findings,
            evidence_sources=[DECISION_ENDPOINT, RISKS_ENDPOINT],
            caveats=[str(decision.get("disclaimer") or "")],
        )
