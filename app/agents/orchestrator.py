"""
orchestrator.py
===============
Runs the four deterministic governance agents for one model and assembles a single
structured, evidence-based review.

What the orchestrator does
--------------------------
* Runs every agent in a fixed order (so output is stable and diffable).
* Counts findings per severity label -- counting findings, not recomputing any
  model metric.
* Copies the committed governance decision verbatim into the review.

What it deliberately does NOT do
--------------------------------
* It forms **no verdict of its own.** ``overall_recommendation`` is the existing
  decision headline, copied. There is no code path in which the agent layer can
  approve, block, soften or escalate anything -- the recommendation string is never
  constructed from agent findings.
* It does not weight, average or score agents. Severity counts are reported so a
  reviewer can triage; they are not combined into an index.
"""

from __future__ import annotations

from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.fairness_agent import FairnessAgent
from app.agents.performance_agent import PerformanceAgent
from app.agents.risk_agent import RiskDecisionAgent
from app.agents.schemas import (
    AgentDescriptor,
    AgentListResponse,
    AgentReport,
    BaseGovernanceAgent,
    GovernanceReview,
    SEVERITY_ORDER,
    SEVERITY_RANK,
    Severity,
)
from app.services import artifact_reader as reader

# Fixed execution order -> deterministic, diffable output.
AGENT_CLASSES: tuple[type[BaseGovernanceAgent], ...] = (
    PerformanceAgent,
    FairnessAgent,
    ExplainabilityAgent,
    RiskDecisionAgent,
)

AGENTS: dict[str, BaseGovernanceAgent] = {cls.agent_name: cls() for cls in AGENT_CLASSES}
AGENT_NAMES: list[str] = [cls.agent_name for cls in AGENT_CLASSES]


class AgentNotFoundError(Exception):
    """Raised for an unknown agent name; mapped to HTTP 404 by the API."""

    def __init__(self, agent_name: str, available: list[str]):
        self.agent_name = agent_name
        self.available = available
        super().__init__(
            f"Unknown agent '{agent_name}'. Available: {', '.join(available)}."
        )


# --------------------------------------------------------------------------- #
# Listing / single agent
# --------------------------------------------------------------------------- #
def list_agents() -> AgentListResponse:
    descriptors: list[AgentDescriptor] = [AGENTS[name].descriptor() for name in AGENT_NAMES]
    return AgentListResponse(count=len(descriptors), agents=descriptors)


def run_agent(agent_name: str, model_name: str) -> AgentReport:
    agent = AGENTS.get(agent_name)
    if agent is None:
        raise AgentNotFoundError(agent_name, AGENT_NAMES)
    return agent.run(model_name)


def _assert_known_model(model_name: str) -> None:
    """
    Fail fast on a model that has no Phase 1 record at all.

    Without this, every agent would independently return 'unavailable' and the
    review would read as though a real model had simply not been audited. A name
    that does not exist deserves a 404, not four empty reports.
    """
    available = reader.evaluated_models()
    if model_name not in available:
        raise reader.ModelNotFoundError(model_name, available, "Phase 1 performance")


# --------------------------------------------------------------------------- #
# Orchestrated review
# --------------------------------------------------------------------------- #
def run_review(model_name: str) -> GovernanceReview:
    _assert_known_model(model_name)

    reports: list[AgentReport] = [AGENTS[name].run(model_name) for name in AGENT_NAMES]

    findings = [f for report in reports for f in report.findings]
    counts = {level: 0 for level in SEVERITY_ORDER}
    for finding in findings:
        counts[finding.severity.value] += 1

    highest = max(
        (f.severity for f in findings),
        key=lambda s: SEVERITY_RANK[s.value],
        default=Severity.INFO,
    )

    # The decision is COPIED, never derived. `overall_recommendation` is the
    # committed headline verbatim -- the agents contribute no verdict.
    preserved = RiskDecisionAgent().preserved_decision()

    unavailable = [
        f"{report.agent_name}: no {report.agent_name} evidence for '{model_name}'"
        for report in reports
        if report.status == "unavailable"
    ]

    notes: list[str] = [
        f"{len(reports)} deterministic agents ran in fixed order: "
        + ", ".join(r.agent_name for r in reports)
        + ".",
        f"{len(findings)} findings in total; highest severity label: {highest.value}.",
        "The recommendation below is the committed governance decision restated "
        "verbatim. The agents did not produce it and cannot change it.",
    ]
    if unavailable:
        notes.append(
            f"{len(unavailable)} agent(s) had no evidence for this model and reported "
            "that explicitly rather than estimating anything."
        )
    else:
        notes.append("All four evidence sources were available for this model.")
    notes.append(
        "Severity labels are a triage convention. They are not measurements and are "
        "not combined into a score."
    )

    return GovernanceReview(
        model_name=model_name,
        agents_run=[r.agent_name for r in reports],
        agents=reports,
        findings_total=len(findings),
        severity_counts=counts,
        highest_severity=highest,
        preserved_decision=preserved,
        overall_recommendation=preserved.headline,
        consensus_notes=notes,
        unavailable_evidence=unavailable,
    )
