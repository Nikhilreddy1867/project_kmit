"""
app/agents/schemas.py
=====================
Pydantic schemas and the shared base class for the Phase 7 governance agents.

What these agents are
---------------------
**Deterministic, rule-based reporting agents.** They are NOT autonomous
decision-makers, NOT language models, and NOT capable of judgement. Each agent:

* reads existing evidence through the same service layer the Phase 5 API routes
  use, so the values it quotes are byte-identical to what those endpoints serve;
* classifies that evidence against **fixed, documented thresholds** declared as
  module constants in each agent;
* emits findings that cite the API endpoint the evidence came from.

Hard constraints, enforced by construction and covered by tests:

1. **No training, no writing.** Agents perform no I/O beyond the read-only
   artefact reader.
2. **No recalculation.** An agent never derives a new metric. Where a judgement
   depends on a threshold comparison, it prefers the **audit's own boolean**
   (``fails_four_fifths_rule``, ``small_group_flag``) over re-deriving it.
3. **No overriding the governance decision.** The Risk agent copies the existing
   decision verbatim; the orchestrator restates it and never forms its own verdict.
4. **No legal or causal claims.** Fairness findings carry the screening-not-legal
   and association-not-causation limitations on every item.
5. **Deterministic output.** No randomness, and deliberately **no timestamps** --
   the same artefacts always produce byte-identical output, so a review can be
   diffed across runs and reproduced in a test.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_CFG = ConfigDict(protected_namespaces=())

AGENT_TYPE = "deterministic-rule-based"

AGENT_DISCLAIMER = (
    "These are deterministic, rule-based reporting agents, not autonomous "
    "decision-makers. They summarise and classify evidence that already exists in "
    "the committed audit artefacts; they do not train models, recalculate metrics, "
    "exercise judgement, or make or alter any governance decision. Severity labels "
    "are a presentation convention for triage, not measurements."
)

DETERMINISM_NOTE = (
    "Output is deterministic: fixed thresholds, no randomness, and no timestamps. "
    "The same artefacts always produce identical output."
)


class Severity(str, Enum):
    """Triage labels. A presentation convention, never a measured quantity."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_RANK: dict[str, int] = {
    Severity.INFO.value: 0,
    Severity.LOW.value: 1,
    Severity.MEDIUM.value: 2,
    Severity.HIGH.value: 3,
    Severity.CRITICAL.value: 4,
}

SEVERITY_ORDER: list[str] = [s.value for s in Severity]


# --------------------------------------------------------------------------- #
# Findings and reports
# --------------------------------------------------------------------------- #
class AgentFinding(BaseModel):
    """One evidence-based observation from one agent."""

    model_config = _CFG

    agent_name: str = Field(description="Which agent produced this finding.")
    finding_id: str = Field(
        description="Stable identifier (e.g. PERF-01), constant across runs."
    )
    severity: Severity = Field(
        description="Triage label from the agent's documented thresholds. "
        "Not a measurement."
    )
    finding: str = Field(description="The observation, in plain language.")
    evidence_source: str = Field(
        description="The API endpoint this evidence is served by."
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Values quoted verbatim from the source artefact. Never "
        "recomputed, rounded or transformed by the agent.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Caveats that constrain how this finding may be read.",
    )
    recommended_action: str = Field(
        description="A concrete next step. Advisory only -- agents cannot act."
    )


class AgentReport(BaseModel):
    """The full output of one agent for one model."""

    model_config = _CFG

    agent_name: str
    agent_role: str
    agent_type: Literal["deterministic-rule-based"] = AGENT_TYPE
    model_name: str
    status: Literal["ok", "unavailable", "error"] = Field(
        description="'unavailable' means the underlying audit does not cover this "
        "model. The agent reports that rather than estimating anything."
    )
    summary: str
    findings: list[AgentFinding]
    evidence_sources: list[str] = Field(
        description="Every API endpoint this report drew on."
    )
    caveats: list[str] = Field(
        default_factory=list, description="Caveats supplied by the source API."
    )


class AgentDescriptor(BaseModel):
    """Self-description of an agent, for GET /api/agents."""

    model_config = _CFG

    agent_name: str
    agent_role: str
    agent_type: Literal["deterministic-rule-based"] = AGENT_TYPE
    reads: list[str] = Field(description="API endpoints this agent draws evidence from.")
    reports: list[str] = Field(description="What this agent reports on.")
    constraints: list[str] = Field(description="What this agent must never do.")


class AgentListResponse(BaseModel):
    model_config = _CFG

    count: int
    agent_type: str = AGENT_TYPE
    agents: list[AgentDescriptor]
    disclaimer: str = AGENT_DISCLAIMER
    determinism: str = DETERMINISM_NOTE


class PreservedDecision(BaseModel):
    """
    The existing governance decision, copied verbatim.

    The orchestrator never forms its own verdict. These fields are copied from
    ``GET /api/governance/decision`` so the agent layer cannot drift from, soften
    or override the committed decision record.
    """

    model_config = _CFG

    research_use: str
    real_world_deployment: str
    headline: str
    blocking_risk_ids: list[str]
    source: str
    note: str = (
        "Copied verbatim from the committed decision record. The agent layer "
        "cannot alter, soften or override it."
    )


class GovernanceReview(BaseModel):
    """The orchestrated multi-agent review for one model."""

    model_config = _CFG

    model_name: str
    review_type: Literal["deterministic-multi-agent"] = "deterministic-multi-agent"
    agent_type: Literal["deterministic-rule-based"] = AGENT_TYPE
    agents_run: list[str]
    agents: list[AgentReport]
    findings_total: int
    severity_counts: dict[str, int] = Field(
        description="Count of findings per severity label. Counting findings, not "
        "recomputing any model metric."
    )
    highest_severity: Severity
    preserved_decision: PreservedDecision
    overall_recommendation: str = Field(
        description="The existing decision headline, restated verbatim. The agents "
        "do not generate a recommendation of their own."
    )
    consensus_notes: list[str]
    unavailable_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence the agents could not obtain for this model.",
    )
    disclaimer: str = AGENT_DISCLAIMER
    determinism: str = DETERMINISM_NOTE


# --------------------------------------------------------------------------- #
# Base agent
# --------------------------------------------------------------------------- #
class BaseGovernanceAgent:
    """
    Shared scaffolding for the four agents.

    Subclasses declare their identity and implement :meth:`run`. This base class
    provides no data access of its own -- each agent calls the read-only service
    layer directly, which keeps its evidence trail explicit.
    """

    agent_name: str = "base"
    agent_role: str = ""
    reads: list[str] = []
    reports: list[str] = []
    constraints: list[str] = [
        "Never trains or re-scores a model.",
        "Never writes any file.",
        "Never recalculates a metric; values are quoted verbatim.",
        "Never makes a legal determination or a causal claim.",
        "Never alters the governance decision.",
    ]

    def descriptor(self) -> AgentDescriptor:
        return AgentDescriptor(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            reads=list(self.reads),
            reports=list(self.reports),
            constraints=list(self.constraints),
        )

    def run(self, model_name: str) -> AgentReport:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------- #
    def _finding(
        self,
        finding_id: str,
        severity: Severity,
        finding: str,
        evidence_source: str,
        evidence: dict[str, Any],
        limitations: list[str],
        recommended_action: str,
    ) -> AgentFinding:
        return AgentFinding(
            agent_name=self.agent_name,
            finding_id=finding_id,
            severity=severity,
            finding=finding,
            evidence_source=evidence_source,
            evidence=evidence,
            limitations=limitations,
            recommended_action=recommended_action,
        )

    def _unavailable(
        self,
        finding_id: str,
        model_name: str,
        scope: str,
        evidence_source: str,
        available: list[str],
        reason: str,
    ) -> AgentReport:
        """
        Build the 'not available' report for a model the audit does not cover.

        This exists so an agent never has to guess. Reporting absence is a valid
        governance output; inventing a number is not.
        """
        finding = self._finding(
            finding_id=finding_id,
            severity=Severity.INFO,
            finding=(
                f"No {scope} evidence exists for '{model_name}', so this agent has "
                f"nothing to report. {reason} Available: "
                f"{', '.join(available) if available else 'none'}."
            ),
            evidence_source=evidence_source,
            evidence={"model_name": model_name, "available_models": available},
            limitations=[
                "Absence of evidence is not evidence of absence: this says nothing "
                "about how the model would score if it were audited.",
                "No values are estimated, interpolated or carried over from another "
                "model.",
            ],
            recommended_action=(
                f"Run the {scope} audit for this model if a report is required, or "
                "restrict conclusions to the models that were audited."
            ),
        )
        return AgentReport(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            model_name=model_name,
            status="unavailable",
            summary=f"{scope.capitalize()} evidence is not available for '{model_name}'.",
            findings=[finding],
            evidence_sources=[evidence_source],
            caveats=[],
        )
