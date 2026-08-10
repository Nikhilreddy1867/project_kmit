"""
app.agents
==========
Phase 7 -- a read-only, **deterministic** multi-agent governance layer.

Four rule-based agents read the existing audit evidence through the same service
layer the Phase 5 API routes use, and report structured findings:

* :class:`~app.agents.performance_agent.PerformanceAgent`
* :class:`~app.agents.fairness_agent.FairnessAgent`
* :class:`~app.agents.explainability_agent.ExplainabilityAgent`
* :class:`~app.agents.risk_agent.RiskDecisionAgent`

:mod:`app.agents.orchestrator` runs all four for a model and assembles one review.

These are reporting agents, not autonomous decision-makers: they never train,
never write, never recalculate a metric, and never alter the governance decision.
See :mod:`app.agents.schemas` for the full constraint list.
"""

from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.fairness_agent import FairnessAgent
from app.agents.orchestrator import (
    AGENT_NAMES,
    AGENTS,
    AgentNotFoundError,
    list_agents,
    run_agent,
    run_review,
)
from app.agents.performance_agent import PerformanceAgent
from app.agents.risk_agent import RiskDecisionAgent

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
    "AgentNotFoundError",
    "ExplainabilityAgent",
    "FairnessAgent",
    "PerformanceAgent",
    "RiskDecisionAgent",
    "list_agents",
    "run_agent",
    "run_review",
]
