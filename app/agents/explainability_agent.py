"""
explainability_agent.py
=======================
Reports the features a model relies on, proxy-feature concerns, and the limits of
what an explanation can support.

Evidence: ``GET /api/models/{model_name}/explainability`` (Phase 3 audit).

Availability
------------
The Phase 3 audit covers the primary model and one comparison model only. For any
other model this agent returns a **status ``unavailable`` report with an explicit
"no evidence exists" finding**. It never estimates importances, never carries
values over from another model, and never infers that an unaudited model is
therefore fine.

Severity policy (fixed, documented, presentation-only)
------------------------------------------------------
* ``HIGH`` for the proxy finding when any watch-listed proxy outranks both
  protected attributes -- the case where a feature-list reading of fairness would
  mislead a reviewer.
* ``MEDIUM`` for the association-not-causation and correlated-dilution limits.
* ``INFO`` for the feature ranking itself.
"""

from __future__ import annotations

from typing import Any

from app.agents.schemas import BaseGovernanceAgent, AgentReport, Severity
from app.schemas.models import ExplainabilityResponse
from app.services import artifact_reader as reader
from app.services import governance_service as svc

CAUSATION_LIMITATIONS = [
    "Association, not causation: importance and SHAP describe how a fitted function "
    "responds to its inputs on one dataset. Neither is a causal estimand.",
    "A high-ranking feature is not a cause of income, and changing that feature for a "
    "person would not necessarily change their earnings.",
]


class ExplainabilityAgent(BaseGovernanceAgent):
    agent_name = "explainability"
    agent_role = (
        "Reports feature reliance, proxy-feature concerns and explanation limitations."
    )
    reads = ["GET /api/models/{model_name}/explainability"]
    reports = [
        "Most-relied-on original features",
        "Protected-attribute ranks and proxy features that outrank them",
        "Association-versus-causation limits and correlated-feature dilution",
    ]
    constraints = BaseGovernanceAgent.constraints + [
        "Never estimates importances for a model the audit did not cover.",
        "Never presents low protected-attribute importance as evidence of fairness.",
    ]

    def run(self, model_name: str) -> AgentReport:
        endpoint = f"GET /api/models/{model_name}/explainability"
        try:
            # Read the API's own response model (see performance_agent for why).
            data = ExplainabilityResponse(
                **svc.build_explainability(model_name)
            ).model_dump()
        except reader.ModelNotFoundError as exc:
            return self._unavailable(
                "EXPL-NA",
                model_name,
                "explainability",
                endpoint,
                exc.available,
                (
                    "The Phase 3 audit covered the primary model and its comparison "
                    "model only. No importances are estimated for other models."
                ),
            )

        importance: list[dict[str, Any]] = data.get("global_importance") or []
        proxy = data.get("proxy_assessment") or {}
        caveats = list(data.get("caveats") or [])
        findings = []

        # -- EXPL-01: what the model relies on ------------------------------ #
        top = importance[:5]
        findings.append(
            self._finding(
                "EXPL-01",
                Severity.INFO,
                (
                    f"'{model_name}' relies most on: "
                    + ", ".join(
                        f"{f.get('feature')} ({f.get('importance_mean')})" for f in top
                    )
                    + f". Importance is the drop in held-out {data.get('scorer')} when "
                    "the column is shuffled, measured over the original "
                    f"{data.get('n_features')} human-readable features."
                ),
                endpoint,
                {
                    "scorer": data.get("scorer"),
                    "n_features": data.get("n_features"),
                    "top_features": [
                        {
                            "rank": f.get("rank"),
                            "feature": f.get("feature"),
                            "importance_mean": f.get("importance_mean"),
                            "importance_std": f.get("importance_std"),
                            "classification": f.get("classification"),
                        }
                        for f in top
                    ],
                },
                CAUSATION_LIMITATIONS
                + [
                    "Where the standard deviation is comparable to the value, the "
                    "feature is not distinguishable from unimportant.",
                    "Rankings shift with the scorer, the split and the model; there is no "
                    "single true importance order.",
                ],
                "Use this to understand model reliance and operational fragility, not to "
                "explain why any individual earns what they earn.",
            )
        )

        # -- EXPL-02: proxy concern (the governance-critical finding) -------- #
        ranks = proxy.get("protected_attribute_ranks") or {}
        above = proxy.get("proxies_ranked_above_all_protected_attributes") or {}
        findings.append(
            self._finding(
                "EXPL-02",
                Severity.HIGH if above else Severity.MEDIUM,
                (
                    "Protected attributes rank low in global importance ("
                    + ", ".join(
                        f"{name} at rank {rank}/{data.get('n_features')}"
                        for name, rank in ranks.items()
                    )
                    + ")"
                    + (
                        ", while "
                        + str(len(above))
                        + " watch-listed proxy feature(s) outrank both: "
                        + ", ".join(
                            f"{k} (rank {v})"
                            for k, v in sorted(above.items(), key=lambda kv: kv[1])
                        )
                        + ". Low importance for a protected attribute is therefore NOT "
                        "evidence of fairness -- the information is carried by "
                        "correlated features."
                        if above
                        else "."
                    )
                ),
                endpoint,
                {
                    "protected_attribute_ranks": ranks,
                    "proxies_ranked_above_all_protected_attributes": above,
                    "n_features": data.get("n_features"),
                    "finding_text": proxy.get("finding"),
                    "implication_text": proxy.get("implication"),
                },
                CAUSATION_LIMITATIONS
                + [
                    "Permutation importance measures each column's INCREMENTAL "
                    "contribution, so a redundantly-encoded attribute correctly scores "
                    "low while its information remains fully in use.",
                    "Removing the protected attributes would not be a mitigation: it "
                    "would leave the proxies and destroy the ability to measure "
                    "disparity.",
                    "This is a structural observation about feature correlation, not a "
                    "claim that these features cause the disparity.",
                ],
                (
                    "Test proxy strength explicitly and gate fairness on disaggregated "
                    "outcome testing, never on a feature list or importance ranking."
                ),
            )
        )

        # -- EXPL-03: correlated-feature dilution --------------------------- #
        lowest = importance[-1] if importance else {}
        findings.append(
            self._finding(
                "EXPL-03",
                Severity.MEDIUM,
                (
                    "Importance scores are lower bounds, because correlated features "
                    "dilute each other. The lowest-ranked feature here is "
                    f"'{lowest.get('feature')}' at {lowest.get('importance_mean')}, "
                    "which reflects redundancy with a correlated twin column rather "
                    "than irrelevance."
                ),
                endpoint,
                {
                    "lowest_ranked_feature": lowest.get("feature"),
                    "lowest_importance_mean": lowest.get("importance_mean"),
                    "lowest_rank": lowest.get("rank"),
                },
                CAUSATION_LIMITATIONS
                + [
                    "A near-zero or negative score must never be read as 'the model does "
                    "not use this information'.",
                    "Permutation also scores the model partly off-manifold: shuffling "
                    "creates records that could not occur.",
                    "Grouped permutation over correlated features would give a truer "
                    "picture and was not run.",
                ],
                "Do not drop features on the basis of low individual importance; test "
                "correlated groups together.",
            )
        )

        # -- EXPL-04: local-explanation limits ------------------------------ #
        local = data.get("local_explanations") or []
        findings.append(
            self._finding(
                "EXPL-04",
                Severity.MEDIUM,
                (
                    f"{len(local)} individual case explanation(s) are published for "
                    f"'{model_name}'. They show how the model reasoned in specific "
                    "instances and support no population-level or subgroup conclusion."
                    if local
                    else (
                        f"No local case explanations are published for '{model_name}'; "
                        "it is a comparison model, so only global importance is available."
                    )
                ),
                endpoint,
                {
                    "n_local_cases": len(local),
                    "case_ids": [c.get("case_id") for c in local],
                    "is_primary": data.get("is_primary"),
                },
                CAUSATION_LIMITATIONS
                + [
                    "Five illustrative cases cannot support any claim about the "
                    "population or about a subgroup.",
                    "SHAP contributions are log-odds, not percentage points.",
                    "TreeSHAP is exact for this model, but Shapley attribution is one "
                    "choice among several that can rank factors differently.",
                    "Case identifiers are synthetic; dataset row indices are withheld.",
                ],
                "Use local cases for illustration and debugging only; never generalise "
                "from them to a group.",
            )
        )

        return AgentReport(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            model_name=model_name,
            status="ok",
            summary=(
                f"'{model_name}' relies most on "
                f"'{top[0].get('feature') if top else 'n/a'}'. Protected attributes rank "
                + ", ".join(f"{k}={v}" for k, v in ranks.items())
                + f" of {data.get('n_features')}"
                + (
                    f", below {len(above)} proxy feature(s) -- low protected-attribute "
                    "importance is not evidence of fairness."
                    if above
                    else "."
                )
            ),
            findings=findings,
            evidence_sources=[endpoint],
            caveats=caveats,
        )
