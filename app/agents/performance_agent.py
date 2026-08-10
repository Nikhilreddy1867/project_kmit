"""
performance_agent.py
====================
Reports held-out performance, error counts and threshold limitations for one model.

Evidence: ``GET /api/models/{model_name}/performance`` (Phase 1 audit).

Severity policy (fixed, documented, presentation-only)
------------------------------------------------------
* Headline metrics are reported at ``INFO`` -- they are context, not a risk.
* Error asymmetry is ``HIGH`` when false negatives exceed false positives,
  because the harm then concentrates on people the model *should* have selected;
  otherwise ``MEDIUM``. This is a **comparison of two counts the audit already
  produced**, not a new metric.
* The untuned 0.5 threshold is ``MEDIUM``: it silently makes a policy choice.
* Single-split evaluation is ``LOW``: a real but bounded limitation.
"""

from __future__ import annotations

from app.agents.schemas import BaseGovernanceAgent, AgentReport, Severity
from app.schemas.models import PerformanceResponse
from app.services import artifact_reader as reader
from app.services import governance_service as svc


class PerformanceAgent(BaseGovernanceAgent):
    agent_name = "performance"
    agent_role = "Reports held-out performance, error counts and threshold limitations."
    reads = ["GET /api/models/{model_name}/performance"]
    reports = [
        "Headline metrics (accuracy, precision, recall, F1, ROC-AUC)",
        "Confusion-matrix counts and error asymmetry",
        "Decision-threshold and single-split limitations",
    ]

    def run(self, model_name: str) -> AgentReport:
        endpoint = f"GET /api/models/{model_name}/performance"
        try:
            # Build the SAME response model the API route returns, then read it.
            # Fields supplied by schema defaults (decision_threshold, positive_class)
            # only exist once the model is constructed -- reading the raw service
            # dict would quote None where the endpoint serves 0.5. Going through the
            # response model makes "agent evidence == API payload" true by
            # construction rather than by coincidence.
            perf = PerformanceResponse(**svc.build_performance(model_name)).model_dump()
        except reader.ModelNotFoundError as exc:
            return self._unavailable(
                "PERF-NA",
                model_name,
                "performance",
                endpoint,
                exc.available,
                "The Phase 1 audit did not evaluate this model.",
            )

        metrics = perf.get("metrics") or {}
        cm = perf.get("confusion_matrix") or {}
        err = perf.get("error_analysis") or {}
        caveats = list(perf.get("caveats") or [])

        fn = err.get("false_negatives")
        fp = err.get("false_positives")
        findings = []

        # -- PERF-01: headline metrics -------------------------------------- #
        findings.append(
            self._finding(
                "PERF-01",
                Severity.INFO,
                (
                    f"Held-out performance for '{model_name}': "
                    f"ROC-AUC {metrics.get('roc_auc')}, F1 {metrics.get('f1')}, "
                    f"accuracy {metrics.get('accuracy')}, precision "
                    f"{metrics.get('precision')}, recall {metrics.get('recall')} "
                    f"on {perf.get('n_test')} test rows."
                ),
                endpoint,
                {
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "n_test": perf.get("n_test"),
                },
                [
                    "Accuracy must not be read alone on this imbalanced target; the "
                    "majority-class floor is high. F1 and ROC-AUC are the informative "
                    "headline metrics.",
                    "Only ROC-AUC is independent of the decision threshold.",
                    "Aggregate metrics conceal group disparities -- see the fairness agent.",
                ],
                "Read F1 and ROC-AUC as the headline; treat accuracy as context only.",
            )
        )

        # -- PERF-02: error counts and asymmetry ---------------------------- #
        asymmetric = fn is not None and fp is not None and fn > fp
        findings.append(
            self._finding(
                "PERF-02",
                Severity.HIGH if asymmetric else Severity.MEDIUM,
                (
                    f"The model produced {fn} false negatives and {fp} false "
                    "positives on the held-out set."
                    + (
                        " False negatives exceed false positives, so the dominant "
                        "error is failing to identify people who genuinely are high "
                        "earners."
                        if asymmetric
                        else " False positives are the dominant error."
                    )
                ),
                endpoint,
                {
                    "false_negatives": fn,
                    "false_positives": fp,
                    "true_positives": cm.get("true_positives"),
                    "true_negatives": cm.get("true_negatives"),
                },
                [
                    "Counts are quoted from the audit; no rate or ratio is derived here.",
                    "Which error type is worse depends on the deployment context, which "
                    "does not exist for this model.",
                    "The error mix is a direct consequence of the 0.5 threshold "
                    "(see PERF-03).",
                ],
                (
                    "Define an explicit cost for each error type before any threshold "
                    "or deployment decision; check the fairness agent for how these "
                    "errors distribute across groups."
                ),
            )
        )

        # -- PERF-03: threshold limitation ---------------------------------- #
        findings.append(
            self._finding(
                "PERF-03",
                Severity.MEDIUM,
                (
                    f"All rates above are measured at the default decision threshold "
                    f"{perf.get('decision_threshold')}, which was inherited and never "
                    "tuned. An unexamined default is silently making the policy "
                    "trade-off between error types."
                ),
                endpoint,
                {
                    "decision_threshold": perf.get("decision_threshold"),
                    "positive_class": perf.get("positive_class"),
                    "threshold_free_metric": "roc_auc",
                    "roc_auc": metrics.get("roc_auc"),
                },
                [
                    "Precision, recall, F1 and accuracy would all change under a "
                    "different threshold; ROC-AUC would not.",
                    "This agent does not recommend a threshold value -- that is a "
                    "governance decision requiring a cost model.",
                ],
                (
                    "Treat the threshold as a documented governance decision: sweep it "
                    "and publish the trade-off curve rather than accepting 0.5."
                ),
            )
        )

        # -- PERF-04: evaluation-design limitation -------------------------- #
        findings.append(
            self._finding(
                "PERF-04",
                Severity.LOW,
                (
                    "Performance rests on a single deterministic train/test split with "
                    "no cross-validation and no confidence intervals, so these are "
                    "point estimates."
                ),
                endpoint,
                {"n_test": perf.get("n_test"), "source": perf.get("source")},
                [
                    "Point estimates without intervals can look more precise than they "
                    "are; small differences between models may not be meaningful.",
                    "The split is deterministic, so results are reproducible -- "
                    "reproducible is not the same as robust.",
                ],
                "Add repeated splits or bootstrap intervals before comparing models on "
                "small metric differences.",
            )
        )

        return AgentReport(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            model_name=model_name,
            status="ok",
            summary=(
                f"'{model_name}' scores ROC-AUC {metrics.get('roc_auc')} and F1 "
                f"{metrics.get('f1')} on {perf.get('n_test')} held-out rows, with "
                f"{fn} false negatives against {fp} false positives at threshold "
                f"{perf.get('decision_threshold')}."
            ),
            findings=findings,
            evidence_sources=[endpoint],
            caveats=caveats,
        )
