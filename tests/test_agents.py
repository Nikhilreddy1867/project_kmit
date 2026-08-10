"""
tests/test_agents.py
====================
Tests for the Phase 7 deterministic governance agents.

The central contract: **where an agent quotes a value, it must be exactly the
value the corresponding API endpoint serves** -- no rounding, no re-derivation, no
drift. Those assertions use ``==`` on the JSON payloads, not approximate
comparison, so a single-ULP change would fail.

Also covered:
* every finding carries all six required fields;
* the fairness agent never asserts a legal violation or causal discrimination;
* the explainability agent reports 'unavailable' instead of inventing values;
* the committed governance decision is preserved field for field;
* output is deterministic across runs;
* running the agents modifies no artefact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
AGENT_NAMES = ["performance", "fairness", "explainability", "risk"]
PRIMARY = "xgboost"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _get(client: TestClient, path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}"
    return response.json()


# --------------------------------------------------------------------------- #
# Registry and routing
# --------------------------------------------------------------------------- #
def test_agents_listing(client: TestClient) -> None:
    body = _get(client, "/api/agents")
    assert body["count"] == 4
    assert [a["agent_name"] for a in body["agents"]] == AGENT_NAMES
    assert body["agent_type"] == "deterministic-rule-based"
    # Must be labelled as non-autonomous.
    assert "not autonomous decision-makers" in body["disclaimer"]
    for agent in body["agents"]:
        assert agent["agent_type"] == "deterministic-rule-based"
        assert agent["reads"] and agent["reports"] and agent["constraints"]
        joined = " ".join(agent["constraints"]).lower()
        assert "never trains" in joined
        assert "never recalculates" in joined


def test_review_route_is_not_captured_as_agent_name(client: TestClient) -> None:
    """`/api/agents/review` must resolve to the review, not to an agent lookup."""
    body = _get(client, f"/api/agents/review?model_name={PRIMARY}")
    assert body["review_type"] == "deterministic-multi-agent"
    assert "agents_run" in body and "agent_name" not in body


def test_unknown_agent_returns_404_with_available(client: TestClient) -> None:
    response = client.get(f"/api/agents/does_not_exist?model_name={PRIMARY}")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert set(body["available"]) == set(AGENT_NAMES)


def test_unknown_model_returns_404(client: TestClient) -> None:
    response = client.get("/api/agents/review?model_name=not_a_model")
    assert response.status_code == 404
    assert PRIMARY in response.json()["available"]


@pytest.mark.parametrize("agent", AGENT_NAMES)
def test_every_finding_has_all_required_fields(client: TestClient, agent: str) -> None:
    report = _get(client, f"/api/agents/{agent}?model_name={PRIMARY}")
    assert report["agent_name"] == agent
    assert report["agent_type"] == "deterministic-rule-based"
    assert report["findings"], f"{agent} produced no findings"
    for finding in report["findings"]:
        assert finding["agent_name"] == agent
        assert finding["finding_id"]
        assert finding["severity"] in {"info", "low", "medium", "high", "critical"}
        assert finding["finding"]
        assert finding["evidence_source"].startswith("GET /api/")
        assert finding["limitations"], "every finding must carry limitations"
        assert finding["recommended_action"]


# --------------------------------------------------------------------------- #
# Evidence fidelity: quoted values must equal the source endpoint exactly
# --------------------------------------------------------------------------- #
def test_performance_agent_quotes_api_exactly(client: TestClient) -> None:
    source = _get(client, f"/api/models/{PRIMARY}/performance")
    report = _get(client, f"/api/agents/performance?model_name={PRIMARY}")
    by_id = {f["finding_id"]: f for f in report["findings"]}

    headline = by_id["PERF-01"]["evidence"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert headline[key] == source["metrics"][key], f"{key} does not match the API"
    assert headline["n_test"] == source["n_test"]

    errors = by_id["PERF-02"]["evidence"]
    assert errors["false_negatives"] == source["error_analysis"]["false_negatives"]
    assert errors["false_positives"] == source["error_analysis"]["false_positives"]
    assert errors["true_positives"] == source["confusion_matrix"]["true_positives"]
    assert errors["true_negatives"] == source["confusion_matrix"]["true_negatives"]

    threshold = by_id["PERF-03"]["evidence"]
    assert threshold["decision_threshold"] == source["decision_threshold"]
    assert threshold["roc_auc"] == source["metrics"]["roc_auc"]


def test_fairness_agent_quotes_api_exactly(client: TestClient) -> None:
    source = _get(client, f"/api/models/{PRIMARY}/fairness")
    report = _get(client, f"/api/agents/fairness?model_name={PRIMARY}")

    summaries = {s["attribute"]: s for s in source["summary"]}
    quoted = {
        f["evidence"]["attribute"]: f["evidence"]
        for f in report["findings"]
        if f["finding_id"].startswith("FAIR-") and "attribute" in f["evidence"]
    }
    assert set(quoted) == set(summaries), "one finding per sensitive attribute expected"

    for attribute, evidence in quoted.items():
        expected = summaries[attribute]
        for key in (
            "reference_group",
            "disparate_impact_ratio_min",
            "disparate_impact_ratio_worst_group",
            "groups_failing_four_fifths",
            "equal_opportunity_difference_max_abs",
            "equal_opportunity_worst_group",
            "equalized_odds_difference",
        ):
            assert evidence[key] == expected[key], f"{attribute}.{key} does not match"


def test_fairness_agent_small_group_evidence_matches_api(client: TestClient) -> None:
    source = _get(client, f"/api/models/{PRIMARY}/fairness")
    report = _get(client, f"/api/agents/fairness?model_name={PRIMARY}")

    expected = {
        g["group"]: g for g in source["groups"] if g["small_group_flag"]
    }
    finding = next(f for f in report["findings"] if f["finding_id"] == "FAIR-SMALL")
    quoted = {g["group"]: g for g in finding["evidence"]["small_groups"]}

    assert set(quoted) == set(expected)
    for group, values in quoted.items():
        assert values["n_samples"] == expected[group]["n_samples"]
        assert values["n_actual_positive"] == expected[group]["n_actual_positive"]
        assert values["tpr_ci95"] == expected[group]["tpr_ci95"]
        assert values["selection_rate_ci95"] == expected[group]["selection_rate_ci95"]


def test_fairness_agent_uses_the_audits_own_four_fifths_flag(client: TestClient) -> None:
    """Severity must follow the audit's boolean, not a re-derived comparison."""
    source = _get(client, f"/api/models/{PRIMARY}/fairness")
    report = _get(client, f"/api/agents/fairness?model_name={PRIMARY}")

    for finding in report["findings"]:
        evidence = finding["evidence"]
        if "groups_below_screening_threshold" not in evidence:
            continue
        expected = [
            g["group"]
            for g in source["groups"]
            if g["attribute"] == evidence["attribute"] and g["fails_four_fifths_rule"]
        ]
        assert evidence["groups_below_screening_threshold"] == expected
        assert finding["severity"] == ("high" if expected else "medium")


def test_explainability_agent_quotes_api_exactly(client: TestClient) -> None:
    source = _get(client, f"/api/models/{PRIMARY}/explainability")
    report = _get(client, f"/api/agents/explainability?model_name={PRIMARY}")
    by_id = {f["finding_id"]: f for f in report["findings"]}

    top = by_id["EXPL-01"]["evidence"]
    assert top["n_features"] == source["n_features"]
    assert top["scorer"] == source["scorer"]
    for quoted, expected in zip(top["top_features"], source["global_importance"][:5]):
        assert quoted["feature"] == expected["feature"]
        assert quoted["rank"] == expected["rank"]
        assert quoted["importance_mean"] == expected["importance_mean"]
        assert quoted["importance_std"] == expected["importance_std"]

    proxy = by_id["EXPL-02"]["evidence"]
    assert (
        proxy["protected_attribute_ranks"]
        == source["proxy_assessment"]["protected_attribute_ranks"]
    )
    assert (
        proxy["proxies_ranked_above_all_protected_attributes"]
        == source["proxy_assessment"]["proxies_ranked_above_all_protected_attributes"]
    )

    local = by_id["EXPL-04"]["evidence"]
    assert local["n_local_cases"] == len(source["local_explanations"])
    assert local["case_ids"] == [c["case_id"] for c in source["local_explanations"]]


def test_risk_agent_quotes_governance_api_exactly(client: TestClient) -> None:
    decision = _get(client, "/api/governance/decision")
    register = _get(client, "/api/governance/risks")
    report = _get(client, f"/api/agents/risk?model_name={PRIMARY}")
    by_id = {f["finding_id"]: f for f in report["findings"]}

    stated = by_id["RISK-01"]["evidence"]
    for key in ("research_use", "real_world_deployment", "headline", "decision_date", "subject"):
        assert stated[key] == decision[key], f"{key} does not match the decision record"
    assert stated["grounds_for_deployment_block"] == decision["grounds_for_deployment_block"]

    critical = by_id["RISK-02"]["evidence"]
    assert critical["total_in_register"] == register["total_in_register"]
    assert critical["counts_by_overall_risk"] == register["counts_by_overall_risk"]
    expected_critical = [
        r["risk_id"] for r in register["risks"] if r["overall_risk"] == "Critical"
    ]
    assert critical["critical_risk_ids"] == expected_critical

    blocking = by_id["RISK-03"]["evidence"]
    assert blocking["blocking_risk_ids"] == decision["blocking_risk_ids"]

    conditions = by_id["RISK-04"]["evidence"]
    assert conditions["conditions_on_research_use"] == decision["conditions_on_research_use"]


# --------------------------------------------------------------------------- #
# Language constraints
# --------------------------------------------------------------------------- #
def test_fairness_agent_makes_no_legal_or_causal_claim(client: TestClient) -> None:
    report = _get(client, f"/api/agents/fairness?model_name={PRIMARY}")
    text = str(report).lower()

    for forbidden in (
        "is illegal",
        "is unlawful",
        "legally discriminat",
        "proves discrimination",
        "demonstrates discrimination",
        "constitutes discrimination",
        "violates the law",
        "causal discrimination",
    ):
        assert forbidden not in text, f"fairness agent asserted: {forbidden!r}"

    # And the disclaiming statements must be present on the findings.
    for finding in report["findings"]:
        joined = " ".join(finding["limitations"]).lower()
        assert "not a finding of unlawful discrimination" in joined
        assert "no causal effect" in joined
        assert "screening convention" in joined or "not a statistical test" in joined


def test_explainability_agent_separates_association_from_causation(
    client: TestClient,
) -> None:
    report = _get(client, f"/api/agents/explainability?model_name={PRIMARY}")
    for finding in report["findings"]:
        joined = " ".join(finding["limitations"]).lower()
        assert "association, not causation" in joined
    text = str(report).lower()
    assert "not evidence of fairness" in text
    assert "would not be a mitigation" in text


# --------------------------------------------------------------------------- #
# Unavailable evidence -- never invented
# --------------------------------------------------------------------------- #
def test_explainability_unavailable_is_reported_not_invented(client: TestClient) -> None:
    report = _get(client, "/api/agents/explainability?model_name=random_forest")
    assert report["status"] == "unavailable"
    assert len(report["findings"]) == 1

    finding = report["findings"][0]
    assert finding["finding_id"] == "EXPL-NA"
    assert finding["severity"] == "info"
    assert set(finding["evidence"]["available_models"]) == {"xgboost", "logistic_regression"}
    # No importance numbers may appear anywhere in an unavailable report.
    assert "importance_mean" not in str(report)
    assert "absence of evidence is not evidence of absence" in str(finding["limitations"]).lower()


def test_review_lists_unavailable_evidence(client: TestClient) -> None:
    body = _get(client, "/api/agents/review?model_name=random_forest")
    assert body["unavailable_evidence"], "review must declare missing evidence"
    statuses = {a["agent_name"]: a["status"] for a in body["agents"]}
    assert statuses["explainability"] == "unavailable"
    assert statuses["performance"] == "ok"
    assert statuses["fairness"] == "ok"
    assert statuses["risk"] == "ok"


# --------------------------------------------------------------------------- #
# Decision preservation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["xgboost", "random_forest", "logistic_regression"])
def test_decision_is_preserved_verbatim(client: TestClient, model: str) -> None:
    decision = _get(client, "/api/governance/decision")
    review = _get(client, f"/api/agents/review?model_name={model}")
    preserved = review["preserved_decision"]

    assert preserved["research_use"] == decision["research_use"] == "conditionally_approved"
    assert preserved["real_world_deployment"] == decision["real_world_deployment"] == "blocked"
    assert preserved["headline"] == decision["headline"]
    assert preserved["blocking_risk_ids"] == decision["blocking_risk_ids"]
    # The orchestrator must restate, never generate, the recommendation.
    assert review["overall_recommendation"] == decision["headline"]
    assert "research" in review["overall_recommendation"].lower()
    assert "blocked from real-world deployment" in review["overall_recommendation"].lower()


def test_review_cannot_upgrade_the_decision(client: TestClient) -> None:
    """No agent output may read as an approval for deployment."""
    review = _get(client, f"/api/agents/review?model_name={PRIMARY}")
    text = str(review).lower()
    assert "approved for deployment" not in text
    assert "approved for production" not in text
    assert "cleared for deployment" not in text
    assert "cannot alter, soften or override" in review["preserved_decision"]["note"]


# --------------------------------------------------------------------------- #
# Orchestration integrity
# --------------------------------------------------------------------------- #
def test_review_structure_and_severity_counts(client: TestClient) -> None:
    review = _get(client, f"/api/agents/review?model_name={PRIMARY}")
    assert review["agents_run"] == AGENT_NAMES
    assert [a["agent_name"] for a in review["agents"]] == AGENT_NAMES

    findings = [f for a in review["agents"] for f in a["findings"]]
    assert review["findings_total"] == len(findings)

    expected: dict[str, int] = {k: 0 for k in review["severity_counts"]}
    for finding in findings:
        expected[finding["severity"]] += 1
    assert review["severity_counts"] == expected

    rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    assert review["highest_severity"] == max(
        (f["severity"] for f in findings), key=lambda s: rank[s]
    )
    assert "not autonomous decision-makers" in review["disclaimer"]


def test_single_agent_matches_its_report_inside_the_review(client: TestClient) -> None:
    review = _get(client, f"/api/agents/review?model_name={PRIMARY}")
    for name in AGENT_NAMES:
        standalone = _get(client, f"/api/agents/{name}?model_name={PRIMARY}")
        embedded = next(a for a in review["agents"] if a["agent_name"] == name)
        assert standalone == embedded, f"{name} differs between endpoints"


def test_output_is_deterministic(client: TestClient) -> None:
    """Same artefacts -> byte-identical review. No timestamps, no randomness."""
    first = _get(client, f"/api/agents/review?model_name={PRIMARY}")
    second = _get(client, f"/api/agents/review?model_name={PRIMARY}")
    assert first == second
    # No timestamp FIELD may exist (the determinism note mentions the word, so
    # check keys rather than substrings of the serialised payload).
    for forbidden_key in ("generated_at", "timestamp", "created_at", "run_id"):
        assert forbidden_key not in first
        for report in first["agents"]:
            assert forbidden_key not in report


def test_agents_do_not_modify_any_artifact(client: TestClient) -> None:
    watched = [
        RESULTS / "model_metrics.csv",
        RESULTS / "fairness" / "fairness_metrics_by_group.csv",
        RESULTS / "explainability" / "global_feature_importance.csv",
        RESULTS / "governance" / "governance_risk_register.csv",
        RESULTS / "governance" / "governance_summary.md",
    ]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}

    for model in ("xgboost", "random_forest", "logistic_regression"):
        _get(client, f"/api/agents/review?model_name={model}")
        for name in AGENT_NAMES:
            client.get(f"/api/agents/{name}?model_name={model}")

    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}
    assert before == after, "running the agents modified an artefact"


def test_agent_endpoints_are_read_only(client: TestClient) -> None:
    assert client.post("/api/agents").status_code == 405
    assert client.delete(f"/api/agents/review?model_name={PRIMARY}").status_code == 405


def test_existing_endpoints_unchanged_by_agent_layer(client: TestClient) -> None:
    """Phase 5 responses must be untouched by adding the agent routes."""
    for path in (
        "/health",
        "/api/models",
        f"/api/models/{PRIMARY}/performance",
        f"/api/models/{PRIMARY}/fairness",
        f"/api/models/{PRIMARY}/explainability",
        "/api/governance/decision",
        "/api/governance/risks",
        "/api/governance/model-card",
    ):
        assert client.get(path).status_code == 200
