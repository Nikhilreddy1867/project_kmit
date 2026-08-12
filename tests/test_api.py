"""
tests/test_api.py
=================
Basic API tests for the Phase 5 governance backend.

Run from the repo root with the venv active:

    pytest -q

The tests assert three things:

1. Endpoints respond and are shaped as the schemas promise.
2. **Served values match the audit CSVs exactly** -- the API must not recompute,
   round or otherwise alter a metric. This is the contract that keeps the API from
   drifting away from the committed evidence, so it is tested explicitly rather
   than trusted.
3. Errors are meaningful: unknown names give 404 with the available alternatives,
   and the read-only contract holds (mutating verbs are rejected).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok", f"missing artefacts: {body['missing_artifacts']}"
    assert body["mode"] == "read-only"
    assert body["service"] == "ai-governance-platform"
    assert body["artifacts_present"] == body["artifacts_expected"]
    assert body["missing_artifacts"] == []
    assert all(a["present"] for a in body["artifacts"])


def test_health_reports_every_declared_artifact(client: TestClient) -> None:
    keys = {a["key"] for a in client.get("/health").json()["artifacts"]}
    for expected in (
        "phase1_metrics",
        "fairness_by_group",
        "fairness_summary",
        "global_importance",
        "local_explanations",
        "risk_register",
        "governance_summary",
        "model_card",
    ):
        assert expected in keys


def test_openapi_and_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "MAAT -- Multi-Agent AI Audit and Trust Framework API"
    for route in (
        "/health",
        "/api/models",
        "/api/models/{model_name}/performance",
        "/api/models/{model_name}/fairness",
        "/api/models/{model_name}/explainability",
        "/api/governance/decision",
        "/api/governance/risks",
        "/api/governance/model-card",
    ):
        assert route in spec["paths"], f"{route} missing from OpenAPI schema"


# --------------------------------------------------------------------------- #
# Models / performance
# --------------------------------------------------------------------------- #
def test_list_models_matches_phase1_csv(client: TestClient) -> None:
    rows = _csv_rows(RESULTS / "model_metrics.csv")
    body = client.get("/api/models").json()

    assert body["count"] == len(rows)
    assert [m["model_name"] for m in body["models"]] == [r["model"] for r in rows]
    assert body["decision_threshold"] == 0.5
    assert body["positive_class"] == ">50K"

    # Values must be identical to the CSV, not merely close.
    for served, expected in zip(body["models"], rows):
        assert served["metrics"]["accuracy"] == float(expected["accuracy"])
        assert served["metrics"]["roc_auc"] == float(expected["roc_auc"])
        assert served["metrics"]["f1"] == float(expected["f1"])
        assert served["confusion_matrix"]["false_negatives"] == int(
            expected["false_negatives"]
        )


def test_primary_model_flagged_once(client: TestClient) -> None:
    models = client.get("/api/models").json()["models"]
    primary = [m for m in models if m["is_primary"]]
    assert len(primary) == 1
    assert primary[0]["model_name"] == "xgboost"


def test_performance_matches_csv_exactly(client: TestClient) -> None:
    expected = {r["model"]: r for r in _csv_rows(RESULTS / "model_metrics.csv")}["xgboost"]
    body = client.get("/api/models/xgboost/performance").json()

    assert body["model_name"] == "xgboost"
    assert body["is_primary"] is True
    assert body["n_test"] == int(expected["n_test"])
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert body["metrics"][key] == float(expected[key]), f"{key} was altered"
    for key in (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ):
        assert body["confusion_matrix"][key] == int(expected[key])
    assert body["caveats"], "performance response must carry interpretation caveats"


def test_unknown_model_returns_404_with_alternatives(client: TestClient) -> None:
    r = client.get("/api/models/not_a_model/performance")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "xgboost" in body["available"]


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def test_fairness_matches_audit_csv(client: TestClient) -> None:
    rows = [
        r
        for r in _csv_rows(RESULTS / "fairness" / "fairness_metrics_by_group.csv")
        if r["model"] == "xgboost"
    ]
    body = client.get("/api/models/xgboost/fairness").json()

    assert len(body["groups"]) == len(rows)
    assert sorted(body["sensitive_attributes"]) == ["race", "sex"]

    served = {(g["attribute"], g["group"]): g for g in body["groups"]}
    for row in rows:
        g = served[(row["attribute"], row["group"])]
        assert g["selection_rate"] == float(row["selection_rate"])
        assert g["tpr"] == float(row["tpr"])
        assert g["fpr"] == float(row["fpr"])
        assert g["disparate_impact_ratio"] == float(row["disparate_impact_ratio"])
        assert g["n_samples"] == int(row["n_samples"])


def test_fairness_reference_groups_and_small_group_flags(client: TestClient) -> None:
    body = client.get("/api/models/xgboost/fairness").json()

    refs = {g["attribute"]: g["group"] for g in body["groups"] if g["is_reference"]}
    assert refs == {"sex": "Male", "race": "White"}

    flagged = {g["group"] for g in body["groups"] if g["small_group_flag"]}
    assert flagged == {"Amer-Indian-Eskimo", "Other"}

    # The response must state what these metrics do NOT establish.
    text = " ".join(body["interpretation"]["does_not_establish"]).lower()
    assert "discrimination" in text and "causation" in text


def test_fairness_attribute_filter(client: TestClient) -> None:
    body = client.get("/api/models/xgboost/fairness?attribute=sex").json()
    assert {g["attribute"] for g in body["groups"]} == {"sex"}
    assert len(body["groups"]) == 2

    bad = client.get("/api/models/xgboost/fairness?attribute=height")
    assert bad.status_code == 404
    assert "sex" in bad.json()["available"]


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_explainability_matches_audit_csv(client: TestClient) -> None:
    rows = _csv_rows(RESULTS / "explainability" / "global_feature_importance.csv")
    body = client.get("/api/models/xgboost/explainability").json()

    assert body["n_features"] == len(rows)
    assert len(body["global_importance"]) == len(rows)

    served = {f["feature"]: f for f in body["global_importance"]}
    for row in rows:
        f = served[row["feature"]]
        assert f["importance_mean"] == float(row["xgboost_perm_roc_auc_mean"])
        assert f["rank"] == int(row["rank"])


def test_explainability_proxy_assessment(client: TestClient) -> None:
    body = client.get("/api/models/xgboost/explainability").json()
    served = {f["feature"]: f for f in body["global_importance"]}

    assert served["sex"]["classification"] == "protected"
    assert served["race"]["classification"] == "protected"
    assert served["marital-status"]["classification"] == "likely_proxy"

    proxy = body["proxy_assessment"]
    assert proxy["protected_attribute_ranks"]["sex"] == 10
    assert proxy["protected_attribute_ranks"]["race"] == 11
    # relationship is sex-coded and must be reported as outranking both.
    assert "relationship" in proxy["proxies_ranked_above_all_protected_attributes"]
    assert "not evidence of fairness" in proxy["implication"].lower()


def test_explainability_local_cases_for_primary_model(client: TestClient) -> None:
    body = client.get("/api/models/xgboost/explainability").json()
    cases = {c["case_id"] for c in body["local_explanations"]}
    assert cases == {
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "borderline_near_0.5",
    }
    for case in body["local_explanations"]:
        assert case["top_factors"]
        assert case["top_factors"][0]["factor_rank"] == 1


def test_explainability_top_n_limit(client: TestClient) -> None:
    body = client.get("/api/models/xgboost/explainability?top_n=5").json()
    assert len(body["global_importance"]) == 5
    assert body["global_importance"][0]["feature"] == "marital-status"


def test_explainability_unaudited_model_is_404_not_fabricated(client: TestClient) -> None:
    """random_forest has no Phase 3 audit; the API must refuse, not invent numbers."""
    r = client.get("/api/models/random_forest/explainability")
    assert r.status_code == 404
    assert set(r.json()["available"]) == {"xgboost", "logistic_regression"}


def test_comparison_model_explainability_has_no_local_cases(client: TestClient) -> None:
    body = client.get("/api/models/logistic_regression/explainability").json()
    assert body["is_primary"] is False
    assert body["local_explanations"] == []
    assert len(body["global_importance"]) == 14


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #
def test_decision_is_research_only_and_deployment_blocked(client: TestClient) -> None:
    body = client.get("/api/governance/decision").json()
    assert body["research_use"] == "conditionally_approved"
    assert body["real_world_deployment"] == "blocked"
    assert body["grounds_for_deployment_block"], "grounds must be parsed from the summary"
    assert body["conditions_on_research_use"], "conditions must be parsed"
    assert body["blocking_risk_ids"], "blocking risks must be identified"
    # No legal or causal claim may be made.
    assert "no finding" in body["disclaimer"].lower()
    assert body["summary_markdown"] is None  # omitted unless requested


def test_decision_can_include_markdown(client: TestClient) -> None:
    body = client.get("/api/governance/decision?include_markdown=true").json()
    assert body["summary_markdown"]
    assert "Decision Record" in body["summary_markdown"]


def test_risks_match_register_csv(client: TestClient) -> None:
    rows = _csv_rows(RESULTS / "governance" / "governance_risk_register.csv")
    body = client.get("/api/governance/risks").json()

    assert body["count"] == len(rows) == body["total_in_register"]
    assert [r["risk_id"] for r in body["risks"]] == [r["risk_id"] for r in rows]

    expected_counts: dict[str, int] = {}
    for row in rows:
        expected_counts[row["overall_risk"]] = expected_counts.get(row["overall_risk"], 0) + 1
    assert body["counts_by_overall_risk"] == expected_counts

    served = {r["risk_id"]: r for r in body["risks"]}
    assert served["R01"]["overall_risk"] == "Critical"
    assert "sex" in served["R01"]["category"].lower()
    for field in ("evidence", "recommended_control", "residual_risk", "owner", "status"):
        assert served["R01"][field], f"R01.{field} must not be empty"


def test_risks_filters(client: TestClient) -> None:
    critical = client.get("/api/governance/risks?overall_risk=Critical").json()
    assert critical["count"] == 5
    assert all(r["overall_risk"] == "Critical" for r in critical["risks"])
    assert critical["total_in_register"] == 12  # unfiltered total still reported

    fairness = client.get("/api/governance/risks?category=Fairness").json()
    assert all("fairness" in r["category"].lower() for r in fairness["risks"])

    blocking = client.get("/api/governance/risks?status=blocking").json()
    assert all("blocking" in r["status"].lower() for r in blocking["risks"])

    empty = client.get("/api/governance/risks?overall_risk=Nonexistent").json()
    assert empty["count"] == 0 and empty["risks"] == []


def test_model_card_content_and_sections(client: TestClient) -> None:
    body = client.get("/api/governance/model-card").json()
    assert body["content"] and body["character_count"] == len(body["content"])
    assert body["sections"]
    joined = " ".join(body["sections"]).lower()
    for topic in ("intended purpose", "non-intended uses", "limitations", "monitoring"):
        assert topic in joined, f"model card must document {topic}"


def test_model_card_sections_only(client: TestClient) -> None:
    body = client.get("/api/governance/model-card?sections_only=true").json()
    assert body["content"] is None
    assert body["sections"]


# --------------------------------------------------------------------------- #
# Read-only contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/models"),
        ("put", "/api/governance/risks"),
        ("delete", "/api/governance/model-card"),
        ("patch", "/api/models/xgboost/performance"),
    ],
)
def test_mutating_verbs_are_rejected(client: TestClient, method: str, path: str) -> None:
    assert getattr(client, method)(path).status_code == 405


def test_cors_allows_local_dashboard_origin(client: TestClient) -> None:
    r = client.get("/api/models", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_foreign_origin(client: TestClient) -> None:
    r = client.get("/api/models", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_read_only_header_present(client: TestClient) -> None:
    assert "read-only" in client.get("/health").headers["X-Data-Source"]


def test_artifacts_are_not_modified_by_serving(client: TestClient) -> None:
    """Hitting every endpoint must leave the evidence files untouched."""
    watched = [
        RESULTS / "model_metrics.csv",
        RESULTS / "fairness" / "fairness_metrics_by_group.csv",
        RESULTS / "explainability" / "global_feature_importance.csv",
        RESULTS / "governance" / "governance_risk_register.csv",
        RESULTS / "governance" / "model_card.md",
    ]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}

    for path in (
        "/health",
        "/api/models",
        "/api/models/xgboost/performance",
        "/api/models/xgboost/fairness",
        "/api/models/xgboost/explainability",
        "/api/governance/decision?include_markdown=true",
        "/api/governance/risks",
        "/api/governance/model-card",
    ):
        assert client.get(path).status_code == 200

    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}
    assert before == after, "serving the API modified an audit artefact"
