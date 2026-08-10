"""
tests/test_dashboard.py
=======================
Headless verification of the Phase 6 Streamlit dashboard.

Uses Streamlit's official ``AppTest`` harness to execute the real app script for
every sidebar page and assert on what it rendered. This is an **integration**
test: it needs the Phase 5 API actually running, so it skips cleanly when the API
is not reachable rather than failing.

    # terminal 1
    uvicorn app.main:app --port 8000
    # terminal 2
    pytest tests/test_dashboard.py -q

What it checks
--------------
* every page runs without raising;
* the governance decision wording appears where required;
* the four-fifths context is never presented as a legal conclusion;
* a model with no explainability audit produces an explicit unavailable message;
* an unreachable API degrades to a friendly error instead of a traceback.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "streamlit_app.py")
API = "http://127.0.0.1:8000"
PAGES = [
    "Overview",
    "Model Performance",
    "Fairness Audit",
    "Explainability",
    "Governance Decision & Risks",
    "Agent Review",
    "Model Registry",
]


def _api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _api_up(), reason=f"governance API not running at {API}"
)


def _run(page: str, timeout: float = 60) -> AppTest:
    """Render one dashboard page and return the finished AppTest."""
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    at.session_state["page"] = page
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    """
    Collect the app's rendered text.

    Includes **expander labels and their nested content**: on the Agent Review page
    each finding is an expander whose label carries the finding id, so a helper that
    only read top-level markdown would miss them.
    """
    parts: list[str] = []
    for element in ("markdown", "caption", "info", "warning", "error", "success", "text"):
        try:
            parts += [getattr(item, "value", "") for item in getattr(at, element)]
        except AttributeError:
            continue
    try:
        for expander in at.expander:
            parts.append(getattr(expander, "label", "") or "")
            for element in ("markdown", "caption", "info", "warning", "error", "success"):
                parts += [
                    getattr(item, "value", "") for item in getattr(expander, element, [])
                ]
    except AttributeError:
        pass
    return " ".join(str(p) for p in parts)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page: str) -> None:
    at = _run(page)
    assert not at.exception, f"{page} raised: {at.exception}"
    assert at.title, f"{page} rendered no title"


def test_overview_shows_decision_and_primary_model() -> None:
    at = _run("Overview")
    text = _all_text(at)
    assert "Conditionally approved for research/education only" in text
    assert "Blocked from real-world deployment" in text
    assert "XGBoost" in " ".join(h.value for h in at.subheader) + text
    # Headline metrics are rendered as metric widgets.
    labels = [m.label for m in at.metric]
    for expected in ("Accuracy", "F1", "ROC-AUC"):
        assert expected in labels


def test_performance_page_has_metrics_and_caveats() -> None:
    at = _run("Model Performance")
    labels = [m.label for m in at.metric]
    for expected in ("Accuracy", "Precision", "Recall / TPR", "F1", "ROC-AUC"):
        assert expected in labels
    assert "False negatives" in labels
    text = _all_text(at)
    # Threshold and error-pattern caveats come from the API.
    assert "0.5 decision threshold" in text or "threshold" in text
    assert at.selectbox("perf_model").value == "xgboost"


def test_fairness_page_charts_table_and_screening_language() -> None:
    at = _run("Fairness Audit")
    assert at.selectbox("fair_model").value
    assert at.selectbox("fair_attribute").value in ("sex", "race")
    assert len(at.dataframe) >= 1, "group table missing"

    text = _all_text(at)
    # Four-fifths must be framed as screening, explicitly not a legal conclusion.
    assert "screening indicator, not a legal conclusion" in text.lower()
    assert "does not establish unlawful discrimination" in text.lower()
    # Non-causation and small-group caveats are surfaced from the API.
    assert "causation" in text.lower()
    assert "small-group uncertainty" in text.lower()


def test_fairness_race_selection_flags_small_groups() -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.session_state["page"] = "Fairness Audit"
    at.run()
    at.selectbox("fair_attribute").set_value("race").run()
    assert not at.exception
    text = _all_text(at)
    assert "Amer-Indian-Eskimo" in text or "Other" in text
    assert "do not rank these groups" in text.lower()


def test_explainability_page_proxy_warnings() -> None:
    at = _run("Explainability")
    assert at.selectbox("explain_model").value == "xgboost"
    text = _all_text(at)
    assert "not evidence of fairness" in text.lower()
    assert "causation" in text.lower()
    assert len(at.dataframe) >= 1, "feature ranking table missing"
    # Local TreeSHAP case selector must be present for the primary model.
    assert at.selectbox("explain_case").value


def test_explainability_unavailable_message_for_unaudited_model() -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.session_state["page"] = "Explainability"
    at.run()
    at.selectbox("explain_model").set_value("random_forest").run()
    assert not at.exception
    text = _all_text(at)
    assert "explainability is not available" in text.lower()
    assert "random_forest" in text
    assert "fabricate" in text.lower() or "404" in text


def test_governance_page_risks_and_model_card() -> None:
    at = _run("Governance Decision & Risks")
    text = _all_text(at)
    assert "Blocked from real-world deployment" in text
    labels = [m.label for m in at.metric]
    assert "Critical" in labels and "Total" in labels
    assert len(at.dataframe) >= 1, "risk register table missing"
    # Model card sections render as expanders.
    assert len(at.expander) >= 5


def test_governance_severity_filter_is_server_side() -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.session_state["page"] = "Governance Decision & Risks"
    at.run()
    at.selectbox("risk_severity").set_value("Critical").run()
    assert not at.exception
    assert "Showing **5** of 12" in _all_text(at)


def test_provenance_section_present_on_every_page() -> None:
    for page in PAGES:
        at = _run(page)
        headers = " ".join(h.value for h in at.subheader)
        assert "Data provenance and limitations" in headers, f"missing on {page}"


def test_agent_review_page_shows_four_agents_and_preserved_decision() -> None:
    at = _run("Agent Review")
    text = _all_text(at)

    # The non-autonomy label must be present and prominent.
    assert "deterministic governance agents, not autonomous" in text.lower()
    assert "do not train models, recalculate metrics" in text.lower()

    # All four agents render as tabs, and the preserved decision is restated.
    assert "Blocked from real-world deployment" in text
    assert "Conditionally approved for research/education only" in text
    assert "cannot alter, soften or override" in text

    labels = [m.label for m in at.metric]
    assert "Findings" in labels and "Highest" in labels
    assert any("Critical" in label for label in labels)
    assert at.selectbox("agent_model").value == "xgboost"


def test_agent_review_page_surfaces_findings_and_evidence_sources() -> None:
    at = _run("Agent Review")
    text = _all_text(at)
    # Every finding carries an evidence source and a recommended action.
    assert "Evidence source:" in text
    assert "Recommended action." in text
    assert "Limitations / caveats" in text
    # Findings from all four agents should be reachable.
    for marker in ("PERF-", "FAIR-", "EXPL-", "RISK-"):
        assert marker in text, f"no {marker} finding rendered"


def test_agent_review_page_reports_unavailable_evidence() -> None:
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.session_state["page"] = "Agent Review"
    at.run()
    at.selectbox("agent_model").set_value("random_forest").run()
    assert not at.exception
    text = _all_text(at)
    assert "evidence unavailable" in text.lower()
    assert "rather than estimating anything" in text.lower()


def test_model_registry_page_renders_run_metadata_and_integrity() -> None:
    at = _run("Model Registry")
    assert not at.exception
    text = _all_text(at)

    # The registry must state that it does not make decisions.
    assert "records evidence; it does not make decisions" in text.lower()

    labels = [m.label for m in at.metric]
    for expected in ("Registered runs", "Active run", "Artefacts", "Checked", "✅ Verified"):
        assert expected in labels, f"missing metric: {expected}"

    # Coverage across all five phases is displayed.
    for phase in ("Performance", "Fairness", "Explainability", "Governance", "Agents"):
        assert phase in labels, f"coverage metric missing: {phase}"

    # Integrity result, digests and the recorded decision.
    assert "integrity:" in text.lower()
    assert "registered digest" in text.lower()
    assert "evidence digest" in text.lower()
    assert "Blocked from real-world deployment" in text
    assert at.selectbox("registry_run").value


def test_model_registry_page_shows_timeline_and_verified_status() -> None:
    at = _run("Model Registry")
    text = _all_text(at)
    # Timeline events from both sources.
    assert "run_registered" in text
    assert "evidence_produced:" in text
    assert "not a signed provenance record" in text
    # With untouched evidence the run must verify.
    assert "verified" in text.lower()
    assert "does not establish" in text.lower()


def test_unreachable_api_degrades_gracefully() -> None:
    """A wrong base URL must produce a friendly error, never a traceback."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.session_state["api_base_url"] = "http://127.0.0.1:9"  # nothing listens here
    at.run()
    assert not at.exception, "unreachable API raised instead of rendering an error"
    text = _all_text(at)
    assert "api unavailable" in text.lower() or "not reachable" in text.lower()
    assert "uvicorn app.main:app" in text, "error should tell the user how to start the API"
