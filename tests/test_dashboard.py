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
    parts: list[str] = []
    for element in ("markdown", "caption", "info", "warning", "error", "success", "text"):
        try:
            parts += [getattr(item, "value", "") for item in getattr(at, element)]
        except AttributeError:
            continue
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
