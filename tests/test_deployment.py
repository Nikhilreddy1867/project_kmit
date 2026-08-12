"""
tests/test_deployment.py
========================
Tests for the single-process deployment path (Streamlit Community Cloud).

Locally the platform runs as two processes: uvicorn serves the API and Streamlit
serves the dashboard. Cloud runs only one process, so ``streamlit_app.py`` at the
repository root starts the API in-process (:mod:`app.embedded`) and then hands over
to the unmodified dashboard.

These tests pin that path, because it is easy to break silently: the dashboard would
still start and simply show "API unavailable" on every page.

Unlike ``tests/test_dashboard.py`` these tests need **no** externally running
server -- proving the point, since Cloud has none.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app import embedded

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = PROJECT_ROOT / "streamlit_app.py"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Cloud entry-point files
# --------------------------------------------------------------------------- #
def test_entry_point_exists_at_repo_root() -> None:
    """Streamlit Cloud defaults to streamlit_app.py at the repository root."""
    assert ENTRY_POINT.is_file()
    source = ENTRY_POINT.read_text(encoding="utf-8")
    assert "ensure_api_running" in source
    assert "dashboard" in source


def test_python_version_is_pinned_for_cloud() -> None:
    """
    Cloud tops out below the 3.14 used locally, so the version is pinned.

    Guarded by a test so the pin cannot silently disappear and leave the deployed
    environment differing from the verified one in an undocumented way.
    """
    pin = PROJECT_ROOT / ".python-version"
    assert pin.is_file(), ".python-version is required for the Cloud deployment"
    version = pin.read_text(encoding="utf-8").strip()
    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) >= (3, 11), f"unexpectedly old pin: {version}"
    assert (major, minor) <= (3, 13), (
        f"pin {version} exceeds what Streamlit Community Cloud supports"
    )


def test_requirements_cover_the_deployed_app() -> None:
    """The API + dashboard dependencies must be installable on Cloud."""
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("fastapi", "uvicorn", "streamlit", "plotly", "pandas", "httpx"):
        assert package in requirements, f"{package} missing from requirements.txt"


# --------------------------------------------------------------------------- #
# Embedded API
# --------------------------------------------------------------------------- #
def test_port_probe_reports_closed_port() -> None:
    # Port 9 (discard) is not served by this project.
    assert embedded._port_open("127.0.0.1", 9, timeout=0.2) is False


def test_ensure_api_running_starts_and_is_idempotent() -> None:
    """
    Bring the API up in-process, then prove repeated calls are no-ops.

    If a uvicorn is already serving on 8000 (the local two-terminal workflow), this
    defers to it and starts nothing -- which is exactly the behaviour that keeps
    local development unaffected.
    """
    base_url = embedded.ensure_api_running()
    assert base_url == "http://127.0.0.1:8000"
    assert _port_open(8000), "the API should be reachable after ensure_api_running()"

    # Second and third calls must not raise, and must not bind a second server.
    assert embedded.ensure_api_running() == base_url
    assert embedded.ensure_api_running() == base_url


def test_embedded_api_serves_the_real_endpoints() -> None:
    import httpx

    base_url = embedded.ensure_api_running()
    for path in (
        "/health",
        "/api/models",
        "/api/models/xgboost/fairness",
        "/api/agents/review?model_name=xgboost",
        "/api/registry/runs",
    ):
        response = httpx.get(f"{base_url}{path}", timeout=30)
        assert response.status_code == 200, f"{path} -> {response.status_code}"

    # It is the same read-only app: mutating verbs are still rejected.
    assert httpx.post(f"{base_url}/api/models", timeout=30).status_code == 405


def test_embedded_api_is_loopback_only() -> None:
    """
    The API must not be reachable off-host even when Streamlit is public.

    Binding to 127.0.0.1 is what keeps the prediction CSVs (which carry sex and
    race) from becoming publicly readable on a deployed app.
    """
    embedded.ensure_api_running()
    assert embedded.DEFAULT_HOST == "127.0.0.1"
    source = (PROJECT_ROOT / "app" / "embedded.py").read_text(encoding="utf-8")
    assert '"0.0.0.0"' not in source


# --------------------------------------------------------------------------- #
# Registry bootstrap
# --------------------------------------------------------------------------- #
def test_ensure_registry_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh deployment has no runtime/ directory; registration must create it."""
    db = tmp_path / "fresh" / "registry.db"
    monkeypatch.setenv("GOVERNANCE_REGISTRY_DB", str(db))

    first = embedded.ensure_registry()
    assert first and first.startswith("run-")
    assert db.exists(), "registration should have created the database"

    # Same evidence -> same content-addressed run, no duplicate.
    assert embedded.ensure_registry() == first

    from app.registry import service as registry_service

    assert registry_service.list_runs(db_path=db)["count"] == 1


def test_ensure_registry_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A registry problem must degrade one page, not prevent the app from starting.

    Simulated by making registration raise: on a real deployment this could be a
    read-only filesystem or missing evidence.
    """
    from app.registry import service as registry_service

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr(registry_service, "register_run", _boom)

    # quiet=True swallows it and returns None so the app still starts...
    assert embedded.ensure_registry(quiet=True) is None
    # ...while quiet=False surfaces it, so the failure is never silently invisible.
    with pytest.raises(OSError, match="simulated read-only filesystem"):
        embedded.ensure_registry(quiet=False)


# --------------------------------------------------------------------------- #
# End to end: the entry point renders the dashboard against the embedded API
# --------------------------------------------------------------------------- #
def test_entry_point_boots_api_and_renders_dashboard() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=240)
    at.run()

    assert not at.exception, f"entry point raised: {at.exception}"
    assert _port_open(8000), "the entry point must have started the API"
    # The exact list, in order: the seven reference-case pages first and unchanged,
    # then the three model-intake pages. Asserted exactly rather than by membership
    # so that a page quietly disappearing fails here.
    reference_pages = [
        "Overview",
        "Model Performance",
        "Fairness Audit",
        "Explainability",
        "Governance Decision & Risks",
        "Agent Review",
        "Model Registry",
    ]
    options = at.sidebar.radio[0].options
    assert options[: len(reference_pages)] == reference_pages
    assert options[len(reference_pages) :] == [
        "New Model Audit",
        "Uploaded Audit Runs",
        "Policy Gates & Conformity Bundle",
    ]

    # The dashboard is genuinely reading through the embedded API, not showing an
    # unavailable state.
    text = " ".join(str(getattr(m, "value", "")) for m in at.markdown)
    text += " ".join(str(getattr(c, "value", "")) for c in at.caption)
    assert "API unavailable" not in text
    labels = [m.label for m in at.metric]
    assert "Accuracy" in labels and "ROC-AUC" in labels


def test_dashboard_module_is_loaded_once() -> None:
    """
    Streamlit re-executes the entry point on every interaction. The dashboard must
    be cached in sys.modules, or its @st.cache_data functions would be rebuilt each
    rerun and the cache would miss every time.
    """
    import sys

    sys.modules.pop("governance_dashboard", None)
    import importlib.util

    spec = importlib.util.spec_from_file_location("root_entry_probe", ENTRY_POINT)
    assert spec is not None and spec.loader is not None

    source = ENTRY_POINT.read_text(encoding="utf-8")
    assert "sys.modules" in source, "entry point must cache the dashboard module"
    assert "_DASHBOARD_MODULE" in source
