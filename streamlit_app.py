"""
streamlit_app.py  (repository root)
===================================
Deployment entry point for Streamlit Community Cloud.

Streamlit Cloud runs a single process and looks for ``streamlit_app.py`` at the
repository root by default. This file is that entry point, and it does two things:

1. starts the Phase 5 governance API **in-process** on loopback
   (see :mod:`app.embedded`), because Cloud cannot run a second server; then
2. hands over to the real dashboard in ``dashboard/streamlit_app.py``, unmodified.

The data contract is unchanged. The dashboard still fetches everything over HTTP
via ``api_client``, still reads no files, and still imports nothing from ``app/``.
All of the deployment glue lives here so those files stay clean and identical
between local and deployed runs.

Local development is unaffected: if a ``uvicorn`` is already serving on port 8000,
this defers to it and starts nothing. You can keep using the two-terminal workflow
in README §3b, or just run this file to get both halves at once:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The dashboard module is loaded under a distinct name and cached in sys.modules.
# This matters: Streamlit re-executes this script on every interaction, and
# re-executing the dashboard module each time would rebuild its @st.cache_data
# functions, invalidating the cache on every click.
_DASHBOARD_MODULE = "governance_dashboard"
_DASHBOARD_PATH = ROOT / "dashboard" / "streamlit_app.py"


def _load_dashboard():
    """Import the dashboard once; return the cached module on later reruns."""
    cached = sys.modules.get(_DASHBOARD_MODULE)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(_DASHBOARD_MODULE, _DASHBOARD_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load the dashboard from {_DASHBOARD_PATH}")

    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module is importable while it is still executing.
    sys.modules[_DASHBOARD_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_DASHBOARD_MODULE, None)
        raise
    return module


def main() -> None:
    from app.embedded import ensure_api_running, ensure_registry

    # Blocks until /health answers, so the first render never races the server.
    ensure_api_running()

    # Populate the Phase 8 registry on a fresh deployment (runtime/ is gitignored,
    # so it does not exist in a clean checkout). Idempotent and non-fatal.
    ensure_registry()

    _load_dashboard().main()


main()
