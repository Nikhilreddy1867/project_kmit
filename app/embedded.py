"""
app/embedded.py
===============
Runs the governance API **inside the current process**, on a background thread.

Why this exists
---------------
Streamlit Community Cloud runs exactly one process per app. The dashboard, however,
is deliberately built to obtain all of its data over HTTP from the Phase 5 API
(``http://127.0.0.1:8000``) rather than by reading files. Locally that means two
terminals; on Cloud there is only one process, so a separately-launched uvicorn
cannot exist.

This module closes that gap **without weakening the data contract**: the API is
started in a daemon thread bound to loopback, and the dashboard keeps talking to it
over HTTP through ``api_client`` exactly as before. What changes is *who starts the
server*, not how data reaches the UI. The dashboard still reads no files, imports
nothing from ``app/``, and recomputes nothing.

Safety properties
-----------------
* **Idempotent.** Streamlit re-executes its script on every interaction, so
  :func:`ensure_api_running` is guarded by a lock and a module-level singleton, and
  it first probes the port. Repeated calls are cheap no-ops.
* **Never double-binds.** If something is already serving on the port -- a locally
  launched ``uvicorn`` during development -- this defers to it and starts nothing.
* **Loopback only.** Bound to ``127.0.0.1``, so the API is not exposed publicly even
  when the Streamlit app is. Only the dashboard, in-process, can reach it.
* **Still read-only.** This starts the same read-only app defined in
  ``app.main``; it adds no routes and no write paths.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Final

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8000
_READY_TIMEOUT: Final = 45.0
_POLL_INTERVAL: Final = 0.25

_lock = threading.Lock()
_server_thread: threading.Thread | None = None
_started_here = False


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """True if something is already accepting connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run_server(host: str, port: int) -> None:
    """Serve ``app.main:app`` on this thread until the process exits."""
    import uvicorn

    from app.main import app

    class _ThreadedServer(uvicorn.Server):
        def install_signal_handlers(self) -> None:
            # Signal handlers can only be installed on the main thread, and the
            # daemon thread dies with the process anyway.
            return None

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        loop="asyncio",
    )
    _ThreadedServer(config).run()


def ensure_api_running(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = _READY_TIMEOUT,
) -> str:
    """
    Guarantee the governance API is reachable, and return its base URL.

    Starts the in-process server only if nothing is already listening. Blocks until
    ``/health`` answers, so the first Streamlit render never races the server.
    """
    global _server_thread, _started_here
    base_url = f"http://{host}:{port}"

    if _port_open(host, port):
        return base_url

    with _lock:
        # Re-check inside the lock: another Streamlit thread may have won the race.
        if _port_open(host, port):
            return base_url
        if _server_thread is None or not _server_thread.is_alive():
            _server_thread = threading.Thread(
                target=_run_server,
                args=(host, port),
                name="governance-api",
                daemon=True,
            )
            _server_thread.start()
            _started_here = True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(host, port):
            return base_url
        time.sleep(_POLL_INTERVAL)

    raise RuntimeError(
        f"The embedded governance API did not become ready on {base_url} within "
        f"{timeout:g}s. Check the application logs for a startup error."
    )


def is_embedded() -> bool:
    """True when this process started the API itself."""
    return _started_here


def ensure_registry(quiet: bool = True) -> str | None:
    """
    Make sure the Phase 8 registry exists, so the Model Registry page has data.

    On a fresh deployment ``runtime/`` does not exist (it is gitignored local
    state), which would leave that page showing its "registry not created yet"
    state. Registering here reads the committed evidence and writes only the
    SQLite file, and it is idempotent -- unchanged evidence refreshes the same
    content-addressed run rather than creating duplicates.

    Failure is deliberately non-fatal: a dashboard that works with one page
    degraded is better than one that will not start. Returns the run id, or
    ``None`` if registration could not complete.
    """
    try:
        from app.registry import service as registry_service

        result = registry_service.register_run()
        return str(result.get("run_id"))
    except Exception as exc:  # pragma: no cover - defensive by intent
        if not quiet:
            raise
        import logging

        logging.getLogger(__name__).warning(
            "Registry auto-registration skipped: %s: %s", type(exc).__name__, exc
        )
        return None
