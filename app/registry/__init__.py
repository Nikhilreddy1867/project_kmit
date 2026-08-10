"""
app.registry
============
Phase 8 -- local governance audit registry and evidence-integrity layer.

Records the current Adult Income audit as one governance review run in a local
SQLite database under ``runtime/`` (gitignored), storing SHA-256 checksums of every
referenced artefact so the evidence a conclusion rests on can later be verified.

* :mod:`app.registry.integrity` -- artefact discovery and SHA-256 verification
* :mod:`app.registry.db`        -- SQLite schema and storage (the only writer)
* :mod:`app.registry.service`   -- register/refresh, read, verify, timeline
* :mod:`app.registry.cli`       -- ``python -m app.registry.cli register``

The registry **records evidence; it does not make decisions.** The governance
decision stored on a run is a copy of the committed decision record.
"""

from app.registry.service import (
    RunNotFoundError,
    check_integrity,
    get_run,
    get_timeline,
    list_runs,
    register_run,
    registry_stats,
)

__all__ = [
    "RunNotFoundError",
    "check_integrity",
    "get_run",
    "get_timeline",
    "list_runs",
    "register_run",
    "registry_stats",
]
