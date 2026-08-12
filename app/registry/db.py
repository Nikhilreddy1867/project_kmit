"""
app/registry/db.py
==================
SQLite storage for the governance audit registry.

Scope of writes
---------------
This is the **only** module in the platform that writes to disk, and it writes to
exactly one place: a local SQLite file under ``runtime/`` (gitignored). It never
touches ``data/``, ``models/``, ``predictions/`` or ``results/`` -- those remain
read-only evidence.

Schema
------
``audit_runs``      one row per governance review run, keyed by a content-addressed
                    run id.
``run_artifacts``   the artefact manifest for a run: path, group, SHA-256, size and
                    mtime **as at registration time**. This is the baseline the
                    integrity endpoint later re-checks against.
``registry_events`` append-only audit trail: registration, refresh, status change,
                    integrity check. Never updated or deleted, so the timeline is
                    a real history rather than a derived guess.

Run types
---------
``audit_runs.run_type`` separates the two kinds of run the registry now holds:
``reference_case`` (the built-in Adult Income audit) and ``uploaded_model`` (a run
produced from a user submission). Every query that could let one kind affect the
other -- superseding in particular -- is scoped by this column, so registering an
upload can never demote or alter the reference run.

The database path comes from the ``GOVERNANCE_REGISTRY_DB`` environment variable
when set, otherwise ``runtime/governance_registry.db``. Reading it per call (rather
than caching a module constant) is what lets tests point at a temporary file.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DEFAULT_DB_FILENAME = "governance_registry.db"
DB_ENV_VAR = "GOVERNANCE_REGISTRY_DB"

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_ARCHIVED = "archived"
VALID_STATUSES = (STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_ARCHIVED)

#: The built-in Adult Income audit, registered from the committed evidence.
RUN_TYPE_REFERENCE = "reference_case"
#: A run produced from a user-uploaded model and dataset, stored under ``runtime/``.
RUN_TYPE_UPLOADED = "uploaded_model"
VALID_RUN_TYPES = (RUN_TYPE_REFERENCE, RUN_TYPE_UPLOADED)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_runs (
    run_id                TEXT PRIMARY KEY,
    schema_version        INTEGER NOT NULL,
    created_at            TEXT NOT NULL,
    refreshed_at          TEXT NOT NULL,
    refresh_count         INTEGER NOT NULL DEFAULT 1,
    dataset_name          TEXT NOT NULL,
    dataset_version       TEXT NOT NULL,
    dataset_context       TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    model_version          TEXT NOT NULL,
    model_run_identifier  TEXT NOT NULL,
    evidence_digest       TEXT NOT NULL,
    artifact_count        INTEGER NOT NULL,
    performance_summary   TEXT NOT NULL,
    governance_decision   TEXT NOT NULL,
    blocking_risk_ids     TEXT NOT NULL,
    audit_coverage        TEXT NOT NULL,
    status                TEXT NOT NULL,
    run_type              TEXT NOT NULL DEFAULT 'reference_case'
);

CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id        TEXT NOT NULL,
    artifact_group TEXT NOT NULL,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    modified_utc  TEXT NOT NULL,
    PRIMARY KEY (run_id, path),
    FOREIGN KEY (run_id) REFERENCES audit_runs (run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS registry_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    event_time TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail     TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs (run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON audit_runs (status);
CREATE INDEX IF NOT EXISTS idx_events_run ON registry_events (run_id, event_id);
"""

#: Columns added after the original schema shipped, as
#: ``(table, column, definition)``. Every entry must be **additive and defaulted**:
#: applying it to a populated database changes no existing value and leaves every
#: pre-existing row meaning exactly what it meant before. That is why no
#: ``SCHEMA_VERSION`` bump accompanies ``run_type`` -- a v1 row read through the new
#: column is still a correct v1 row, namely a reference-case run.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("audit_runs", "run_type", f"TEXT NOT NULL DEFAULT '{RUN_TYPE_REFERENCE}'"),
)

_POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_runs_type ON audit_runs (run_type, status);
"""


def utc_now() -> str:
    """Current UTC time, ISO-8601, second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """
    Decide where the registry database lives.

    Precedence: explicit argument > ``GOVERNANCE_REGISTRY_DB`` > the default under
    ``runtime/``. Resolved per call so tests can redirect it with an env var.
    """
    if db_path is not None:
        return Path(db_path)
    from_env = os.environ.get(DB_ENV_VAR)
    if from_env:
        return Path(from_env)
    return RUNTIME_DIR / DEFAULT_DB_FILENAME


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Column names of ``table``, or an empty set if the table does not exist."""
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def migrate(connection: sqlite3.Connection) -> list[str]:
    """
    Apply the additive column migrations, if any are missing. Idempotent.

    Returns the columns actually added, so a caller can log a real change rather
    than announcing work that did not happen.

    Run on **every** connection, including read-only ones, for a specific reason:
    a database created before ``run_type`` existed would otherwise raise on the
    first read that referenced the column. Migrating on open means an older
    registry keeps working without anyone being told to run a migration command.
    A failure here is swallowed rather than raised -- the caller wanted to read the
    registry, and a read should not die because the file happens to be on
    read-only storage.
    """
    added: list[str] = []
    try:
        for table, column, definition in _ADDITIVE_COLUMNS:
            existing = table_columns(connection, table)
            if not existing or column in existing:
                continue
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            added.append(f"{table}.{column}")
        if added:
            connection.executescript(_POST_MIGRATION_INDEXES)
        connection.commit()
    except sqlite3.OperationalError:
        return added
    return added


def connect(db_path: str | Path | None = None, *, create: bool = True) -> sqlite3.Connection:
    """
    Open the registry database.

    With ``create=False`` a missing file raises :class:`FileNotFoundError`, which
    the API turns into a 503 telling the caller to run the register command --
    better than silently serving an empty registry as though no audit existed.
    """
    path = resolve_db_path(db_path)
    if not path.exists():
        if not create:
            raise FileNotFoundError(
                f"Registry database not found at {path}. "
                "Create it with: python -m app.registry.cli register"
            )
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if create:
        connection.executescript(_SCHEMA)
        connection.executescript(_POST_MIGRATION_INDEXES)
        connection.commit()
    migrate(connection)
    return connection


def init_db(db_path: str | Path | None = None) -> Path:
    """Create the database and schema if absent. Idempotent."""
    path = resolve_db_path(db_path)
    with connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.executescript(_POST_MIGRATION_INDEXES)
        connection.commit()
        migrate(connection)
    return path


# --------------------------------------------------------------------------- #
# JSON helpers -- SQLite has no native JSON column type
# --------------------------------------------------------------------------- #
def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def loads(value: str | None) -> Any:
    if value in (None, ""):
        return None
    return json.loads(value)


def record_event(
    connection: sqlite3.Connection, run_id: str, event_type: str, detail: str
) -> None:
    """Append an event. The events table is never updated or deleted from."""
    connection.execute(
        "INSERT INTO registry_events (run_id, event_time, event_type, detail) "
        "VALUES (?, ?, ?, ?)",
        (run_id, utc_now(), event_type, detail),
    )
