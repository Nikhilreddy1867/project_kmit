"""
app/registry/service.py
=======================
Registry operations: register/refresh a run, list runs, read one, verify integrity,
build a timeline.

Idempotency, by construction
----------------------------
A run's id is **content-addressed**: it is derived from the evidence digest, which
is a SHA-256 over the sorted ``path:sha256`` pairs of every registered artefact,
plus the dataset and subject-model identity. Therefore:

* Re-running registration against unchanged evidence produces the **same run id**,
  so the existing row is refreshed in place -- ``created_at`` is preserved,
  ``refreshed_at`` and ``refresh_count`` advance, and no duplicate run appears.
* If any artefact changes, the digest changes, so a **new** run id is created and
  the previously active run is marked ``superseded``. History is never rewritten.

The registry records evidence. It does not evaluate or alter the governance
decision -- the decision is copied from the committed record.

Two kinds of run
---------------
:func:`register_run` registers the built-in Adult Income evidence as a
``reference_case`` run. :func:`register_uploaded_run` indexes a completed
user-submitted audit from ``runtime/audits/<audit_run_id>/`` as an ``uploaded_model``
run. They share the storage and the integrity machinery but are otherwise isolated:
superseding is scoped by run type, so no number of uploads can demote, alter or
outrank the reference case, and ``active_run_id`` continues to mean the active
reference run.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from app.agents import orchestrator
from app.registry import db as registry_db
from app.registry import integrity as integrity_mod
from app.registry.integrity import ArtifactRecord
from app.schemas.models import DecisionResponse, PerformanceResponse
from app.services import artifact_reader as reader
from app.services import governance_service as svc

PROJECT_ROOT = integrity_mod.PROJECT_ROOT

DATASET_NAME = "UCI Adult / Census Income"
DATASET_VERSION = "uci-id-2"
DATASET_CONTEXT = (
    "1994 US Census extract retrieved via ucimlrepo (dataset id 2); 48,842 records, "
    "14 original features, target `income` encoded >50K = 1. The label reflects the "
    "1994 US labour market, including its inequalities."
)

SUBJECT_MODEL = reader.PRIMARY_MODEL  # the Phase 1 selected baseline

INTEGRITY_INTERPRETATION = [
    "A 'verified' result means the bytes on disk still hash to the values recorded "
    "at registration time. It says nothing about whether the evidence was correct "
    "in the first place.",
    "A 'changed' result is not by itself evidence of wrongdoing: re-running an audit "
    "script legitimately rewrites its outputs. It means this run's conclusions no "
    "longer describe the files currently on disk, so a new run should be registered.",
    "Checksums detect modification, not authorship. This is a local integrity check, "
    "not a signed or notarised provenance chain.",
    "File modification times are informational only and are excluded from the "
    "evidence digest, so touching a file without changing it does not break integrity.",
]


class RunNotFoundError(Exception):
    """Raised when a run id is not in the registry; mapped to HTTP 404."""

    def __init__(self, run_id: str, available: list[str]):
        self.run_id = run_id
        self.available = available
        super().__init__(
            f"Audit run '{run_id}' is not in the registry."
            + (f" Known runs: {', '.join(available)}." if available else " The registry is empty.")
        )


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def _run_id(evidence_digest: str, model_name: str, model_version: str) -> str:
    """
    Build the content-addressed run id.

    Includes the subject model identity as well as the evidence digest, so two
    runs over the same artefact set but a different subject model are distinct.
    """
    composite = hashlib.sha256(
        f"{DATASET_VERSION}|{model_name}|{model_version}|{evidence_digest}".encode("utf-8")
    ).hexdigest()
    return f"run-{composite[:16]}"


def _model_version(artifacts: list[ArtifactRecord], model_name: str) -> tuple[str, str]:
    """
    Derive a content-based model version from the fitted pipeline's checksum.

    Returns ``(model_version, model_run_identifier)``. Using the artefact's own
    SHA-256 means the version changes if and only if the serialised model changes
    -- no manual version bumping to forget.
    """
    target = f"models/{model_name}_pipeline.joblib"
    match = next((a for a in artifacts if a.path == target), None)
    if match is None:
        return ("unavailable", f"{model_name}:artifact-missing")
    version = f"sha256:{match.sha256[:16]}"
    run_identifier = (
        f"{model_name}@{version} · split=stratified 80/20 · random_state=42 · "
        f"threshold=0.5"
    )
    return (version, run_identifier)


# --------------------------------------------------------------------------- #
# Evidence collection (read-only)
# --------------------------------------------------------------------------- #
def _performance_summary(model_name: str) -> dict[str, Any]:
    """Quote the API's performance payload verbatim. No metric is recomputed."""
    perf = PerformanceResponse(**svc.build_performance(model_name)).model_dump()
    metrics = perf.get("metrics") or {}
    cm = perf.get("confusion_matrix") or {}
    return {
        "model_name": model_name,
        "n_test": perf.get("n_test"),
        "decision_threshold": perf.get("decision_threshold"),
        "positive_class": perf.get("positive_class"),
        **{k: metrics.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        **{
            k: cm.get(k)
            for k in (
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            )
        },
    }


def _recorded_decision() -> tuple[dict[str, Any], list[str]]:
    """Copy the committed decision and its blocking risk ids."""
    decision = DecisionResponse(**svc.build_decision()).model_dump()
    recorded = {
        "research_use": str(decision.get("research_use")),
        "real_world_deployment": str(decision.get("real_world_deployment")),
        "headline": str(decision.get("headline")),
        "decision_date": decision.get("decision_date"),
        "source": "GET /api/governance/decision",
        # Stored explicitly rather than left to the response-model default, so the
        # disclaimer travels with the data: anyone opening the SQLite file directly
        # sees that the registry copied this decision and did not make it.
        "note": (
            "Copied from the committed decision record. The registry does not make "
            "or change decisions."
        ),
    }
    blocking = [str(r) for r in decision.get("blocking_risk_ids") or []]
    return recorded, blocking


def _coverage(artifacts: list[ArtifactRecord]) -> dict[str, Any]:
    """Determine which audit phases this evidence set covers."""
    paths = {a.path for a in artifacts}
    groups = {a.group for a in artifacts}

    models_evaluated = reader.evaluated_models()
    try:
        sensitive = sorted(
            {str(r["attribute"]) for r in reader.read_csv("fairness_by_group")}
        )
    except reader.ArtifactError:
        sensitive = []

    performance = "results/model_metrics.csv" in paths
    fairness = "results/fairness/fairness_metrics_by_group.csv" in paths
    explainability = "results/explainability/global_feature_importance.csv" in paths
    governance = {
        "results/governance/model_card.md",
        "results/governance/governance_risk_register.csv",
        "results/governance/governance_summary.md",
    } <= paths
    agents = len(orchestrator.AGENT_NAMES) > 0

    return {
        "performance": performance,
        "fairness": fairness,
        "explainability": explainability,
        "governance": governance,
        "agents": agents,
        "complete": all([performance, fairness, explainability, governance, agents]),
        "models_evaluated": models_evaluated,
        "models_with_explainability": list(reader.EXPLAINED_MODELS),
        "sensitive_attributes": sensitive,
        "agent_names": list(orchestrator.AGENT_NAMES),
        "artifact_groups": sorted(groups),
    }


# --------------------------------------------------------------------------- #
# Registration (idempotent)
# --------------------------------------------------------------------------- #
def register_run(
    db_path: str | Path | None = None,
    root: Path = PROJECT_ROOT,
    model_name: str = SUBJECT_MODEL,
) -> dict[str, Any]:
    """
    Create or refresh the current audit-run record from the existing evidence.

    Idempotent: unchanged evidence maps to the same content-addressed run id, so
    the existing row is refreshed rather than duplicated. Writes only to the
    registry database -- the evidence itself is opened read-only.
    """
    artifacts = integrity_mod.discover_artifacts(root)
    if not artifacts:
        raise FileNotFoundError(
            f"No audit artefacts found under {root}. Run the audit scripts first "
            "(python src/train.py, then the fairness and explainability audits)."
        )

    digest = integrity_mod.evidence_digest(artifacts)
    model_version, model_run_identifier = _model_version(artifacts, model_name)
    run_id = _run_id(digest, model_name, model_version)

    performance = _performance_summary(model_name)
    decision, blocking = _recorded_decision()
    coverage = _coverage(artifacts)

    resolved_db = registry_db.resolve_db_path(db_path)
    now = registry_db.utc_now()
    superseded: list[str] = []

    with registry_db.connect(resolved_db) as connection:
        existing = connection.execute(
            "SELECT run_id, created_at, refresh_count FROM audit_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        if existing is None:
            # New evidence state -> new run. Demote any other active run -- but only
            # among reference-case runs. Uploaded-model runs describe entirely
            # different evidence and are not competing versions of this one, so
            # re-registering the reference case must leave them exactly as they are.
            for row in connection.execute(
                "SELECT run_id FROM audit_runs "
                "WHERE status = ? AND run_type = ? AND run_id != ?",
                (registry_db.STATUS_ACTIVE, registry_db.RUN_TYPE_REFERENCE, run_id),
            ).fetchall():
                superseded.append(row["run_id"])
            for old in superseded:
                connection.execute(
                    "UPDATE audit_runs SET status = ? WHERE run_id = ?",
                    (registry_db.STATUS_SUPERSEDED, old),
                )
                registry_db.record_event(
                    connection,
                    old,
                    "status_changed",
                    f"Marked superseded by newly registered run {run_id}.",
                )

            connection.execute(
                """
                INSERT INTO audit_runs (
                    run_id, schema_version, created_at, refreshed_at, refresh_count,
                    dataset_name, dataset_version, dataset_context,
                    model_name, model_version, model_run_identifier,
                    evidence_digest, artifact_count,
                    performance_summary, governance_decision, blocking_risk_ids,
                    audit_coverage, status, run_type
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    registry_db.SCHEMA_VERSION,
                    now,
                    now,
                    DATASET_NAME,
                    DATASET_VERSION,
                    DATASET_CONTEXT,
                    model_name,
                    model_version,
                    model_run_identifier,
                    digest,
                    len(artifacts),
                    registry_db.dumps(performance),
                    registry_db.dumps(decision),
                    registry_db.dumps(blocking),
                    registry_db.dumps(coverage),
                    registry_db.STATUS_ACTIVE,
                    registry_db.RUN_TYPE_REFERENCE,
                ),
            )
            registry_db.record_event(
                connection,
                run_id,
                "run_registered",
                f"Registered {len(artifacts)} artefacts; evidence digest "
                f"{digest[:16]}…; subject model {model_name} {model_version}.",
            )
            action = "superseded_previous" if superseded else "created"
            created_at, refresh_count = now, 1
        else:
            # Same evidence -> refresh in place. created_at is preserved.
            created_at = existing["created_at"]
            refresh_count = int(existing["refresh_count"]) + 1
            connection.execute(
                """
                UPDATE audit_runs
                   SET refreshed_at = ?, refresh_count = ?, status = ?,
                       performance_summary = ?, governance_decision = ?,
                       blocking_risk_ids = ?, audit_coverage = ?,
                       artifact_count = ?, model_version = ?, model_run_identifier = ?
                 WHERE run_id = ?
                """,
                (
                    now,
                    refresh_count,
                    registry_db.STATUS_ACTIVE,
                    registry_db.dumps(performance),
                    registry_db.dumps(decision),
                    registry_db.dumps(blocking),
                    registry_db.dumps(coverage),
                    len(artifacts),
                    model_version,
                    model_run_identifier,
                    run_id,
                ),
            )
            registry_db.record_event(
                connection,
                run_id,
                "run_refreshed",
                f"Refresh #{refresh_count}: evidence unchanged (digest "
                f"{digest[:16]}…), record updated in place.",
            )
            action = "refreshed"

        # Replace the artefact manifest for this run. Safe because the manifest is
        # a function of the run id: identical evidence -> identical rows.
        connection.execute("DELETE FROM run_artifacts WHERE run_id = ?", (run_id,))
        connection.executemany(
            "INSERT INTO run_artifacts (run_id, artifact_group, path, sha256, "
            "size_bytes, modified_utc) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (run_id, a.group, a.path, a.sha256, a.size_bytes, a.modified_utc)
                for a in artifacts
            ],
        )
        connection.commit()

    return {
        "run_id": run_id,
        "run_type": registry_db.RUN_TYPE_REFERENCE,
        "action": action,
        "created_at": created_at,
        "refreshed_at": now,
        "refresh_count": refresh_count,
        "artifact_count": len(artifacts),
        "evidence_digest": digest,
        "superseded_run_ids": superseded,
        "database": _rel_db(resolved_db),
        "message": {
            "created": "New audit run registered from the current evidence.",
            "refreshed": "Evidence unchanged; existing run refreshed in place "
            "(no duplicate created).",
            "superseded_previous": "Evidence changed; new run registered and the "
            "previously active run marked superseded.",
        }[action],
    }


def _rel_db(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Registration of uploaded-model runs
# --------------------------------------------------------------------------- #
#: Governance decision text stored for every uploaded run. Fixed, not computed: no
#: measurement on an uploaded model can turn into a deployment authorisation, so
#: there is no code path that could vary this value.
UPLOADED_DEPLOYMENT_POSITION = (
    "blocked -- not authorised for real-world deployment by this platform"
)


def _uploaded_artifact_records(manifest: dict[str, Any]) -> list[ArtifactRecord]:
    """
    Build the artefact manifest for an uploaded run from what is on disk now.

    Sources, in order: the twelve artefacts listed in the run manifest, then
    ``manifest.json`` itself (which cannot record its own checksum, so the registry
    hashes it here), then the two upload source files named by the evidence manifest.
    Every checksum is recomputed from the bytes rather than copied from the run's own
    JSON -- the registry's baseline has to be an independent measurement, otherwise
    verifying it later would only confirm that a file agrees with itself.
    """
    from app.onboarding import runtime_store as store

    audit_run_id = str(manifest["audit_run_id"])
    directory = store.audit_dir(audit_run_id)

    records: list[ArtifactRecord] = []
    seen: set[str] = set()

    def add(path: Path, group: str) -> None:
        if not path.is_file():
            return
        record = store.checksum_record(path, group)
        if record["path"] in seen:
            return
        seen.add(record["path"])
        records.append(
            ArtifactRecord(
                group=record["group"],
                path=record["path"],
                sha256=record["sha256"],
                size_bytes=int(record["size_bytes"]),
                modified_utc=record["modified_utc"],
            )
        )

    for name in store.AUDIT_ARTIFACT_NAMES:
        add(directory / name, "audit_run")

    try:
        evidence = store.read_json(audit_run_id, "evidence_manifest.json")
    except store.AuditArtifactMissing:
        evidence = {}
    for entry in evidence.get("artifacts", []):
        if entry.get("group") != "upload_source" or not entry.get("path"):
            continue
        add(PROJECT_ROOT / str(entry["path"]), "upload_source")

    return records


def _uploaded_performance_summary(
    audit_run_id: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Quote the run's own ``performance.json``. Nothing is recomputed here."""
    from app.onboarding import runtime_store as store

    try:
        performance = store.read_json(audit_run_id, "performance.json")
    except store.AuditArtifactMissing:
        return {"available": False, "reason": "performance.json was not written."}

    cm = performance.get("confusion_matrix") or {}
    return {
        "model_name": str((manifest.get("model_metadata") or {}).get("model_name")),
        "n_test": performance.get("n_samples"),
        "decision_threshold": performance.get("decision_threshold"),
        "threshold_applied": performance.get("threshold_applied"),
        "positive_class": performance.get("positive_class"),
        **{
            key: performance.get(key)
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
        **{
            key: cm.get(key)
            for key in (
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            )
        },
        "roc_auc_unavailable_reason": performance.get("roc_auc_unavailable_reason"),
        "source": f"GET /api/onboarding/audits/{audit_run_id}/performance",
    }


def _uploaded_decision(
    manifest: dict[str, Any], governance: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """
    Record the run's governance state in the same shape as the reference decision.

    ``research_use`` carries the governance state (``review_required``,
    ``insufficient_evidence`` or ``blocked_by_policy``) so the listing endpoint can
    render both kinds of run through one code path. ``real_world_deployment`` is
    constant: none of those three states is an approval.
    """
    state = str(
        governance.get("governance_state")
        or manifest.get("overall_governance_state")
        or "unknown"
    )
    decision = {
        "research_use": state,
        "real_world_deployment": UPLOADED_DEPLOYMENT_POSITION,
        "headline": str(
            governance.get("state_meaning")
            or "Deterministic gate evidence for human governance review."
        ),
        "decision_date": manifest.get("created_at"),
        "source": f"GET /api/onboarding/audits/{manifest.get('audit_run_id')}/governance",
        "note": (
            "Derived by the Governance-as-Code policy engine from this run's own "
            "evidence. It is decision-support for a human reviewer: not a legal "
            "compliance finding, not proof of discrimination, not a causal claim and "
            "not a deployment authorisation."
        ),
    }
    blocking = [str(c) for c in governance.get("blocking_controls") or []]
    return decision, blocking


def register_uploaded_run(
    audit_run_id: str, db_path: str | Path | None = None
) -> str:
    """
    Index a completed uploaded-model audit run in the registry.

    Called by :func:`app.onboarding.audit_service.create_audit` once every artefact
    has been written. The registry row is an *index and integrity baseline* over
    evidence that already exists -- this function computes no metric and makes no
    decision, it copies what the run recorded and hashes the files it points at.

    Differences from :func:`register_run`, all deliberate:

    * The registry run id **is** the audit-run id, so one identifier addresses the
      run in ``runtime/``, in the registry, and in the Conformity Bundle.
    * Nothing is superseded. Two uploaded runs are two different submissions, not
      two versions of one, and neither is a newer view of the reference case.
    * Re-registering the same audit-run id refreshes the row in place, which is what
      makes the call safe to repeat after a re-evaluation.

    Returns the registry run id.
    """
    from app.onboarding import runtime_store as store

    if not store.audit_dir(audit_run_id).is_dir():
        raise store.AuditRunNotFound(audit_run_id, store.list_audit_ids())

    manifest = store.read_json(audit_run_id, "manifest.json")
    try:
        governance = store.read_json(audit_run_id, "governance_summary.json")
    except store.AuditArtifactMissing:
        governance = {}

    model_meta = manifest.get("model_metadata") or {}
    target = manifest.get("target_configuration") or {}
    try:
        dataset_meta = store.read_json(audit_run_id, "uploaded_dataset_metadata.json")
    except store.AuditArtifactMissing:
        dataset_meta = {}

    artifacts = _uploaded_artifact_records(manifest)
    digest = integrity_mod.evidence_digest(artifacts)

    model_checksum = str(manifest.get("model_checksum") or "")
    dataset_checksum = str(manifest.get("dataset_checksum") or "")
    model_name = str(model_meta.get("model_name") or "uploaded model")
    model_version = str(
        model_meta.get("model_version")
        or (f"sha256:{model_checksum[:16]}" if model_checksum else "unavailable")
    )
    model_run_identifier = (
        f"{model_name}@sha256:{model_checksum[:16]} · "
        f"target={target.get('target_column')} · "
        f"positive={target.get('positive_class')} · "
        f"threshold={manifest.get('decision_threshold')}"
    )

    dataset_name = str(
        dataset_meta.get("original_filename_label")
        or dataset_meta.get("stored_filename")
        or "uploaded dataset"
    )
    dataset_version = (
        f"sha256:{dataset_checksum[:16]}" if dataset_checksum else "unavailable"
    )
    sensitive = target.get("sensitive_columns") or []
    dataset_context = (
        f"User-uploaded labelled CSV: {dataset_meta.get('row_count', 'unknown')} rows, "
        f"{dataset_meta.get('column_count', 'unknown')} columns; target "
        f"`{target.get('target_column')}` with positive class "
        f"`{target.get('positive_class')}`; sensitive columns selected for fairness "
        f"reporting: {', '.join(sensitive) if sensitive else 'none'}. Provenance, "
        "labelling quality and representativeness are asserted by the uploader and "
        "have not been independently verified by this platform."
    )

    coverage = dict(manifest.get("audit_coverage") or {})
    coverage["complete"] = bool(coverage) and all(
        bool(value) for value in coverage.values()
    )
    performance = _uploaded_performance_summary(audit_run_id, manifest)
    decision, blocking = _uploaded_decision(manifest, governance)

    resolved_db = registry_db.resolve_db_path(db_path)
    now = registry_db.utc_now()

    with registry_db.connect(resolved_db) as connection:
        existing = connection.execute(
            "SELECT run_id, created_at, refresh_count FROM audit_runs WHERE run_id = ?",
            (audit_run_id,),
        ).fetchone()

        if existing is None:
            connection.execute(
                """
                INSERT INTO audit_runs (
                    run_id, schema_version, created_at, refreshed_at, refresh_count,
                    dataset_name, dataset_version, dataset_context,
                    model_name, model_version, model_run_identifier,
                    evidence_digest, artifact_count,
                    performance_summary, governance_decision, blocking_risk_ids,
                    audit_coverage, status, run_type
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_run_id,
                    registry_db.SCHEMA_VERSION,
                    str(manifest.get("created_at") or now),
                    now,
                    dataset_name,
                    dataset_version,
                    dataset_context,
                    model_name,
                    model_version,
                    model_run_identifier,
                    digest,
                    len(artifacts),
                    registry_db.dumps(performance),
                    registry_db.dumps(decision),
                    registry_db.dumps(blocking),
                    registry_db.dumps(coverage),
                    registry_db.STATUS_ACTIVE,
                    registry_db.RUN_TYPE_UPLOADED,
                ),
            )
            registry_db.record_event(
                connection,
                audit_run_id,
                "uploaded_run_registered",
                f"Indexed uploaded-model audit run with {len(artifacts)} artefacts; "
                f"evidence digest {digest[:16]}…; governance state "
                f"{decision['research_use']}. No existing run was superseded and the "
                "Adult Income reference case is unaffected.",
            )
        else:
            connection.execute(
                """
                UPDATE audit_runs
                   SET refreshed_at = ?, refresh_count = ?, status = ?, run_type = ?,
                       performance_summary = ?, governance_decision = ?,
                       blocking_risk_ids = ?, audit_coverage = ?,
                       artifact_count = ?, evidence_digest = ?
                 WHERE run_id = ?
                """,
                (
                    now,
                    int(existing["refresh_count"]) + 1,
                    registry_db.STATUS_ACTIVE,
                    registry_db.RUN_TYPE_UPLOADED,
                    registry_db.dumps(performance),
                    registry_db.dumps(decision),
                    registry_db.dumps(blocking),
                    registry_db.dumps(coverage),
                    len(artifacts),
                    digest,
                    audit_run_id,
                ),
            )
            registry_db.record_event(
                connection,
                audit_run_id,
                "uploaded_run_refreshed",
                f"Re-indexed after re-evaluation: {len(artifacts)} artefacts, evidence "
                f"digest {digest[:16]}…, governance state {decision['research_use']}.",
            )

        connection.execute(
            "DELETE FROM run_artifacts WHERE run_id = ?", (audit_run_id,)
        )
        connection.executemany(
            "INSERT INTO run_artifacts (run_id, artifact_group, path, sha256, "
            "size_bytes, modified_utc) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    audit_run_id,
                    a.group,
                    a.path,
                    a.sha256,
                    a.size_bytes,
                    a.modified_utc,
                )
                for a in artifacts
            ],
        )
        connection.commit()

    return audit_run_id


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def _row_artifacts(connection: sqlite3.Connection, run_id: str) -> list[ArtifactRecord]:
    rows = connection.execute(
        "SELECT artifact_group, path, sha256, size_bytes, modified_utc "
        "FROM run_artifacts WHERE run_id = ? ORDER BY artifact_group, path",
        (run_id,),
    ).fetchall()
    return [
        ArtifactRecord(
            group=r["artifact_group"],
            path=r["path"],
            sha256=r["sha256"],
            size_bytes=int(r["size_bytes"]),
            modified_utc=r["modified_utc"],
        )
        for r in rows
    ]


def _known_run_ids(connection: sqlite3.Connection) -> list[str]:
    return [
        r["run_id"]
        for r in connection.execute("SELECT run_id FROM audit_runs ORDER BY created_at")
    ]


def list_runs(
    db_path: str | Path | None = None,
    status: str | None = None,
    run_type: str | None = None,
) -> dict[str, Any]:
    """
    List registered runs, newest first.

    ``active_run_id`` deliberately means *the active reference-case run*. It is what
    the existing dashboard and API callers use to reach the Adult Income audit, so it
    is scoped by run type: registering any number of uploaded audits must never move
    it. Uploaded runs are reported separately in ``uploaded_run_ids``.
    """
    resolved = registry_db.resolve_db_path(db_path)
    with registry_db.connect(resolved, create=False) as connection:
        query = "SELECT * FROM audit_runs"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if run_type:
            clauses.append("run_type = ?")
            params.append(run_type)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, run_id"
        rows = connection.execute(query, tuple(params)).fetchall()

        active = connection.execute(
            "SELECT run_id FROM audit_runs WHERE status = ? AND run_type = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (registry_db.STATUS_ACTIVE, registry_db.RUN_TYPE_REFERENCE),
        ).fetchone()
        uploaded = [
            r["run_id"]
            for r in connection.execute(
                "SELECT run_id FROM audit_runs WHERE run_type = ? "
                "ORDER BY created_at DESC, run_id",
                (registry_db.RUN_TYPE_UPLOADED,),
            ).fetchall()
        ]

    runs = []
    for row in rows:
        decision = registry_db.loads(row["governance_decision"]) or {}
        coverage = registry_db.loads(row["audit_coverage"]) or {}
        runs.append(
            {
                "run_id": row["run_id"],
                "run_type": row["run_type"],
                "status": row["status"],
                "created_at": row["created_at"],
                "refreshed_at": row["refreshed_at"],
                "refresh_count": int(row["refresh_count"]),
                "dataset_name": row["dataset_name"],
                "dataset_version": row["dataset_version"],
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "evidence_digest": row["evidence_digest"],
                "artifact_count": int(row["artifact_count"]),
                "coverage_complete": bool(coverage.get("complete")),
                "research_use": str(decision.get("research_use")),
                "real_world_deployment": str(decision.get("real_world_deployment")),
            }
        )

    return {
        "count": len(runs),
        "active_run_id": active["run_id"] if active else None,
        "active_reference_run_id": active["run_id"] if active else None,
        "uploaded_run_ids": uploaded,
        "database": _rel_db(resolved),
        "filters_applied": {"status": status, "run_type": run_type},
        "run_type_note": (
            "'reference_case' runs describe the built-in Adult Income audit. "
            "'uploaded_model' runs describe user submissions stored under runtime/. "
            "The two are never versions of each other and neither supersedes the other."
        ),
        "runs": runs,
    }


def get_run(run_id: str, db_path: str | Path | None = None) -> dict[str, Any]:
    """Full detail for one run, including its artefact manifest."""
    resolved = registry_db.resolve_db_path(db_path)
    with registry_db.connect(resolved, create=False) as connection:
        row = connection.execute(
            "SELECT * FROM audit_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id, _known_run_ids(connection))
        artifacts = _row_artifacts(connection, run_id)

    return {
        "run_id": row["run_id"],
        "run_type": row["run_type"],
        "schema_version": int(row["schema_version"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "refreshed_at": row["refreshed_at"],
        "refresh_count": int(row["refresh_count"]),
        "dataset_name": row["dataset_name"],
        "dataset_version": row["dataset_version"],
        "dataset_context": row["dataset_context"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "model_run_identifier": row["model_run_identifier"],
        "evidence_digest": row["evidence_digest"],
        "artifact_count": int(row["artifact_count"]),
        "performance_summary": registry_db.loads(row["performance_summary"]),
        "governance_decision": registry_db.loads(row["governance_decision"]),
        "blocking_risk_ids": registry_db.loads(row["blocking_risk_ids"]) or [],
        "audit_coverage": registry_db.loads(row["audit_coverage"]),
        "artifacts": [
            {
                "group": a.group,
                "path": a.path,
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
                "modified_utc": a.modified_utc,
            }
            for a in artifacts
        ],
    }


def check_integrity(
    run_id: str, db_path: str | Path | None = None, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """
    Recompute every registered checksum and report verified / missing / changed.

    Records an ``integrity_checked`` event so checks appear on the timeline. This
    is the one read path that writes -- to the registry's own event log only.
    """
    resolved = registry_db.resolve_db_path(db_path)
    with registry_db.connect(resolved, create=False) as connection:
        row = connection.execute(
            "SELECT run_id, status, evidence_digest FROM audit_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id, _known_run_ids(connection))
        registered = _row_artifacts(connection, run_id)

    results = integrity_mod.verify_artifacts(registered, root=root)
    summary = integrity_mod.summarise_verification(results)

    # Digest over the registered artefact set as it stands now. Missing files are
    # skipped, so an incomplete set also shifts the digest -- as it should.
    present = [
        ArtifactRecord(
            group=r.group,
            path=r.path,
            sha256=r.actual_sha256 or "",
            size_bytes=r.actual_size_bytes or 0,
            modified_utc="",
        )
        for r in results
        if r.status != "missing"
    ]
    current_digest = integrity_mod.evidence_digest(present)
    checked_at = registry_db.utc_now()

    with registry_db.connect(resolved) as connection:
        registry_db.record_event(
            connection,
            run_id,
            "integrity_checked",
            f"{summary['integrity_status']}: {summary['verified_count']} verified, "
            f"{summary['changed_count']} changed, {summary['missing_count']} missing "
            f"of {summary['artifacts_checked']} artefacts.",
        )
        connection.commit()

    return {
        "run_id": run_id,
        "run_status": row["status"],
        "checked_at": checked_at,
        **summary,
        "registered_evidence_digest": row["evidence_digest"],
        "current_evidence_digest": current_digest,
        "interpretation": INTEGRITY_INTERPRETATION,
        "artifacts": [
            {
                "group": r.group,
                "path": r.path,
                "status": r.status,
                "expected_sha256": r.expected_sha256,
                "actual_sha256": r.actual_sha256,
                "expected_size_bytes": r.expected_size_bytes,
                "actual_size_bytes": r.actual_size_bytes,
                "detail": r.detail,
            }
            for r in results
        ],
    }


def get_timeline(
    run_id: str, db_path: str | Path | None = None, limit: int = 200
) -> dict[str, Any]:
    """
    Chronological history for a run.

    Combines two sources: the append-only registry event log, and 'evidence'
    events derived from the modification time of the newest artefact in each audit
    phase -- which is when that phase last wrote its outputs.

    The event log is append-only by design, so repeated integrity checks accumulate.
    ``limit`` returns the most recent ``limit`` events; ``total_events`` and
    ``truncated`` always report the full count, so a window is never a silent cap.
    """
    resolved = registry_db.resolve_db_path(db_path)
    with registry_db.connect(resolved, create=False) as connection:
        row = connection.execute(
            "SELECT run_id, status FROM audit_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id, _known_run_ids(connection))
        artifacts = _row_artifacts(connection, run_id)
        event_rows = connection.execute(
            "SELECT event_time, event_type, detail FROM registry_events "
            "WHERE run_id = ? ORDER BY event_id",
            (run_id,),
        ).fetchall()

    events: list[dict[str, Any]] = []

    # Evidence phases, newest artefact per group = when that phase last wrote.
    by_group: dict[str, list[ArtifactRecord]] = {}
    for artifact in artifacts:
        by_group.setdefault(artifact.group, []).append(artifact)
    for group, records in by_group.items():
        newest = max(records, key=lambda r: r.modified_utc)
        events.append(
            {
                "event_time": newest.modified_utc,
                "event_type": f"evidence_produced:{group}",
                "source": "evidence",
                "detail": (
                    f"{len(records)} {group} artefact(s); most recently written "
                    f"{newest.path}."
                ),
            }
        )

    for event in event_rows:
        events.append(
            {
                "event_time": event["event_time"],
                "event_type": event["event_type"],
                "source": "registry",
                "detail": event["detail"],
            }
        )

    events.sort(key=lambda e: (e["event_time"], e["event_type"]))
    total = len(events)
    windowed = events[-limit:] if limit and total > limit else events
    return {
        "run_id": run_id,
        "run_status": row["status"],
        "count": len(windowed),
        "total_events": total,
        "truncated": total > len(windowed),
        "events": windowed,
    }


def registry_stats(db_path: str | Path | None = None) -> dict[str, Any]:
    """Counts by status; reports ``initialised: false`` instead of raising."""
    resolved = registry_db.resolve_db_path(db_path)
    if not resolved.exists():
        return {
            "total_runs": 0,
            "active": 0,
            "superseded": 0,
            "archived": 0,
            "database": _rel_db(resolved),
            "by_run_type": {run_type: 0 for run_type in registry_db.VALID_RUN_TYPES},
            "initialised": False,
        }
    with registry_db.connect(resolved, create=False) as connection:
        counts = {
            status: connection.execute(
                "SELECT COUNT(*) AS n FROM audit_runs WHERE status = ?", (status,)
            ).fetchone()["n"]
            for status in registry_db.VALID_STATUSES
        }
        by_run_type = {
            run_type: int(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM audit_runs WHERE run_type = ?",
                    (run_type,),
                ).fetchone()["n"]
            )
            for run_type in registry_db.VALID_RUN_TYPES
        }
        total = connection.execute("SELECT COUNT(*) AS n FROM audit_runs").fetchone()["n"]
    return {
        "total_runs": int(total),
        "active": int(counts[registry_db.STATUS_ACTIVE]),
        "superseded": int(counts[registry_db.STATUS_SUPERSEDED]),
        "archived": int(counts[registry_db.STATUS_ARCHIVED]),
        "by_run_type": by_run_type,
        "database": _rel_db(resolved),
        "initialised": True,
    }
