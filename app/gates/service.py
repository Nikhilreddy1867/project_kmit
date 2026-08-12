"""
app/gates/service.py
====================
Orchestration for Governance-as-Code: load a run's evidence, evaluate the policy,
write the four governance artefacts, and manage the waiver register.

Division of responsibility
--------------------------
:mod:`app.gates.policy_engine` decides; this module reads, writes and stores. Keeping
the decision logic free of I/O is what makes it testable as a pure function of
(evidence, policy, waivers) and what makes re-evaluation a genuine integrity check.

Writes
------
Four artefacts, all inside the run's own ``runtime/audits/<audit_run_id>/``
directory: ``gate_evaluation.json``, ``governance_summary.json``,
``conformity_bundle.json``, ``traceability.json``. They are *generated* documents, so
:func:`evaluate_run` may legitimately recompute them -- which is exactly what
``POST /api/gates/runs/{audit_run_id}/evaluate`` does. Nothing here writes an upload,
and nothing here can produce a path outside ``runtime/``.

Waiver register
---------------
Waivers live in a table in the runtime SQLite database, created lazily by
:func:`ensure_waiver_table` rather than by editing :mod:`app.registry.db`'s schema
script -- so an existing registry database gains the table on first use without a
migration and without any change to how the reference case is stored.

Three properties the register enforces, because a waiver mechanism without them is
decorative:

* **No automatic creation or approval.** There is no code path that inserts a waiver
  other than :func:`create_waiver`, which requires a caller-supplied owner,
  rationale, expiry and at least one compensating control.
* **Time-bounded.** :func:`list_waivers` recomputes ``status`` against the clock on
  every read, so an expired waiver stops having an effect without anyone running a
  job. The stored row is never mutated to hide that it once applied.
* **The Release Gate cannot be waived.** RG-01 is marked ``waiver_eligible: false``
  in the policy, and :func:`create_waiver` refuses it outright rather than accepting
  a waiver that would silently never apply.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from app.gates import conformity_bundle, policy_engine
from app.gates.schemas import (
    ConformityBundle,
    GateEvaluation,
    PolicyProfile,
    TraceabilityMatrix,
    Waiver,
    WaiverIn,
)
from app.onboarding import runtime_store as store
from app.registry import db as registry_db

_WAIVER_SCHEMA = """
CREATE TABLE IF NOT EXISTS governance_waivers (
    waiver_id             TEXT PRIMARY KEY,
    audit_run_id          TEXT NOT NULL,
    control_id            TEXT NOT NULL,
    gate                  TEXT,
    scope                 TEXT NOT NULL,
    owner                 TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    expires_at            TEXT NOT NULL,
    rationale             TEXT NOT NULL,
    compensating_controls TEXT NOT NULL,
    revoked_at            TEXT
);

CREATE INDEX IF NOT EXISTS idx_waivers_run
    ON governance_waivers (audit_run_id, control_id);
"""


class WaiverRejected(Exception):
    """A waiver was refused because it would not be an accountable decision."""

    def __init__(self, code: str, message: str, hint: str | None = None):
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, treating a bare datetime as UTC."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Waiver register
# --------------------------------------------------------------------------- #
def ensure_waiver_table(connection: sqlite3.Connection) -> None:
    """Create the waiver table if absent. Idempotent; no other schema is touched."""
    connection.executescript(_WAIVER_SCHEMA)
    connection.commit()


def _waiver_from_row(row: sqlite3.Row, now: datetime) -> Waiver:
    """Build a waiver, deriving ``status`` from the clock rather than from storage."""
    expires = _parse_iso(row["expires_at"])
    if row["revoked_at"]:
        status = "revoked"
    elif expires is None or expires <= now:
        status = "expired"
    else:
        status = "active"
    return Waiver(
        waiver_id=row["waiver_id"],
        audit_run_id=row["audit_run_id"],
        control_id=row["control_id"],
        gate=row["gate"],
        scope=row["scope"],
        owner=row["owner"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        rationale=row["rationale"],
        compensating_controls=registry_db.loads(row["compensating_controls"]) or [],
        status=status,  # type: ignore[arg-type]
    )


def list_waivers(audit_run_id: str, db_path: Any = None) -> list[Waiver]:
    """Every waiver ever recorded for a run, with status recomputed against now."""
    try:
        connection = registry_db.connect(db_path)
    except FileNotFoundError:
        return []
    now = datetime.now(timezone.utc)
    try:
        ensure_waiver_table(connection)
        rows = connection.execute(
            "SELECT * FROM governance_waivers WHERE audit_run_id = ? "
            "ORDER BY created_at, waiver_id",
            (audit_run_id,),
        ).fetchall()
    finally:
        connection.close()
    return [_waiver_from_row(row, now) for row in rows]


def create_waiver(
    audit_run_id: str, payload: WaiverIn, policy: PolicyProfile, db_path: Any = None
) -> Waiver:
    """
    Record an explicit, human-created waiver against one control.

    Every rejection below corresponds to a rule in the policy's ``waiver_rules``. The
    checks live here, at the only insertion point, so there is no way to reach the
    table without passing them.
    """
    controls = {c.control_id: c for c in policy.controls}
    control = controls.get(payload.control_id)
    if control is None:
        raise WaiverRejected(
            "unknown_control",
            f"'{payload.control_id}' is not a control in policy "
            f"{policy.policy_id} v{policy.policy_version}.",
            hint="Valid controls: " + ", ".join(sorted(controls)),
        )
    if not control.waiver_eligible:
        raise WaiverRejected(
            "control_not_waiver_eligible",
            f"{control.control_id} ({control.title}) is not waiver-eligible under this "
            "policy, so no waiver can be recorded against it.",
            hint="A waiver can never satisfy or override the Release Gate: release is "
            "a human accountability decision, not a risk that can be accepted by "
            "annotation.",
        )

    expires = _parse_iso(payload.expires_at)
    if expires is None:
        raise WaiverRejected(
            "invalid_expiry",
            f"'{payload.expires_at}' is not a valid ISO-8601 timestamp.",
        )
    if expires <= datetime.now(timezone.utc):
        raise WaiverRejected(
            "expiry_in_the_past",
            "The expiry must be in the future. A waiver that is already expired has no "
            "effect, and recording one would misrepresent the run's status.",
        )

    # Confirm the run exists before writing a waiver that could never apply.
    if not store.audit_dir(audit_run_id).is_dir():
        raise store.AuditRunNotFound(audit_run_id, store.list_audit_ids())

    waiver_id = f"waiver-{uuid.uuid4().hex[:12]}"
    created_at = _utc_now()
    connection = registry_db.connect(db_path)
    try:
        ensure_waiver_table(connection)
        connection.execute(
            "INSERT INTO governance_waivers (waiver_id, audit_run_id, control_id, "
            "gate, scope, owner, created_at, expires_at, rationale, "
            "compensating_controls, revoked_at) VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
            (
                waiver_id,
                audit_run_id,
                control.control_id,
                control.gate,
                payload.scope,
                payload.owner,
                created_at,
                payload.expires_at,
                payload.rationale,
                registry_db.dumps(payload.compensating_controls),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    return Waiver(
        waiver_id=waiver_id,
        audit_run_id=audit_run_id,
        control_id=control.control_id,
        gate=control.gate,
        scope=payload.scope,
        owner=payload.owner,
        created_at=created_at,
        expires_at=payload.expires_at,
        rationale=payload.rationale,
        compensating_controls=list(payload.compensating_controls),
        status="active",
    )


def revoke_waiver(audit_run_id: str, waiver_id: str, db_path: Any = None) -> Waiver:
    """
    Mark a waiver revoked.

    The row is retained with a ``revoked_at`` stamp rather than deleted: a waiver that
    once applied is part of the run's history, and erasing it would make the timeline
    a description of the present rather than a record.
    """
    connection = registry_db.connect(db_path)
    try:
        ensure_waiver_table(connection)
        cursor = connection.execute(
            "UPDATE governance_waivers SET revoked_at = ? "
            "WHERE waiver_id = ? AND audit_run_id = ? AND revoked_at IS NULL",
            (_utc_now(), waiver_id, audit_run_id),
        )
        connection.commit()
        if cursor.rowcount == 0:
            raise WaiverRejected(
                "waiver_not_found",
                f"No active waiver '{waiver_id}' exists for run '{audit_run_id}'.",
            )
        row = connection.execute(
            "SELECT * FROM governance_waivers WHERE waiver_id = ?", (waiver_id,)
        ).fetchone()
    finally:
        connection.close()
    return _waiver_from_row(row, datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# Evidence loading
# --------------------------------------------------------------------------- #
#: Evidence documents the policy engine reads, and whether the engine can proceed
#: without them. A missing optional document yields NOT_EVALUATED for its control
#: rather than an error, which is why the load is tolerant.
_CONTEXT_ARTIFACTS: dict[str, str] = {
    "model_metadata": "uploaded_model_metadata.json",
    "dataset_metadata": "uploaded_dataset_metadata.json",
    "performance": "performance.json",
    "fairness": "fairness.json",
    "explainability": "explainability.json",
    "risk_summary": "risk_summary.json",
    "evidence_manifest": "evidence_manifest.json",
}


def load_context(audit_run_id: str) -> dict[str, Any]:
    """
    Read a run's evidence documents into the dict the policy engine consumes.

    A missing document becomes ``None`` instead of raising, so a partially-written
    run is reported control-by-control as unevaluated rather than failing wholesale
    with a 500. The run directory itself must exist -- that is a genuine 404.
    """
    if not store.audit_dir(audit_run_id).is_dir():
        raise store.AuditRunNotFound(audit_run_id, store.list_audit_ids())

    context: dict[str, Any] = {}
    for key, name in _CONTEXT_ARTIFACTS.items():
        try:
            context[key] = store.read_json(audit_run_id, name)
        except store.AuditArtifactMissing:
            context[key] = None
    try:
        context["governance_summary"] = store.read_json(
            audit_run_id, "governance_summary.json"
        )
    except store.AuditArtifactMissing:
        context["governance_summary"] = None
    return context


# --------------------------------------------------------------------------- #
# Governance summary
# --------------------------------------------------------------------------- #
GOVERNANCE_LIMITATIONS: tuple[str, ...] = (
    "This is an academic prototype producing deterministic decision-support evidence. "
    "It is not a certified EU AI Act, NIST AI RMF or ISO/IEC 42001 compliance system.",
    "No governance state produced for an uploaded model authorises deployment. Human "
    "review is always required.",
    "Nothing here establishes legal compliance or non-compliance, proves "
    "discrimination, or demonstrates causation.",
    "All metrics describe the single uploaded dataset at the configured threshold. "
    "They do not establish that behaviour transfers to any other population.",
    "Evidence integrity is SHA-256 change detection. No digital signature scheme is "
    "implemented.",
    "The built-in Adult Income reference case has its own separate, unchanged "
    "governance decision, reached before this run existed and unaffected by it.",
)


def build_governance_summary(
    audit_run_id: str,
    evaluation: GateEvaluation,
    context: dict[str, Any],
    audit_coverage: dict[str, bool],
) -> dict[str, Any]:
    """
    Fold the gate evaluation and the risk summary into one governance record.

    The state itself comes from :func:`policy_engine.derive_governance_state`, so the
    summary cannot disagree with the gates it is derived from.
    """
    state, meaning, grounds = policy_engine.derive_governance_state(evaluation)
    risk = context.get("risk_summary") or {}
    unavailable = [name for name, available in audit_coverage.items() if not available]

    return {
        "audit_run_id": audit_run_id,
        "governance_state": state,
        "state_meaning": meaning,
        "state_grounds": grounds,
        "human_review_required": True,
        "deployment_authorisation": "not_granted",
        "policy_profile_id": evaluation.policy_profile_id,
        "policy_version": evaluation.policy_version,
        "gate_summary": evaluation.gate_summary,
        "blocking_controls": evaluation.blocking_controls,
        "evidence_coverage_score": evaluation.evidence_coverage_score,
        "control_coverage_score": evaluation.control_coverage_score,
        "coverage_metric_caveat": evaluation.coverage_metric_caveat,
        "audit_coverage": audit_coverage,
        "unavailable_capabilities": unavailable,
        "risks": risk.get("risks", []),
        "severity_counts": risk.get("severity_counts", {}),
        "reference_case_note": (
            "The built-in Adult Income reference case has its own separate, unchanged "
            "governance decision (research/education only = conditionally approved; "
            "real-world deployment = blocked). Nothing in this run affects it."
        ),
        "limitations": list(GOVERNANCE_LIMITATIONS),
        "notice": (
            "Deterministic decision-support evidence for human governance review. Not "
            "a legal compliance assessment, not a finding of discrimination, not a "
            "causal claim, and not an authorisation to deploy."
        ),
    }


# --------------------------------------------------------------------------- #
# Evaluation + artefact writing
# --------------------------------------------------------------------------- #
GATE_ARTIFACTS: tuple[str, ...] = (
    "gate_evaluation.json",
    "governance_summary.json",
    "conformity_bundle.json",
    "traceability.json",
)


def refresh_run_manifest(
    audit_run_id: str,
    evaluation: GateEvaluation,
    bundle: ConformityBundle,
    governance: dict[str, Any],
    evaluated_at: str,
) -> None:
    """
    Bring ``manifest.json`` back into agreement with a re-evaluated run.

    Called only when the manifest already exists, i.e. on re-evaluation --
    :func:`app.onboarding.audit_service.create_audit` writes the manifest itself,
    after the first evaluation.

    Without this, a legitimate re-evaluation would leave the run's index describing a
    verdict that no longer exists: the list endpoint would report the superseded gate
    summary, and the integrity check would call four freshly written files "modified"
    because their recorded checksums belonged to the previous evaluation.

    What is refreshed is deliberately narrow -- **only the four artefacts this
    evaluation rewrote**, plus the verdict fields that describe them. The measurement
    artefacts' baselines are left exactly as they were, and ``evidence_manifest.json``
    is never rewritten at all. That separation is what keeps tamper detection real: a
    re-evaluation cannot bless an edited ``performance.json``, because the seal that
    covers it is not touched here and takes precedence when the two are compared.
    """
    try:
        manifest = store.read_json(audit_run_id, "manifest.json")
    except (store.AuditArtifactMissing, store.AuditRunNotFound):
        return

    refreshed = {
        name: store.checksum_record(store.artifact_path(audit_run_id, name), "audit_run")
        for name in GATE_ARTIFACTS
        if store.artifact_path(audit_run_id, name).is_file()
    }
    by_path = {record["path"]: record for record in refreshed.values()}
    manifest["generated_artifacts"] = [
        by_path.get(entry.get("path"), entry)
        for entry in (manifest.get("generated_artifacts") or [])
    ]

    manifest["gate_summary"] = evaluation.gate_summary
    manifest["overall_governance_state"] = governance["governance_state"]
    manifest["conformity_bundle_id"] = bundle.bundle_id
    manifest["policy_profile_id"] = evaluation.policy_profile_id
    manifest["policy_version"] = evaluation.policy_version
    manifest["policy_checksum"] = evaluation.policy_checksum
    # created_at is never touched: the run was created once, and re-evaluating its
    # evidence does not make it a new run.
    manifest["last_evaluated_at"] = evaluated_at
    manifest["evaluation_count"] = int(manifest.get("evaluation_count") or 1) + 1
    manifest["re_evaluation_note"] = (
        "The policy gates were re-evaluated over this run's existing evidence. No "
        "measurement was recomputed, the model was not re-run, and "
        "evidence_manifest.json -- the sealed baseline for the measurement artefacts -- "
        "was not modified."
    )
    store.write_json(
        store.artifact_path(audit_run_id, "manifest.json"), manifest, allow_replace=True
    )


def evaluate_run(
    audit_run_id: str,
    *,
    policy_id: str | None = None,
    allow_replace: bool = False,
    db_path: Any = None,
    audit_coverage: dict[str, bool] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate the policy for one run and write the four governance artefacts.

    Called twice in a run's life: once by
    :func:`app.onboarding.audit_service.create_audit` with ``allow_replace=False``
    (the artefacts must not already exist), and again by the re-evaluate endpoint with
    ``allow_replace=True``.

    Returns the evaluation, bundle, traceability and governance summary, plus
    ``changed`` -- ``False`` when the recomputation reproduced the stored result
    exactly, which is the expected outcome for unchanged evidence and is the practical
    proof that the engine is deterministic.
    """
    context = load_context(audit_run_id)
    metadata = context.get("model_metadata") or {}
    # The run records which policy it was audited against, so a re-evaluation reuses
    # that policy by default rather than silently switching to whatever the current
    # default happens to be.
    policy = policy_engine.load_policy(policy_id or metadata.get("policy_profile_id"))
    coverage = audit_coverage or dict(metadata.get("audit_coverage") or {})

    previous_summary = (context.get("governance_summary") or {}).get("gate_summary")
    previous_bundle: str | None = None
    try:
        previous_bundle = str(
            store.read_json(audit_run_id, "conformity_bundle.json").get("bundle_id")
        )
    except store.AuditArtifactMissing:
        previous_bundle = None

    timestamp = now or _utc_now()
    waivers = list_waivers(audit_run_id, db_path)
    evaluation = policy_engine.evaluate(
        audit_run_id, context, policy, waivers=waivers, evaluated_at=timestamp
    )

    store.write_json(
        store.artifact_path(audit_run_id, "gate_evaluation.json"),
        evaluation.model_dump(mode="json"),
        allow_replace=allow_replace,
    )

    governance = build_governance_summary(audit_run_id, evaluation, context, coverage)
    store.write_json(
        store.artifact_path(audit_run_id, "governance_summary.json"),
        governance,
        allow_replace=allow_replace,
    )

    # The bundle embeds the governance summary, so it is built from a context that
    # includes the version just written rather than the stale one loaded above.
    context["governance_summary"] = governance
    bundle = conformity_bundle.build_bundle(
        audit_run_id, evaluation, policy, context, timestamp
    )
    store.write_json(
        store.artifact_path(audit_run_id, "conformity_bundle.json"),
        bundle.model_dump(mode="json"),
        allow_replace=allow_replace,
    )

    traceability = conformity_bundle.build_traceability(
        audit_run_id, evaluation, policy, timestamp
    )
    store.write_json(
        store.artifact_path(audit_run_id, "traceability.json"),
        traceability.model_dump(mode="json"),
        allow_replace=allow_replace,
    )

    # On a re-evaluation the run manifest is now describing the previous verdict, so
    # bring it back into agreement. Skipped on first evaluation, where create_audit
    # writes the manifest immediately after this returns.
    refresh_run_manifest(audit_run_id, evaluation, bundle, governance, timestamp)

    changed = (
        previous_summary is not None
        and previous_bundle is not None
        and (
            previous_summary != evaluation.gate_summary
            or previous_bundle != bundle.bundle_id
        )
    )

    return {
        "policy": policy,
        "evaluation": evaluation,
        "governance": governance,
        "bundle": bundle,
        "traceability": traceability,
        "changed": bool(changed),
        "first_evaluation": previous_bundle is None,
        "artifacts_rewritten": list(GATE_ARTIFACTS),
        "evaluated_at": timestamp,
    }


# --------------------------------------------------------------------------- #
# Read accessors
# --------------------------------------------------------------------------- #
def get_evaluation(audit_run_id: str) -> GateEvaluation:
    return GateEvaluation(**store.read_json(audit_run_id, "gate_evaluation.json"))


def get_bundle(audit_run_id: str) -> ConformityBundle:
    return ConformityBundle(**store.read_json(audit_run_id, "conformity_bundle.json"))


def get_traceability(audit_run_id: str) -> TraceabilityMatrix:
    return TraceabilityMatrix(**store.read_json(audit_run_id, "traceability.json"))
