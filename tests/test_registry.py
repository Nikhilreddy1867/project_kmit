"""
tests/test_registry.py
======================
Tests for the Phase 8 governance audit registry and evidence-integrity layer.

Safety
------
No test writes to ``data/``, ``models/``, ``predictions/`` or ``results/``. The
deliberate integrity-mismatch tests operate on **temporary copies** of artefacts in
``tmp_path`` (real files, real SHA-256 recomputation) or on a temporary registry
database -- never on the repository's evidence. A final test asserts the real
artefacts are byte-identical after the whole module has run.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import db as registry_db
from app.registry import integrity as integrity_mod
from app.registry import service

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

WATCHED = [
    RESULTS / "model_metrics.csv",
    RESULTS / "fairness" / "fairness_metrics_by_group.csv",
    RESULTS / "explainability" / "global_feature_importance.csv",
    RESULTS / "governance" / "governance_risk_register.csv",
    RESULTS / "governance" / "governance_summary.md",
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """A registry database isolated to this test."""
    return tmp_path / "registry.db"


def _get(client: TestClient, path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}"
    return response.json()


# --------------------------------------------------------------------------- #
# Registration and idempotency
# --------------------------------------------------------------------------- #
def test_register_creates_a_run(temp_db: Path) -> None:
    result = service.register_run(db_path=temp_db)
    assert result["action"] == "created"
    assert result["run_id"].startswith("run-")
    assert result["artifact_count"] > 0
    assert len(result["evidence_digest"]) == 64  # full SHA-256 hex
    assert result["refresh_count"] == 1
    assert temp_db.exists()


def test_register_is_idempotent(temp_db: Path) -> None:
    """Re-registering unchanged evidence refreshes in place; no duplicate run."""
    first = service.register_run(db_path=temp_db)
    second = service.register_run(db_path=temp_db)
    third = service.register_run(db_path=temp_db)

    # Same content-addressed id every time.
    assert first["run_id"] == second["run_id"] == third["run_id"]
    assert first["evidence_digest"] == third["evidence_digest"]

    # created_at is preserved; refresh bookkeeping advances.
    assert second["action"] == "refreshed" and third["action"] == "refreshed"
    assert second["created_at"] == first["created_at"] == third["created_at"]
    assert (first["refresh_count"], second["refresh_count"], third["refresh_count"]) == (1, 2, 3)

    # Exactly one row, still active.
    listing = service.list_runs(db_path=temp_db)
    assert listing["count"] == 1
    assert listing["runs"][0]["status"] == registry_db.STATUS_ACTIVE
    assert listing["active_run_id"] == first["run_id"]


def test_register_records_required_fields(temp_db: Path) -> None:
    run_id = service.register_run(db_path=temp_db)["run_id"]
    run = service.get_run(run_id, db_path=temp_db)

    assert run["dataset_name"] == service.DATASET_NAME
    assert run["dataset_version"] and run["dataset_context"]
    assert run["model_name"] == "xgboost"
    assert run["model_version"].startswith("sha256:")  # content-based model version
    assert "random_state=42" in run["model_run_identifier"]
    assert run["created_at"] and run["refreshed_at"]
    assert run["status"] in registry_db.VALID_STATUSES

    # Performance summary is quoted, not recomputed.
    perf = run["performance_summary"]
    assert perf["model_name"] == "xgboost"
    for key in ("accuracy", "f1", "roc_auc", "false_negatives", "n_test"):
        assert perf[key] is not None

    # Coverage across all five phases.
    coverage = run["audit_coverage"]
    for phase in ("performance", "fairness", "explainability", "governance", "agents"):
        assert coverage[phase] is True, f"{phase} not covered"
    assert coverage["complete"] is True

    # Every artefact carries a full SHA-256 and a source path.
    assert run["artifact_count"] == len(run["artifacts"]) > 0
    for artifact in run["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["path"] and artifact["group"]
        assert artifact["size_bytes"] >= 0


def test_registered_decision_matches_the_committed_record(
    client: TestClient, temp_db: Path
) -> None:
    """The registry copies the decision; it must not alter it."""
    decision = _get(client, "/api/governance/decision")
    run_id = service.register_run(db_path=temp_db)["run_id"]
    run = service.get_run(run_id, db_path=temp_db)

    recorded = run["governance_decision"]
    assert recorded["research_use"] == decision["research_use"] == "conditionally_approved"
    assert recorded["real_world_deployment"] == decision["real_world_deployment"] == "blocked"
    assert recorded["headline"] == decision["headline"]
    assert run["blocking_risk_ids"] == decision["blocking_risk_ids"]
    assert "does not make or change decisions" in recorded["note"]


def test_changed_evidence_creates_new_run_and_supersedes_previous(
    temp_db: Path, tmp_path: Path
) -> None:
    """
    A different evidence digest must produce a NEW run and demote the old one.

    Simulated by rewriting the stored digest of the first run in the temporary
    database, so re-registering computes an id that does not match it. The real
    artefacts are untouched.
    """
    first = service.register_run(db_path=temp_db)

    with sqlite3.connect(temp_db) as connection:
        connection.execute(
            "UPDATE audit_runs SET run_id = ?, evidence_digest = ? WHERE run_id = ?",
            ("run-stale0000000000", "0" * 64, first["run_id"]),
        )
        connection.execute(
            "UPDATE run_artifacts SET run_id = ? WHERE run_id = ?",
            ("run-stale0000000000", first["run_id"]),
        )
        connection.commit()

    second = service.register_run(db_path=temp_db)
    assert second["action"] == "superseded_previous"
    assert "run-stale0000000000" in second["superseded_run_ids"]

    listing = service.list_runs(db_path=temp_db)
    assert listing["count"] == 2
    statuses = {r["run_id"]: r["status"] for r in listing["runs"]}
    assert statuses["run-stale0000000000"] == registry_db.STATUS_SUPERSEDED
    assert statuses[second["run_id"]] == registry_db.STATUS_ACTIVE
    assert listing["active_run_id"] == second["run_id"]


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "sample.csv"
    payload = b"model,accuracy\nxgboost,0.8781860988842256\n"
    target.write_bytes(payload)
    assert integrity_mod.sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_evidence_digest_is_order_independent_but_content_sensitive(tmp_path: Path) -> None:
    def record(path: str, sha: str) -> integrity_mod.ArtifactRecord:
        return integrity_mod.ArtifactRecord("g", path, sha, 1, "t")

    a, b = record("x.csv", "aa" * 32), record("y.csv", "bb" * 32)
    assert integrity_mod.evidence_digest([a, b]) == integrity_mod.evidence_digest([b, a])
    # Content change -> different digest.
    assert integrity_mod.evidence_digest([a, b]) != integrity_mod.evidence_digest(
        [a, record("y.csv", "cc" * 32)]
    )
    # Removal -> different digest.
    assert integrity_mod.evidence_digest([a, b]) != integrity_mod.evidence_digest([a])


def test_verify_detects_verified_changed_and_missing(tmp_path: Path) -> None:
    """
    Deliberate integrity mismatch, on real files.

    Three artefacts are copied into a temp tree, registered, then one is modified
    and one deleted. Verification runs real SHA-256 recomputation against those
    copies -- the repository's evidence is never involved.
    """
    root = tmp_path / "evidence"
    (root / "results" / "governance").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)

    keep = root / "results" / "model_metrics.csv"
    tamper = root / "results" / "governance" / "governance_risk_register.csv"
    remove = root / "data" / "raw" / "adult_metadata.txt"
    shutil.copy2(RESULTS / "model_metrics.csv", keep)
    shutil.copy2(RESULTS / "governance" / "governance_risk_register.csv", tamper)
    remove.write_text("placeholder\n", encoding="utf-8")

    registered = integrity_mod.discover_artifacts(root)
    paths = {r.path for r in registered}
    assert {"results/model_metrics.csv",
            "results/governance/governance_risk_register.csv",
            "data/raw/adult_metadata.txt"} <= paths

    # Baseline: everything verifies.
    baseline = integrity_mod.summarise_verification(
        integrity_mod.verify_artifacts(registered, root=root)
    )
    assert baseline["integrity_status"] == "verified"
    assert baseline["integrity_ok"] is True
    assert baseline["changed_count"] == 0 and baseline["missing_count"] == 0

    # Now break it: append a byte to one file, delete another.
    with tamper.open("a", encoding="utf-8") as handle:
        handle.write("R99,Injected,tampered row,,,,,,,,,\n")
    remove.unlink()

    after = integrity_mod.verify_artifacts(registered, root=root)
    summary = integrity_mod.summarise_verification(after)

    assert summary["integrity_status"] == "modified_and_incomplete"
    assert summary["integrity_ok"] is False
    assert summary["changed_files"] == ["results/governance/governance_risk_register.csv"]
    assert summary["missing_files"] == ["data/raw/adult_metadata.txt"]
    assert "results/model_metrics.csv" in summary["verified_files"]

    changed = next(r for r in after if r.status == "changed")
    assert changed.actual_sha256 != changed.expected_sha256
    assert changed.actual_size_bytes > changed.expected_size_bytes

    missing = next(r for r in after if r.status == "missing")
    assert missing.actual_sha256 is None


def test_touching_a_file_without_changing_content_keeps_integrity(tmp_path: Path) -> None:
    """mtime is excluded from the digest, so a touch must not break integrity."""
    root = tmp_path / "evidence"
    (root / "results").mkdir(parents=True)
    target = root / "results" / "model_metrics.csv"
    shutil.copy2(RESULTS / "model_metrics.csv", target)

    registered = integrity_mod.discover_artifacts(root)
    import os
    import time

    os.utime(target, (time.time() + 120, time.time() + 120))

    summary = integrity_mod.summarise_verification(
        integrity_mod.verify_artifacts(registered, root=root)
    )
    assert summary["integrity_status"] == "verified"


def test_integrity_reports_mismatch_through_the_service(temp_db: Path) -> None:
    """
    End-to-end mismatch detection: corrupt a STORED checksum in the temp database
    so the recomputed value cannot match. Evidence files stay untouched.
    """
    run_id = service.register_run(db_path=temp_db)["run_id"]
    assert service.check_integrity(run_id, db_path=temp_db)["integrity_ok"] is True

    with sqlite3.connect(temp_db) as connection:
        connection.execute(
            "UPDATE run_artifacts SET sha256 = ? WHERE run_id = ? AND path = ?",
            ("f" * 64, run_id, "results/model_metrics.csv"),
        )
        connection.commit()

    result = service.check_integrity(run_id, db_path=temp_db)
    assert result["integrity_ok"] is False
    assert result["integrity_status"] == "modified"
    assert result["changed_files"] == ["results/model_metrics.csv"]

    # The per-file record surfaces both sides of the mismatch.
    changed = next(a for a in result["artifacts"] if a["status"] == "changed")
    assert changed["expected_sha256"] == "f" * 64
    assert changed["actual_sha256"] != changed["expected_sha256"]
    assert len(changed["actual_sha256"]) == 64

    # Note: `current_evidence_digest` is recomputed from the files on disk, which
    # are deliberately untouched here, so it still equals the original. Only the
    # stored baseline was corrupted. The digest diverges when files really change
    # -- covered by test_verify_detects_verified_changed_and_missing.
    assert result["current_evidence_digest"] == result["registered_evidence_digest"]

    # A changed file must not be reported as wrongdoing.
    assert any("not by itself evidence of wrongdoing" in i for i in result["interpretation"])


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #
def test_timeline_records_registration_refresh_and_integrity(temp_db: Path) -> None:
    run_id = service.register_run(db_path=temp_db)["run_id"]
    service.register_run(db_path=temp_db)
    service.check_integrity(run_id, db_path=temp_db)

    timeline = service.get_timeline(run_id, db_path=temp_db)
    types = [e["event_type"] for e in timeline["events"]]
    assert "run_registered" in types
    assert "run_refreshed" in types
    assert "integrity_checked" in types
    assert any(t.startswith("evidence_produced:") for t in types)

    sources = {e["source"] for e in timeline["events"]}
    assert sources == {"registry", "evidence"}
    # Chronologically ordered.
    times = [e["event_time"] for e in timeline["events"]]
    assert times == sorted(times)


def test_unknown_run_raises_with_available_ids(temp_db: Path) -> None:
    known = service.register_run(db_path=temp_db)["run_id"]
    with pytest.raises(service.RunNotFoundError) as excinfo:
        service.get_run("run-does-not-exist", db_path=temp_db)
    assert known in excinfo.value.available


def test_missing_database_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    absent = tmp_path / "never_created.db"
    with pytest.raises(FileNotFoundError):
        service.list_runs(db_path=absent)
    stats = service.registry_stats(db_path=absent)
    assert stats["initialised"] is False and stats["total_runs"] == 0


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_api_runs_list(client: TestClient) -> None:
    body = _get(client, "/api/registry/runs")
    assert body["count"] >= 1
    assert body["active_run_id"]
    assert body["database"].startswith("runtime/")
    assert "does not make" in body["disclaimer"]
    run = next(r for r in body["runs"] if r["run_id"] == body["active_run_id"])
    assert run["status"] == "active"
    assert run["research_use"] == "conditionally_approved"
    assert run["real_world_deployment"] == "blocked"
    assert run["coverage_complete"] is True


def test_api_run_detail(client: TestClient) -> None:
    run_id = _get(client, "/api/registry/runs")["active_run_id"]
    body = _get(client, f"/api/registry/runs/{run_id}")
    assert body["run_id"] == run_id
    assert body["model_version"].startswith("sha256:")
    assert body["artifact_count"] == len(body["artifacts"])
    assert body["audit_coverage"]["complete"] is True
    assert body["blocking_risk_ids"]
    groups = {a["group"] for a in body["artifacts"]}
    for expected in ("dataset", "models", "predictions", "fairness", "explainability", "governance"):
        assert expected in groups


def test_api_integrity_endpoint(client: TestClient) -> None:
    run_id = _get(client, "/api/registry/runs")["active_run_id"]
    body = _get(client, f"/api/registry/runs/{run_id}/integrity")
    assert body["integrity_status"] == "verified"
    assert body["integrity_ok"] is True
    assert body["artifacts_checked"] == body["verified_count"] > 0
    assert body["missing_count"] == 0 and body["changed_count"] == 0
    assert body["registered_evidence_digest"] == body["current_evidence_digest"]
    assert body["interpretation"]
    assert all(a["status"] == "verified" for a in body["artifacts"])


def test_api_timeline_endpoint(client: TestClient) -> None:
    run_id = _get(client, "/api/registry/runs")["active_run_id"]
    body = _get(client, f"/api/registry/runs/{run_id}/timeline")
    assert body["count"] >= 2
    assert body["events"]
    assert "not a signed provenance record" in body["note"]


def test_api_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/registry/runs/run-nonexistent")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_api_status_filter(client: TestClient) -> None:
    body = _get(client, "/api/registry/runs?status=active")
    assert all(r["status"] == "active" for r in body["runs"])
    assert body["filters_applied"]["status"] == "active"


def test_registry_endpoints_are_read_only(client: TestClient) -> None:
    assert client.post("/api/registry/runs").status_code == 405
    run_id = _get(client, "/api/registry/runs")["active_run_id"]
    assert client.delete(f"/api/registry/runs/{run_id}").status_code == 405


def test_existing_endpoints_unaffected(client: TestClient) -> None:
    for path in (
        "/health",
        "/api/models",
        "/api/governance/decision",
        "/api/agents/review?model_name=xgboost",
    ):
        assert client.get(path).status_code == 200


# --------------------------------------------------------------------------- #
# The evidence must be untouched by everything above
# --------------------------------------------------------------------------- #
def test_registry_never_modifies_the_evidence(client: TestClient, temp_db: Path) -> None:
    before = {
        p: (integrity_mod.sha256_file(p), p.stat().st_mtime_ns, p.stat().st_size)
        for p in WATCHED
    }

    service.register_run(db_path=temp_db)
    run_id = service.list_runs(db_path=temp_db)["runs"][0]["run_id"]
    service.check_integrity(run_id, db_path=temp_db)
    service.get_timeline(run_id, db_path=temp_db)
    api_run = _get(client, "/api/registry/runs")["active_run_id"]
    _get(client, f"/api/registry/runs/{api_run}/integrity")

    after = {
        p: (integrity_mod.sha256_file(p), p.stat().st_mtime_ns, p.stat().st_size)
        for p in WATCHED
    }
    assert before == after, "the registry modified an evidence artefact"
