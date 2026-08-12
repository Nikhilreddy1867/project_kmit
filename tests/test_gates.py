"""
tests/test_gates.py
===================
Governance-as-Code tests: the policy gates, the waiver register, the Conformity
Bundle and the traceability matrix.

The onboarding suite asks whether the *measurements* are honest. This one asks
whether the *decisions* built on them are accountable, which is a different
property and mostly a set of refusals:

* the engine is deterministic -- the same evidence and policy version produce the
  same bundle id, so a changed bundle means changed evidence;
* Release and Operations can never come out ``PASS``, whatever the metrics say;
* a waiver is only ever created by a named human, with an expiry, against a
  control the policy says may be waived -- and never against the Release Gate;
* revoking a waiver keeps the row, because erasing an accepted risk would erase
  the fact that it was once accepted;
* the coverage numbers are named as governance coverage and never as compliance.

These tests exercise the real runtime tree and the real SQLite register for the
reasons set out in ``conftest.py``. Waivers are written against audit runs the
tests created and whose directories they remove afterwards; the built-in Adult
Income reference case is only ever read.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import PROJECT_ROOT, intake_files, intake_form

POLICY_GATES = ["DG", "TG", "VG", "RG", "OG"]

#: Gates the policy forbids from ever being computed to PASS.
NEVER_AUTO_PASS = ("RG", "OG")

#: The only controls the shipped policy marks waiver-eligible.
WAIVER_ELIGIBLE = {"VG-01", "VG-02", "VG-03"}


def future_expiry(days: int = 30) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
        timespec="seconds"
    )


def waiver_payload(control_id: str = "VG-02", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "control_id": control_id,
        "scope": "This audit run only; synthetic test data.",
        "owner": "MAAT test suite (named human stand-in)",
        "expires_at": future_expiry(),
        "rationale": (
            "Recorded by the test suite to verify that an accepted risk is attributed, "
            "time-bounded and visible. Not a real governance decision."
        ),
        "compensating_controls": ["Manual review of the group metrics before any use."],
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Policy profile
# --------------------------------------------------------------------------- #
def test_policy_profile_is_served_with_thresholds_and_rules(
    client: TestClient,
) -> None:
    body = client.get("/api/gates/policies").json()
    assert body["count"] >= 1
    policy = body["policies"][0]

    assert [g["gate_code"] for g in policy["gates"]] == POLICY_GATES
    assert policy["policy_version"]
    assert policy["checksum"], "the policy file is not checksummed"

    # Served as a list of named thresholds rather than a mapping, so that the order
    # the policy author wrote them in survives serialisation.
    thresholds = {t["name"]: t for t in policy["thresholds"]}
    assert thresholds["roc_auc_min"]["value"] == 0.85
    assert thresholds["f1_min"]["value"] == 0.65
    assert thresholds["disparate_impact_ratio_min"]["value"] == 0.80

    # A threshold with no evidence behind it is NOT_EVALUATED, never a pass.
    for name in ("roc_auc_min", "f1_min", "disparate_impact_ratio_min"):
        assert thresholds[name]["on_absent_evidence"] == "NOT_EVALUATED"


def test_policy_marks_only_validation_controls_waiver_eligible(
    client: TestClient,
) -> None:
    """
    Waiver-eligibility is a property of the policy, not of the request.

    Release and Operations controls are excluded at the policy level, so no request
    shape can reach a waiver against them.
    """
    policy = client.get("/api/gates/policies").json()["policies"][0]
    eligible = {c["control_id"] for c in policy["controls"] if c["waiver_eligible"]}
    assert eligible == WAIVER_ELIGIBLE
    assert set(policy["never_auto_pass"]["gates"]) == set(NEVER_AUTO_PASS)


def test_coverage_metrics_are_named_governance_coverage_not_compliance(
    client: TestClient,
) -> None:
    """
    The naming is load-bearing: a "compliance score" would be a claim this platform
    is not entitled to make.
    """
    policy = client.get("/api/gates/policies").json()["policies"][0]
    for metric in policy["coverage_metrics"].values():
        naming = metric["naming"].lower()
        assert "governance coverage metric" in naming
        assert "not a certified regulatory compliance score" in naming


# --------------------------------------------------------------------------- #
# Gate evaluation
# --------------------------------------------------------------------------- #
def test_evaluation_covers_every_gate_with_a_reason(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    body = client.get(f"/api/gates/runs/{run_id}/evaluation").json()

    assert body["deterministic"] is True
    assert [g["gate_code"] for g in body["gates"]] == POLICY_GATES
    assert set(body["gate_summary"]) == set(POLICY_GATES)

    for gate in body["gates"]:
        assert gate["status"] in {"PASS", "WAIVE", "BLOCK", "NOT_EVALUATED"}
        assert gate["reason"], f"{gate['gate_code']} has no stated reason"
        assert gate["control_ids"], f"{gate['gate_code']} has no controls"

    for control in body["controls"]:
        assert control["gate_result"] in {"PASS", "WAIVE", "BLOCK", "NOT_EVALUATED"}
        assert control["reason"], f"{control['control_id']} has no stated reason"
        assert control["policy_requirement"]


def test_release_and_operations_gates_are_never_passed(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    The central refusal of the whole platform: no computation may conclude that a
    user-uploaded model is releasable.
    """
    run_id = audit_run["audit_run_id"]
    body = client.get(f"/api/gates/runs/{run_id}/evaluation").json()

    for gate in body["gates"]:
        if gate["gate_code"] in NEVER_AUTO_PASS:
            assert gate["never_auto_pass"] is True
            assert gate["status"] in {"BLOCK", "NOT_EVALUATED"}, (
                f"{gate['gate_code']} came out {gate['status']}"
            )

    for control in body["controls"]:
        if control["gate"] in NEVER_AUTO_PASS:
            assert control["gate_result"] in {"BLOCK", "NOT_EVALUATED"}

    assert body["release_gate_note"]
    assert "not" in body["release_gate_note"].lower()


def test_governance_never_authorises_deployment(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    governance = client.get(f"/api/onboarding/audits/{run_id}/governance").json()
    assert governance["human_review_required"] is True
    # A closed vocabulary rather than prose: "not_granted" is the only value this
    # platform can ever produce for an uploaded model, and asserting the literal
    # means a future value like "granted" fails here rather than reading as prose.
    assert governance["deployment_authorisation"] == "not_granted"


def test_evaluation_is_deterministic_and_reports_no_change(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    Re-evaluating unchanged evidence must produce an identical result.

    ``changed: false`` is the assertion that matters: it is what makes a *changed*
    bundle meaningful evidence that the underlying artefacts moved.
    """
    run_id = audit_run["audit_run_id"]
    first = client.get(f"/api/gates/runs/{run_id}/evaluation").json()

    response = client.post(f"/api/gates/runs/{run_id}/evaluate")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["changed"] is False, "identical evidence produced a different result"

    second = client.get(f"/api/gates/runs/{run_id}/evaluation").json()
    for field in ("gate_summary", "status_counts", "blocking_controls"):
        assert first[field] == second[field]
    assert [c["gate_result"] for c in first["controls"]] == [
        c["gate_result"] for c in second["controls"]
    ]
    # evaluated_at is expected to move; nothing else is.
    assert first["policy_checksum"] == second["policy_checksum"]


def test_bundle_id_is_stable_for_identical_evidence_and_policy(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    before = client.get(f"/api/gates/runs/{run_id}/bundle").json()

    evaluated = client.post(f"/api/gates/runs/{run_id}/evaluate").json()
    after = client.get(f"/api/gates/runs/{run_id}/bundle").json()

    assert before["bundle_id"] == after["bundle_id"]
    assert evaluated["conformity_bundle_id"] == before["bundle_id"]
    assert before["evidence_digest"] == after["evidence_digest"]


def test_bundle_id_changes_when_runtime_evidence_changes(
    client: TestClient, audit_run: dict[str, Any], evidence_unchanged: None
) -> None:
    """
    Content addressing has to actually address content.

    A runtime artefact belonging to this test's own run is deliberately edited and
    then restored. ``evidence_unchanged`` runs alongside to prove the immutable tree
    is not what moved.
    """
    from app.onboarding import runtime_store as store

    run_id = audit_run["audit_run_id"]
    before = client.get(f"/api/gates/runs/{run_id}/bundle").json()

    target = store.artifact_path(run_id, "performance.json")
    original = target.read_bytes()
    try:
        payload = json.loads(original)
        payload["_tamper_marker"] = "written by test_bundle_id_changes"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        integrity = client.get(f"/api/onboarding/audits/{run_id}/integrity").json()
        assert integrity["integrity_ok"] is False
        assert integrity["changed_count"] >= 1
        changed = [a for a in integrity["artifacts"] if a["status"] == "changed"]
        assert any(a["artifact"] == "performance.json" for a in changed)

        client.post(f"/api/gates/runs/{run_id}/evaluate")
        after = client.get(f"/api/gates/runs/{run_id}/bundle").json()
        assert after["bundle_id"] != before["bundle_id"]
    finally:
        target.write_bytes(original)
        client.post(f"/api/gates/runs/{run_id}/evaluate")


def test_bundle_records_no_signature_rather_than_claiming_one(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    No signing key exists in this prototype, so ``signature`` must be null.

    Reporting anything else would be claiming a cryptographic guarantee the platform
    does not provide.
    """
    run_id = audit_run["audit_run_id"]
    bundle = client.get(f"/api/gates/runs/{run_id}/bundle").json()
    assert bundle["signature"] is None
    assert bundle["evidence_digest"], "the bundle is not content-addressed at all"
    text = json.dumps(bundle).lower()
    assert "digitally signed" not in text


def test_bundle_carries_traceable_evidence_and_disclaimers(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    bundle = client.get(f"/api/gates/runs/{run_id}/bundle").json()

    assert bundle["audit_run_id"] == run_id
    assert bundle["run_type"] == "uploaded_model"
    assert bundle["policy_checksum"]
    assert bundle["gate_sequence"] == POLICY_GATES
    assert set(bundle["gate_decisions"]) == set(POLICY_GATES)
    assert bundle["control_findings"], "no control findings in the bundle"
    assert bundle["evidence"], "no evidence listed in the bundle"
    assert bundle["disclaimers"], "no disclaimers in the bundle"

    for item in bundle["evidence"]:
        assert item["path"].replace("\\", "/").startswith("runtime/"), item["path"]
        if item["status"] == "verified":
            assert len(item["sha256"]) == 64
            assert item["size_bytes"] > 0

    disclaimers = " ".join(bundle["disclaimers"]).lower()
    for forbidden in ("legally compliant", "proves discrimination", "approved for deployment"):
        assert forbidden not in disclaimers
    # The four claims the platform is forbidden from making are each denied by name.
    assert "not a legal compliance assessment" in disclaimers
    assert "does not prove that discrimination occurred" in disclaimers
    assert "no causal mechanism" in disclaimers
    assert "does not authorise deployment" in disclaimers
    assert "no digital signature is applied" in disclaimers

    caveat = bundle["coverage_metric_caveat"].lower()
    assert "governance coverage metric" in caveat
    assert "not certified regulatory compliance" in caveat


# --------------------------------------------------------------------------- #
# Traceability
# --------------------------------------------------------------------------- #
def test_traceability_rows_resolve_to_real_files_with_matching_checksums(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    Traceability is only worth anything if the paths and checksums it cites resolve.

    Every row is followed to disk here, rather than trusted because it is a string
    that looks like a path.
    """
    from app.registry.integrity import sha256_file

    run_id = audit_run["audit_run_id"]
    matrix = client.get(f"/api/gates/runs/{run_id}/traceability").json()

    assert matrix["rows"], "the traceability matrix is empty"
    assert matrix["unresolved_evidence"] == [], matrix["unresolved_evidence"]

    control_ids = {r["control_id"] for r in matrix["rows"]}
    policy = client.get("/api/gates/policies").json()["policies"][0]
    assert control_ids == {c["control_id"] for c in policy["controls"]}

    checked = 0
    for row in matrix["rows"]:
        assert row["policy_requirement"]
        assert row["gate"] in POLICY_GATES
        path = row["evidence_artifact_path"]
        if not path:
            # Controls with no machine evidence (release authorisation, monitoring)
            # must say so rather than cite a file that does not exist.
            assert row["evidence_status"] in {
                "not_applicable", "not_required", "absent", "not_available"
            }
            assert row["limitation"]
            continue
        resolved = (PROJECT_ROOT / path).resolve()
        assert resolved.is_relative_to((PROJECT_ROOT / "runtime").resolve()), path
        if row["evidence_status"] == "verified":
            assert resolved.is_file(), f"{path} does not exist"
            assert row["expected_checksum"] == sha256_file(resolved)
            assert row["actual_checksum"] == row["expected_checksum"]
            checked += 1
    assert checked >= 3, "almost nothing in the matrix was verifiable"


def test_coverage_scores_are_fractions_carrying_their_caveat(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    for url in (
        f"/api/gates/runs/{run_id}/evaluation",
        f"/api/gates/runs/{run_id}/traceability",
    ):
        body = client.get(url).json()
        for key in ("evidence_coverage_score", "control_coverage_score"):
            score = body[key]
            assert score is None or 0.0 <= score <= 1.0, (url, key, score)
        caveat = body["coverage_metric_caveat"].lower()
        assert "coverage" in caveat
        assert (
            "not certified regulatory compliance" in caveat
            or "not a compliance percentage" in caveat
        ), caveat
        assert "compliance score" not in caveat.replace(
            "not certified regulatory compliance scores", ""
        )


# --------------------------------------------------------------------------- #
# Waiver register
# --------------------------------------------------------------------------- #
def test_new_run_has_no_waivers(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """The platform never creates a waiver on its own behalf."""
    run_id = audit_run["audit_run_id"]
    assert client.get(f"/api/gates/runs/{run_id}/waivers").json() == []

    evaluation = client.get(f"/api/gates/runs/{run_id}/evaluation").json()
    assert evaluation["waivers_applied"] == []
    assert all(c["waiver_id"] is None for c in evaluation["controls"])
    assert all(g["status"] != "WAIVE" for g in evaluation["gates"])


def test_waiver_against_the_release_gate_is_refused(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    A waiver can never satisfy or override the Release Gate.

    Both Release controls are attempted, so the guarantee does not rest on one
    control id happening to be excluded.
    """
    run_id = audit_run["audit_run_id"]
    for control_id in ("RG-01", "RG-02", "OG-01", "OG-02"):
        response = client.post(
            f"/api/gates/runs/{run_id}/waivers", json=waiver_payload(control_id)
        )
        assert response.status_code == 422, (control_id, response.text)
        assert response.json()["error"] == "control_not_waiver_eligible"
    assert client.get(f"/api/gates/runs/{run_id}/waivers").json() == []


def test_waiver_against_an_unknown_control_is_refused(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    response = client.post(
        f"/api/gates/runs/{run_id}/waivers", json=waiver_payload("XX-99")
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "unknown_control"


@pytest.mark.parametrize(
    "expires_at, expected",
    [
        ("", "invalid_expiry"),
        ("not-a-date", "invalid_expiry"),
        ("2019-01-01T00:00:00+00:00", "expiry_in_the_past"),
    ],
)
def test_waiver_must_be_time_bounded(
    client: TestClient, audit_run: dict[str, Any], expires_at: str, expected: str
) -> None:
    """A waiver with no usable expiry is an open-ended override, not a waiver."""
    run_id = audit_run["audit_run_id"]
    response = client.post(
        f"/api/gates/runs/{run_id}/waivers",
        json=waiver_payload(expires_at=expires_at),
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"] == expected


@pytest.mark.parametrize(
    "missing", ["control_id", "scope", "owner", "expires_at", "rationale"]
)
def test_waiver_schema_requires_every_accountability_field(
    client: TestClient, audit_run: dict[str, Any], missing: str
) -> None:
    """
    Each field is what makes the waiver attributable. None of them is defaulted.
    """
    run_id = audit_run["audit_run_id"]
    payload = waiver_payload()
    payload.pop(missing)
    response = client.post(f"/api/gates/runs/{run_id}/waivers", json=payload)
    assert response.status_code == 422, (missing, response.text)
    assert client.get(f"/api/gates/runs/{run_id}/waivers").json() == []


def test_explicit_waiver_is_recorded_attributed_and_applied(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    The one path that does create a waiver: an explicit request naming an owner, an
    expiry and a rationale, against a control the policy allows to be waived.
    """
    run_id = audit_run["audit_run_id"]
    payload = waiver_payload("VG-02")
    created = client.post(f"/api/gates/runs/{run_id}/waivers", json=payload)
    assert created.status_code == 201, created.text
    waiver = created.json()

    assert waiver["waiver_id"]
    assert waiver["audit_run_id"] == run_id
    assert waiver["control_id"] == "VG-02"
    assert waiver["gate"] == "VG"
    assert waiver["owner"] == payload["owner"]
    assert waiver["status"] == "active"
    assert waiver["created_by_platform"] is False
    assert "does not make the requirement met" in waiver["notice"]

    assert [w["waiver_id"] for w in
            client.get(f"/api/gates/runs/{run_id}/waivers").json()] == [
        waiver["waiver_id"]
    ]

    # Recording a waiver does not by itself change the evaluation: the user has to
    # ask for a re-evaluation, so the effect of accepting a risk is a visible act.
    client.post(f"/api/gates/runs/{run_id}/evaluate")
    evaluation = client.get(f"/api/gates/runs/{run_id}/evaluation").json()
    finding = next(c for c in evaluation["controls"] if c["control_id"] == "VG-02")
    if finding["gate_result"] == "WAIVE":
        assert finding["waiver_id"] == waiver["waiver_id"]
        applied = [w["waiver_id"] for w in evaluation["waivers_applied"]]
        assert waiver["waiver_id"] in applied
    else:
        # The control was already satisfied, so there was nothing to waive. A waiver
        # must never *downgrade* a passing control.
        assert finding["gate_result"] == "PASS"


def test_a_waived_validation_control_does_not_release_the_model(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    Accepting a validation risk cannot cascade into a release decision.
    """
    run_id = audit_run["audit_run_id"]
    for control_id in sorted(WAIVER_ELIGIBLE):
        client.post(
            f"/api/gates/runs/{run_id}/waivers", json=waiver_payload(control_id)
        )
    client.post(f"/api/gates/runs/{run_id}/evaluate")

    evaluation = client.get(f"/api/gates/runs/{run_id}/evaluation").json()
    for gate_code in NEVER_AUTO_PASS:
        assert evaluation["gate_summary"][gate_code] in {"BLOCK", "NOT_EVALUATED"}

    governance = client.get(f"/api/onboarding/audits/{run_id}/governance").json()
    assert governance["human_review_required"] is True
    assert governance["governance_state"] in {
        "review_required",
        "insufficient_evidence",
        "blocked_by_policy",
    }


def test_revoking_a_waiver_keeps_the_record(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    The register is a history. Deleting a revoked waiver would erase the fact that
    the risk was once accepted, which is the one thing the register exists to hold.
    """
    run_id = audit_run["audit_run_id"]
    waiver = client.post(
        f"/api/gates/runs/{run_id}/waivers", json=waiver_payload("VG-03")
    ).json()

    revoked = client.post(
        f"/api/gates/runs/{run_id}/waivers/{waiver['waiver_id']}/revoke"
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["status"] == "revoked"

    listed = client.get(f"/api/gates/runs/{run_id}/waivers").json()
    assert [w["waiver_id"] for w in listed] == [waiver["waiver_id"]]
    assert listed[0]["status"] == "revoked"
    assert listed[0]["owner"] == waiver["owner"]
    assert listed[0]["rationale"] == waiver["rationale"]

    # A revoked waiver has no effect: the control reverts.
    client.post(f"/api/gates/runs/{run_id}/evaluate")
    evaluation = client.get(f"/api/gates/runs/{run_id}/evaluation").json()
    finding = next(c for c in evaluation["controls"] if c["control_id"] == "VG-03")
    assert finding["gate_result"] != "WAIVE"
    assert finding["waiver_id"] is None


def test_revoking_an_unknown_waiver_is_a_clean_refusal(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    response = client.post(
        f"/api/gates/runs/{run_id}/waivers/waiver-does-not-exist/revoke"
    )
    assert response.status_code in {404, 422}
    assert response.json()["error"] == "waiver_not_found"


# --------------------------------------------------------------------------- #
# Storage boundaries
# --------------------------------------------------------------------------- #
def test_waivers_live_only_in_the_runtime_sqlite_state(
    client: TestClient, audit_run: dict[str, Any], evidence_unchanged: None
) -> None:
    """
    A waiver is runtime state, not evidence.

    It must be in the runtime SQLite register and nowhere else -- specifically not
    written into the audit run's evidence artefacts, whose checksums are what the
    bundle attests to.
    """
    from app.onboarding import runtime_store as store
    from app.registry.db import resolve_db_path

    run_id = audit_run["audit_run_id"]
    payload = waiver_payload("VG-01")
    marker = "unique-waiver-marker-6f2a1c"
    payload["rationale"] = f"{payload['rationale']} {marker}"
    created = client.post(f"/api/gates/runs/{run_id}/waivers", json=payload)
    assert created.status_code == 201, created.text

    db_path = resolve_db_path(None).resolve()
    assert db_path.is_relative_to(store.RUNTIME_DIR.resolve()), db_path

    connection = sqlite3.connect(db_path)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM governance_waivers WHERE audit_run_id = ?", (run_id,)
        ).fetchall()
    finally:
        connection.close()
    assert [r["control_id"] for r in rows] == ["VG-01"]
    assert marker in rows[0]["rationale"]

    # The marker appears in no file anywhere in the repository.
    for path in sorted(PROJECT_ROOT.rglob("*.json")) + sorted(
        PROJECT_ROOT.rglob("*.csv")
    ):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        assert marker not in text, f"waiver text leaked into {path}"


def test_no_waiver_exists_for_the_builtin_reference_case(
    client: TestClient,
) -> None:
    """
    The reference case is not governed by this policy and carries no fabricated
    waiver. Its own recorded decision is what stands.
    """
    from app.gates import service as gates_service

    for reference_id in ("adult-income", "adult_income", "reference", "builtin"):
        assert gates_service.list_waivers(reference_id) == []

    from app.registry.db import connect

    connection = connect()
    try:
        gates_service.ensure_waiver_table(connection)
        total = connection.execute(
            "SELECT COUNT(*) FROM governance_waivers WHERE audit_run_id NOT LIKE ?",
            ("audit-%",),
        ).fetchone()[0]
    finally:
        connection.close()
    assert total == 0, "a waiver exists against something that is not an uploaded run"

    decision = client.get("/api/governance/decision").json()
    assert "waiver" not in json.dumps(decision).lower()


def test_gate_artifacts_are_written_only_inside_the_run_directory(
    client: TestClient, audit_run: dict[str, Any], evidence_unchanged: None
) -> None:
    from app.onboarding import runtime_store as store

    run_id = audit_run["audit_run_id"]
    response = client.post(f"/api/gates/runs/{run_id}/evaluate").json()
    assert response["artifacts_rewritten"], "re-evaluation wrote nothing at all"

    run_dir = store.audit_dir(run_id).resolve()
    for name in response["artifacts_rewritten"]:
        path = (run_dir / Path(name).name).resolve()
        assert path.is_relative_to(run_dir), name
        assert path.is_file(), name


# --------------------------------------------------------------------------- #
# Unknown runs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "suffix", ["/evaluation", "/bundle", "/traceability", "/waivers"]
)
def test_gate_endpoints_404_for_an_unknown_run(
    client: TestClient, suffix: str
) -> None:
    response = client.get(f"/api/gates/runs/no-such-run{suffix}")
    assert response.status_code == 404, response.text
    body = response.json()
    assert body["error"]
    assert body["message"]


def test_evaluating_an_unknown_run_is_refused(client: TestClient) -> None:
    response = client.post("/api/gates/runs/no-such-run/evaluate")
    assert response.status_code == 404, response.text


def test_unknown_policy_id_is_refused_with_alternatives(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    response = client.post(
        f"/api/gates/runs/{run_id}/evaluate?policy_profile_id=no-such-policy"
    )
    assert response.status_code in {404, 422}, response.text
    body = response.json()
    assert body["error"]
    assert body.get("available") or body.get("hint")
