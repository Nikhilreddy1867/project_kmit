"""
app/gates/policy_engine.py
==========================
The Governance-as-Code engine: it reads a versioned policy file and an audit run's
evidence, and produces gate decisions.

Why the policy lives in JSON and not in this file
-------------------------------------------------
Every threshold, control, owner and limitation comes from
``app/gates/policies/research_governance_policy_v1.json``. This module contains the
*evaluation logic* only. Changing a threshold is therefore a policy edit with a
version bump and a new checksum -- visible in the Conformity Bundle -- rather than a
code change that silently reinterprets past decisions. The policy checksum is
recorded in every evaluation so a reader can tell exactly which policy text produced
a given result.

Determinism
-----------
Nothing here samples, learns, caches across runs or reads the clock except to stamp
``evaluated_at``. Identical evidence plus an identical policy version produces a
byte-identical set of gate results, which is what makes re-evaluation a meaningful
integrity check rather than a fresh opinion.

The two results that are not achievements
-----------------------------------------
``NOT_EVALUATED`` means the evidence a control needs does not exist. ``WAIVE`` means
a named human accepted an unmet requirement until a stated expiry. Neither is a
pass, and the governance state derived at the bottom of this module treats them
accordingly: a run whose Validation Gate could not be assessed becomes
``insufficient_evidence``, never ``review_required``.

Release and Operations can never pass
-------------------------------------
:func:`evaluate` reads ``never_auto_pass`` from the policy and excludes those gates
from the PASS branch structurally, so no combination of evidence can make this
platform state that an uploaded model is production-ready.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.gates.schemas import (
    FAIRNESS_GATE_NOTICE,
    ControlFinding,
    GateEvaluation,
    GateResult,
    PolicyControl,
    PolicyGate,
    PolicyProfile,
    PolicyThreshold,
    Waiver,
)
from app.onboarding import runtime_store as store
from app.registry.integrity import sha256_file

POLICY_DIR = Path(__file__).resolve().parent / "policies"
DEFAULT_POLICY_ID = "research_governance_policy_v1"

#: Controls that assess evidence this prototype *can* produce. A NOT_EVALUATED here
#: means the user could have supplied more (probabilities, sensitive columns), so the
#: run is short of evidence rather than merely awaiting a human.
ASSESSABLE_GATES = ("DG", "TG", "VG")


class PolicyNotFound(Exception):
    """No policy profile with that id ships with this build."""

    def __init__(self, policy_id: str, available: list[str]):
        self.policy_id = policy_id
        self.available = available
        super().__init__(
            f"No policy profile '{policy_id}'. Available: {', '.join(available) or 'none'}."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    """
    SHA-256 of an in-memory buffer.

    Defined here rather than added to :mod:`app.registry.integrity`, which hashes
    *files* for the reference case and is deliberately left untouched.
    """
    import hashlib

    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def available_policy_ids() -> list[str]:
    if not POLICY_DIR.is_dir():
        return []
    return sorted(p.stem for p in POLICY_DIR.glob("*.json"))


def load_policy(policy_id: str | None = None) -> PolicyProfile:
    """
    Read one policy file and hash it.

    The checksum is over the file bytes as loaded, so an evaluation can be tied to
    the exact policy text rather than only to its declared version number -- an edit
    without a version bump is still detectable.
    """
    resolved = policy_id or DEFAULT_POLICY_ID
    path = POLICY_DIR / f"{resolved}.json"
    if not path.is_file():
        raise PolicyNotFound(resolved, available_policy_ids())

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))

    controls = [PolicyControl(**c) for c in payload["controls"]]
    by_gate: dict[str, list[str]] = {}
    for control in controls:
        by_gate.setdefault(control.gate, []).append(control.control_id)

    never = set(payload.get("never_auto_pass", {}).get("gates", []))
    gates = [
        PolicyGate(
            **g,
            controls=by_gate.get(g["gate_code"], []),
            never_auto_pass=g["gate_code"] in never,
        )
        for g in payload["gate_sequence"]
    ]

    thresholds = [
        PolicyThreshold(name=name, **spec)
        for name, spec in payload.get("thresholds", {}).items()
    ]

    return PolicyProfile(
        policy_id=payload["policy_id"],
        policy_name=payload["policy_name"],
        policy_version=payload["policy_version"],
        policy_status=payload.get("policy_status", "active"),
        effective_from=payload.get("effective_from"),
        purpose=payload["purpose"],
        applies_to=payload.get("applies_to", {}),
        gates=gates,
        controls=controls,
        thresholds=thresholds,
        statuses=payload["statuses"],
        gate_result_rule=payload["gate_result_rule"],
        never_auto_pass=payload.get("never_auto_pass", {}),
        decision_semantics=payload.get("decision_semantics", {}),
        waiver_rules=payload.get("waiver_rules", {}),
        coverage_metrics=payload.get("coverage_metrics", {}),
        limitations=payload.get("limitations", []),
        source_file=store.relative_path(path),
        checksum=_sha256_bytes(raw),
    )


def load_all_policies() -> list[PolicyProfile]:
    return [load_policy(pid) for pid in available_policy_ids()]


def threshold_value(policy: PolicyProfile, name: str, default: Any = None) -> Any:
    for threshold in policy.thresholds:
        if threshold.name == name:
            return threshold.value
    return default


# --------------------------------------------------------------------------- #
# Evidence resolution
# --------------------------------------------------------------------------- #
def _expected_checksums(evidence_manifest: dict[str, Any]) -> dict[str, str]:
    """Map repo-relative path -> SHA-256 recorded when the artefact was created."""
    return {
        str(entry.get("path")): str(entry.get("sha256"))
        for entry in evidence_manifest.get("artifacts", [])
        if entry.get("path") and entry.get("sha256")
    }


def _check_artifact(
    audit_run_id: str, name: str | None, expected: dict[str, str]
) -> tuple[str | None, str | None, str | None, str]:
    """
    Resolve one artefact and compare its checksum to the recorded one.

    Returns ``(relative_path, expected_sha, actual_sha, evidence_status)``.
    """
    if not name:
        return None, None, None, "not_applicable"

    path = store.artifact_path(audit_run_id, name)
    relative = store.relative_path(path)
    expected_sha = expected.get(relative)
    if not path.is_file():
        return relative, expected_sha, None, "missing"

    actual_sha = sha256_file(path)
    if expected_sha is None:
        # No recorded checksum to compare against -- the artefact exists, but its
        # integrity is unestablished. Reported as such rather than as verified.
        return relative, None, actual_sha, "missing"
    status = "verified" if actual_sha == expected_sha else "changed"
    return relative, expected_sha, actual_sha, status


# --------------------------------------------------------------------------- #
# Per-control rules
# --------------------------------------------------------------------------- #
def _finding(
    control: PolicyControl,
    *,
    path: str | None,
    expected: str | None,
    actual: str | None,
    evidence_status: str,
    gate_result: str,
    reason: str,
    action: str,
    observed: dict[str, Any] | None = None,
) -> ControlFinding:
    return ControlFinding(
        control_id=control.control_id,
        gate=control.gate,
        title=control.title,
        policy_requirement=control.requirement,
        evidence_artifact_path=path,
        source_api_endpoint=control.api_endpoint,
        expected_checksum=expected,
        actual_checksum=actual,
        evidence_status=evidence_status,  # type: ignore[arg-type]
        gate_result=gate_result,  # type: ignore[arg-type]
        observed=observed or {},
        reason=reason,
        limitation=control.limitation,
        recommended_action=action,
        waiver_eligible=control.waiver_eligible,
    )


def _eval_dg01(control, ctx, evidence) -> ControlFinding:
    path, expected, actual, status = evidence
    dataset = ctx.get("dataset_metadata") or {}
    if status == "missing":
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="The dataset provenance record was not found, so provenance could "
            "not be assessed.",
            action="Re-run the audit so the dataset metadata artefact is produced.",
        )
    required = {
        "sha256": dataset.get("sha256"),
        "row_count": dataset.get("row_count"),
        "column_count": dataset.get("column_count"),
        "target_column": dataset.get("target_column"),
        "decision_context": (ctx.get("model_metadata") or {}).get("decision_context"),
    }
    absent = [k for k, v in required.items() if v in (None, "", 0)]
    if absent:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="The dataset provenance record is incomplete; missing: "
            + ", ".join(absent) + ".",
            action="Supply the missing provenance fields and re-run the audit.",
            observed=required,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason=f"Dataset identified by checksum with {required['row_count']} rows, "
        f"{required['column_count']} columns and a declared decision context.",
        action="No action. Note that provenance is as declared by the uploader and "
        "cannot be independently verified here.",
        observed=required,
    )


def _eval_dg02(control, ctx, evidence) -> ControlFinding:
    """
    Recompute the stored dataset file's hash against the upload-time record.

    Two files are in play, and keeping them apart is the point. ``expected_checksum``
    and ``actual_checksum`` always describe the artefact named in
    ``evidence_artifact_path``, so a reviewer who hashes that path gets the value the
    row shows. The *subject* of the control -- the stored dataset -- is reported under
    ``observed`` as ``source_path``/``recorded_sha256``/``recomputed_sha256``. Putting
    the dataset's hashes in the artefact's checksum columns would make every row look
    tampered with to anyone who actually followed it, which is the opposite of what a
    traceability matrix is for.

    ``evidence_status`` therefore answers "is the cited evidence document intact?" and
    ``gate_result`` answers "does the requirement hold?". They are allowed to differ:
    an intact record can perfectly well record a dataset that has since changed.
    """
    path, expected, actual, status = evidence
    dataset = ctx.get("dataset_metadata") or {}
    recorded = dataset.get("sha256")
    source = dataset.get("stored_path")
    observed = {"recorded_sha256": recorded, "source_path": source}

    source_path = store.PROJECT_ROOT / str(source) if source else None
    if status == "missing" or not recorded or source_path is None or not source_path.is_file():
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="The stored dataset file or its upload-time checksum is not "
            "available, so integrity could not be checked.",
            action="Re-upload the dataset to re-establish an integrity baseline.",
            observed=observed,
        )

    current = sha256_file(source_path)
    observed["recomputed_sha256"] = current
    if current != recorded:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="The stored dataset file no longer matches the checksum recorded at "
            f"upload time ({recorded} became {current}), so the audited evidence has "
            "changed since the audit ran.",
            action="Treat every metric in this run as unverified and re-run the audit "
            "on a known-good dataset.",
            observed=observed,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason="The stored dataset still hashes to the checksum recorded at upload "
        f"({recorded}).",
        action="No action. This detects change only -- it is not a digital signature "
        "and does not authenticate the uploader.",
        observed=observed,
    )


def _eval_tg01(control, ctx, evidence) -> ControlFinding:
    path, expected, actual, status = evidence
    model = ctx.get("model_metadata") or {}
    recorded = model.get("sha256")
    source = model.get("stored_path")
    observed = {
        "model_name": model.get("model_name"),
        "model_version": model.get("model_version"),
        "model_owner": model.get("model_owner"),
        "recorded_sha256": recorded,
        "security_acknowledged": model.get("security_acknowledged"),
    }
    if status == "missing":
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="The model identity record was not found.",
            action="Re-run the audit so the model metadata artefact is produced.",
            observed=observed,
        )

    absent = [
        k
        for k in ("model_name", "model_version", "model_owner", "recorded_sha256")
        if not observed.get(k)
    ]
    if absent:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="The model identity record is incomplete; missing: "
            + ", ".join(absent) + ".",
            action="Declare the missing model metadata and re-run the audit.",
            observed=observed,
        )
    if not model.get("security_acknowledged"):
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="No record exists that the joblib deserialisation warning was "
            "acknowledged before the model was loaded.",
            action="Re-submit with the security acknowledgement given.",
            observed=observed,
        )

    source_path = store.PROJECT_ROOT / str(source) if source else None
    if source_path is not None and source_path.is_file():
        current = sha256_file(source_path)
        observed["recomputed_sha256"] = current
        if current != recorded:
            return _finding(
                control, path=path, expected=recorded, actual=current,
                evidence_status="changed", gate_result="BLOCK",
                reason="The stored model file no longer matches the checksum recorded "
                "at upload time.",
                action="Re-run the audit; the audited artefact is no longer the one on "
                "disk.",
                observed=observed,
            )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason=f"Model '{observed['model_name']}' v{observed['model_version']} "
        f"identified by checksum, owned by {observed['model_owner']}, with the "
        "security acknowledgement recorded before loading.",
        action="No action. This identifies the audited artefact; it evidences nothing "
        "about how the model was trained.",
        observed=observed,
    )


def _eval_tg02(control, ctx, evidence) -> ControlFinding:
    path, expected, actual, status = evidence
    compatibility = (ctx.get("model_metadata") or {}).get("feature_compatibility") or {}
    observed = {
        "method": compatibility.get("method"),
        "missing_features": compatibility.get("missing_features", []),
        "unexpected_features_count": len(compatibility.get("unexpected_features", [])),
        "matched_feature_count": compatibility.get("matched_feature_count"),
    }
    if status == "missing" or not compatibility:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status="missing", gate_result="NOT_EVALUATED",
            reason="No feature-compatibility record is available.",
            action="Re-run the audit.", observed=observed,
        )
    if not compatibility.get("checked"):
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="The estimator exposes neither feature_names_in_ nor "
            "n_features_in_, so compatibility could not be checked structurally.",
            action="Use an estimator that records its input features so this control "
            "can be evaluated.",
            observed=observed,
        )
    if not compatibility.get("compatible"):
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="The dataset does not provide every feature the model expects; "
            f"missing: {', '.join(compatibility.get('missing_features', [])[:10])}.",
            action="Supply the missing columns, or audit a model fitted on this "
            "dataset's feature set.",
            observed=observed,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason=f"All expected input features are present ({observed['method']}).",
        action="No action. Matching names do not guarantee matching meaning, which "
        "this check cannot see.",
        observed=observed,
    )


def _eval_vg01(control, ctx, evidence, policy: PolicyProfile) -> ControlFinding:
    path, expected, actual, status = evidence
    performance = ctx.get("performance") or {}
    roc_min = float(threshold_value(policy, "roc_auc_min", 0.85))
    f1_min = float(threshold_value(policy, "f1_min", 0.65))

    roc = performance.get("roc_auc")
    f1 = performance.get("f1")
    observed = {
        "roc_auc": roc,
        "roc_auc_min": roc_min,
        "f1": f1,
        "f1_min": f1_min,
        "roc_auc_unavailable_reason": performance.get("roc_auc_unavailable_reason"),
        "decision_threshold": performance.get("decision_threshold"),
    }

    if status == "missing" or not performance:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status="missing", gate_result="NOT_EVALUATED",
            reason="No performance evidence is available.",
            action="Re-run the audit.", observed=observed,
        )

    failures: list[str] = []
    unassessed: list[str] = []

    if f1 is None:
        unassessed.append("F1 is undefined for this run (empty denominator)")
    elif float(f1) < f1_min:
        failures.append(f"F1 {float(f1):.4f} is below the configured floor {f1_min:.2f}")

    if roc is None:
        unassessed.append(
            "ROC-AUC is unavailable: "
            + str(observed["roc_auc_unavailable_reason"] or "no probability output")
        )
    elif float(roc) < roc_min:
        failures.append(
            f"ROC-AUC {float(roc):.4f} is below the configured floor {roc_min:.2f}"
        )

    if failures:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="Configured research performance threshold not met: "
            + "; ".join(failures) + ".",
            action="Improve the model, revisit the decision threshold, or record an "
            "explicit time-bounded waiver with compensating controls.",
            observed=observed,
        )
    if unassessed:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="The performance requirement could not be fully assessed: "
            + "; ".join(unassessed)
            + ". No value was estimated in place of the missing one.",
            action="Supply a probability-capable estimator so the ranking threshold "
            "can be evaluated.",
            observed=observed,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason=f"ROC-AUC {float(roc):.4f} >= {roc_min:.2f} and F1 {float(f1):.4f} >= "
        f"{f1_min:.2f} on the uploaded test set.",
        action="No action. These are held-out results on one dataset and do not "
        "establish that performance transfers.",
        observed=observed,
    )


def _eval_vg02(control, ctx, evidence, policy: PolicyProfile) -> ControlFinding:
    path, expected, actual, status = evidence
    fairness = ctx.get("fairness") or {}
    floor = float(threshold_value(policy, "disparate_impact_ratio_min", 0.80))
    fairness_status = fairness.get("status")

    if status == "missing" or not fairness:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status="missing", gate_result="NOT_EVALUATED",
            reason="No fairness evidence is available.",
            action="Re-run the audit selecting sensitive columns.",
        )

    if fairness_status == "not_provided_by_user":
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="No sensitive columns were selected, so no fairness screening was "
            "performed. This is an absence of evidence -- not a pass, and not a claim "
            "that the model is fair.",
            action="Re-run the audit selecting the sensitive attributes relevant to "
            "the stated decision context, where such data is lawfully available.",
            observed={"fairness_status": fairness_status, "threshold": floor},
        )

    attributes = fairness.get("attributes", [])
    ratios = [
        (a.get("attribute"), a.get("min_disparate_impact_ratio"), a.get("worst_group"))
        for a in attributes
        if a.get("min_disparate_impact_ratio") is not None
    ]
    observed = {
        "threshold": floor,
        "per_attribute_min_ratio": {name: value for name, value, _ in ratios},
        "small_groups_present": any(a.get("small_groups_present") for a in attributes),
    }

    if not ratios:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="Every disparate impact ratio was undefined (the reference group "
            "had no selected outcomes), so the screen could not be applied. Undefined "
            "was not converted to zero.",
            action="Check the positive class and threshold: a reference group with no "
            "predicted positives makes every ratio undefined.",
            observed=observed,
        )

    failing = [(n, v, g) for n, v, g in ratios if float(v) < floor]
    if failing:
        detail = "; ".join(
            f"{name}: {float(value):.4f} (worst group '{group}')"
            for name, value, group in failing
        )
        uncertainty = (
            " One or more affected groups are below the small-group threshold, so the "
            "difference may reflect sampling noise."
            if observed["small_groups_present"]
            else ""
        )
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason=f"Disparate impact ratio below the configured {floor:.2f} screening "
            f"threshold -- {detail}. {FAIRNESS_GATE_NOTICE}{uncertainty}",
            action="Have a human reviewer examine the affected groups, their base "
            "rates, the decision threshold and the feature set. Do not read this as a "
            "finding of discrimination.",
            observed=observed,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason=f"Every group's disparate impact ratio is at or above the configured "
        f"{floor:.2f} screening threshold. {FAIRNESS_GATE_NOTICE} Passing this screen "
        "is not a finding that the model is fair.",
        action="No action from the screen itself. Fairness criteria conflict "
        "mathematically, so passing one does not imply the others.",
        observed=observed,
    )


def _eval_vg03(control, ctx, evidence) -> ControlFinding:
    path, expected, actual, status = evidence
    explainability = ctx.get("explainability") or {}
    capabilities = (ctx.get("model_metadata") or {}).get("model_capabilities") or {}
    supported = bool(capabilities.get("supports_permutation_importance"))
    observed = {
        "explainability_status": explainability.get("status"),
        "model_type_supported": supported,
        "feature_count": len(explainability.get("global_importance", [])),
    }

    if status == "missing" or not explainability:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status="missing", gate_result="NOT_EVALUATED",
            reason="No explainability evidence is available.",
            action="Re-run the audit.", observed=observed,
        )
    if not supported:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="NOT_EVALUATED",
            reason="Explainability not available for this model type in the current "
            "local prototype, so this control does not apply. No importance scores "
            "were invented to fill the gap.",
            action="Supply a model type this prototype can explain, or provide "
            "external explainability evidence to the reviewer.",
            observed=observed,
        )
    if explainability.get("status") != "available" or not explainability.get(
        "global_importance"
    ):
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="The model type is supported but no global importance evidence was "
            "produced: "
            + str(explainability.get("status_detail") or "reason not recorded"),
            action="Investigate why importance computation failed, or record an "
            "explicit time-bounded waiver.",
            observed=observed,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="PASS",
        reason=f"Global importance evidence exists for {observed['feature_count']} "
        "original input feature(s).",
        action="No action. Importance is associational and correlated features share "
        "it, so a low score is not evidence a feature is unused.",
        observed=observed,
    )


def _eval_rg01(control, ctx, evidence) -> ControlFinding:
    return _finding(
        control, path=None, expected=None, actual=None,
        evidence_status="not_applicable", gate_result="NOT_EVALUATED",
        reason="No human release authorisation record exists, and this prototype "
        "provides no way to create one. This control can never be satisfied by "
        "computation and is explicitly not waiver-eligible.",
        action="Route the Conformity Bundle to the accountable release authority for "
        "a documented decision outside this platform.",
    )


def _eval_rg02(control, ctx, evidence, validation_blocked: bool) -> ControlFinding:
    path, expected, actual, status = evidence
    risk = ctx.get("risk_summary") or {}
    observed = {
        "risk_count": risk.get("total"),
        "severity_counts": risk.get("severity_counts", {}),
        "validation_gate_blocked": validation_blocked,
    }
    if status == "missing" or not risk:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status="missing", gate_result="NOT_EVALUATED",
            reason="No risk summary is available to the release authority.",
            action="Re-run the audit.", observed=observed,
        )
    if validation_blocked:
        return _finding(
            control, path=path, expected=expected, actual=actual,
            evidence_status=status, gate_result="BLOCK",
            reason="Deployment risk includes an unresolved Validation Gate block, so "
            "release cannot proceed on this evidence.",
            action="Resolve the blocking validation control, or record an explicit "
            "time-bounded waiver on it, before seeking a release decision.",
            observed=observed,
        )
    return _finding(
        control, path=path, expected=expected, actual=actual,
        evidence_status=status, gate_result="NOT_EVALUATED",
        reason=f"A deterministic risk summary with {observed['risk_count']} item(s) is "
        "available to the release authority. Making risk visible is not a release "
        "decision, so the Release Gate remains unevaluated by design.",
        action="Have the accountable human review the risk summary and record a "
        "decision outside this platform.",
        observed=observed,
    )


def _eval_operations(control, ctx, evidence) -> ControlFinding:
    return _finding(
        control, path=None, expected=None, actual=None,
        evidence_status="not_applicable", gate_result="NOT_EVALUATED",
        reason="Out of scope for this prototype, which accepts a model file and a "
        "dataset only. Not assessed, therefore neither passed nor failed.",
        action="Establish this operational control before any real-world use is "
        "considered, and evidence it outside this platform.",
    )


# --------------------------------------------------------------------------- #
# Waivers
# --------------------------------------------------------------------------- #
def apply_waivers(
    findings: list[ControlFinding], waivers: list[Waiver]
) -> tuple[list[ControlFinding], list[Waiver]]:
    """
    Downgrade BLOCK to WAIVE where an active, waiver-eligible waiver exists.

    Three guards, each of which exists because its absence would make the waiver
    register decorative: the control must be waiver-eligible in the policy (so RG-01
    can never be waived), the waiver must be ``active`` (an expired waiver has no
    effect and the control reverts to BLOCK), and the reason string retains the
    original failure so a reader sees *what* was accepted, not just that something was.
    """
    active = {w.control_id: w for w in waivers if w.status == "active"}
    applied: list[Waiver] = []
    result: list[ControlFinding] = []

    for finding in findings:
        waiver = active.get(finding.control_id)
        if (
            waiver is not None
            and finding.gate_result == "BLOCK"
            and finding.waiver_eligible
        ):
            result.append(
                finding.model_copy(
                    update={
                        "gate_result": "WAIVE",
                        "waiver_id": waiver.waiver_id,
                        "reason": (
                            f"WAIVED until {waiver.expires_at} by {waiver.owner}. The "
                            f"requirement remains unmet: {finding.reason} Accepted "
                            f"risk scope: {waiver.scope}."
                        ),
                        "recommended_action": (
                            "Track to the waiver expiry. Compensating controls: "
                            + "; ".join(waiver.compensating_controls)
                            + "."
                        ),
                    }
                )
            )
            applied.append(waiver)
        else:
            result.append(finding)
    return result, applied


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    audit_run_id: str,
    context: dict[str, Any],
    policy: PolicyProfile,
    waivers: list[Waiver] | None = None,
    evaluated_at: str | None = None,
) -> GateEvaluation:
    """
    Evaluate every control, then fold the results up into gate decisions.

    ``context`` carries the run's already-loaded evidence documents plus the evidence
    manifest, so this function performs no orchestration and no writes -- it is a pure
    function of (evidence, policy, waivers) apart from the ``evaluated_at`` stamp.
    """
    expected = _expected_checksums(context.get("evidence_manifest") or {})
    controls = {c.control_id: c for c in policy.controls}

    # Two passes: the Release Gate's risk-visibility control depends on whether the
    # Validation Gate blocked, which is only known once VG's controls are decided.
    findings: dict[str, ControlFinding] = {}

    def evidence_for(control: PolicyControl):
        return _check_artifact(audit_run_id, control.evidence_artifact, expected)

    simple_handlers = {
        "DG-01": lambda c: _eval_dg01(c, context, evidence_for(c)),
        "DG-02": lambda c: _eval_dg02(c, context, evidence_for(c)),
        "TG-01": lambda c: _eval_tg01(c, context, evidence_for(c)),
        "TG-02": lambda c: _eval_tg02(c, context, evidence_for(c)),
        "VG-01": lambda c: _eval_vg01(c, context, evidence_for(c), policy),
        "VG-02": lambda c: _eval_vg02(c, context, evidence_for(c), policy),
        "VG-03": lambda c: _eval_vg03(c, context, evidence_for(c)),
        "RG-01": lambda c: _eval_rg01(c, context, evidence_for(c)),
        "OG-01": lambda c: _eval_operations(c, context, evidence_for(c)),
        "OG-02": lambda c: _eval_operations(c, context, evidence_for(c)),
    }

    for control_id, handler in simple_handlers.items():
        control = controls.get(control_id)
        if control is not None:
            findings[control_id] = handler(control)

    validation_blocked = any(
        f.gate_result == "BLOCK" for f in findings.values() if f.gate == "VG"
    )
    if "RG-02" in controls:
        findings["RG-02"] = _eval_rg02(
            controls["RG-02"], context, evidence_for(controls["RG-02"]), validation_blocked
        )

    ordered = [findings[c.control_id] for c in policy.controls if c.control_id in findings]
    ordered, waivers_applied = apply_waivers(ordered, waivers or [])
    by_id = {f.control_id: f for f in ordered}

    # -- gate folding -------------------------------------------------------- #
    gate_results: list[GateResult] = []
    for gate in sorted(policy.gates, key=lambda g: g.order):
        members = [by_id[cid] for cid in gate.controls if cid in by_id]
        counts: dict[str, int] = {}
        for member in members:
            counts[member.gate_result] = counts.get(member.gate_result, 0) + 1

        blocking = [m for m in members if m.gate_result == "BLOCK"]
        if blocking:
            status = "BLOCK"
            reason = (
                f"{len(blocking)} control(s) did not meet the configured policy: "
                + ", ".join(m.control_id for m in blocking)
                + "."
            )
        elif gate.never_auto_pass:
            status = "NOT_EVALUATED"
            reason = (
                f"The {gate.gate_name} can never be passed by computation in this "
                "platform: it depends on human accountability and operational evidence "
                "this prototype does not hold. This is not a pass and not a failure."
            )
        elif any(m.gate_result == "NOT_EVALUATED" for m in members):
            unassessed = [
                m.control_id for m in members if m.gate_result == "NOT_EVALUATED"
            ]
            status = "NOT_EVALUATED"
            reason = (
                "Some controls could not be assessed with the evidence supplied: "
                + ", ".join(unassessed)
                + ". Absent evidence is not treated as a pass."
            )
        elif members and any(m.gate_result in ("PASS", "WAIVE") for m in members):
            waived = [m.control_id for m in members if m.gate_result == "WAIVE"]
            status = "PASS"
            reason = (
                f"All {len(members)} control(s) met the configured policy"
                + (
                    f", with {', '.join(waived)} covered by an explicit time-bounded "
                    "waiver rather than satisfied."
                    if waived
                    else "."
                )
            )
        else:
            status = "NOT_EVALUATED"
            reason = "No controls were evaluated for this gate."

        gate_results.append(
            GateResult(
                gate_code=gate.gate_code,
                gate_name=gate.gate_name,
                order=gate.order,
                owner=gate.owner,
                question=gate.question,
                status=status,  # type: ignore[arg-type]
                reason=reason,
                never_auto_pass=gate.never_auto_pass,
                control_ids=[m.control_id for m in members],
                status_counts=counts,
            )
        )

    status_counts: dict[str, int] = {}
    for finding in ordered:
        status_counts[finding.gate_result] = status_counts.get(finding.gate_result, 0) + 1

    evidence_coverage, control_coverage = coverage_scores(ordered, policy)
    fairness_finding = by_id.get("VG-02")

    return GateEvaluation(
        audit_run_id=audit_run_id,
        policy_profile_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_checksum=policy.checksum,
        evaluated_at=evaluated_at or _utc_now(),
        gates=gate_results,
        controls=ordered,
        gate_summary={g.gate_code: g.status for g in gate_results},
        status_counts=status_counts,
        blocking_controls=[f.control_id for f in ordered if f.gate_result == "BLOCK"],
        evidence_coverage_score=evidence_coverage,
        control_coverage_score=control_coverage,
        waivers_applied=waivers_applied,
        fairness_gate_notice=(
            FAIRNESS_GATE_NOTICE
            if fairness_finding is not None
            and fairness_finding.gate_result in ("BLOCK", "WAIVE", "PASS")
            else None
        ),
        release_gate_note=str(
            policy.never_auto_pass.get(
                "reason",
                "Release and Operations gates are never automatically passed.",
            )
        ),
        limitations=policy.limitations,
    )


def coverage_scores(
    findings: list[ControlFinding], policy: PolicyProfile
) -> tuple[float | None, float | None]:
    """
    The two governance coverage metrics.

    * **evidence coverage** = verified required evidence / total required evidence,
      counting only controls that both require evidence and name an artefact. A
      control whose evidence cannot exist in this prototype (RG-01, OG-01, OG-02) has
      no artefact to verify and is excluded from the denominator rather than counted
      as a permanent failure.
    * **control coverage** = evaluated applicable controls / total applicable
      controls, where "evaluated" means the result is not ``NOT_EVALUATED``.

    Both are *coverage* measures: how much of the policy could be assessed with the
    evidence supplied. Neither is a compliance percentage and neither confers
    certification.
    """
    by_id = {f.control_id: f for f in findings}

    required = [
        c for c in policy.controls if c.evidence_required and c.evidence_artifact
    ]
    verified = sum(
        1
        for c in required
        if by_id.get(c.control_id)
        and by_id[c.control_id].evidence_status == "verified"
    )
    evidence_coverage = (verified / len(required)) if required else None

    applicable = [c for c in policy.controls if c.control_id in by_id]
    evaluated = sum(
        1 for c in applicable if by_id[c.control_id].gate_result != "NOT_EVALUATED"
    )
    control_coverage = (evaluated / len(applicable)) if applicable else None
    return (
        round(evidence_coverage, 4) if evidence_coverage is not None else None,
        round(control_coverage, 4) if control_coverage is not None else None,
    )


# --------------------------------------------------------------------------- #
# Governance state
# --------------------------------------------------------------------------- #
def derive_governance_state(
    evaluation: GateEvaluation,
) -> tuple[str, str, list[str]]:
    """
    Fold a gate evaluation into one of three states, none of which is an approval.

    Precedence is deliberate: a policy block outranks missing evidence, because "we
    measured this and it failed the configured threshold" is a stronger statement
    than "we could not measure it". The best reachable state is
    ``review_required`` -- a human must still decide.
    """
    grounds: list[str] = []
    blocked = [g for g in evaluation.gates if g.status == "BLOCK"]
    if blocked:
        for gate in blocked:
            grounds.append(f"{gate.gate_code} ({gate.gate_name}): {gate.reason}")
        grounds.append(
            "A blocked gate is a configured research-policy result. It is not a legal "
            "conclusion, not proof of discrimination, and not a statement that the "
            "model is unlawful."
        )
        return (
            "blocked_by_policy",
            "One or more gates did not meet the configured Research Governance Policy. "
            "This is deterministic decision-support evidence for a human reviewer, not "
            "a legal finding.",
            grounds,
        )

    unassessed = [
        f
        for f in evaluation.controls
        if f.gate_result == "NOT_EVALUATED" and f.gate in ASSESSABLE_GATES
    ]
    if unassessed:
        for finding in unassessed:
            grounds.append(f"{finding.control_id} ({finding.title}): {finding.reason}")
        grounds.append(
            "Absent evidence is reported as absent. It was not treated as a pass and "
            "no value was estimated to fill the gap."
        )
        return (
            "insufficient_evidence",
            "The evidence supplied was not sufficient to assess every applicable "
            "control. This is neither a pass nor a failure.",
            grounds,
        )

    grounds.append(
        "Every assessable control in the Data, Training and Validation gates met the "
        "configured policy."
    )
    grounds.append(
        "The Release and Operations gates remain NOT_EVALUATED by design: they require "
        "human authorisation and operational evidence this prototype does not hold."
    )
    grounds.append(
        "No deployment authorisation is granted or implied. A human reviewer must make "
        "the decision."
    )
    return (
        "review_required",
        "All assessable controls met the configured research policy. A human reviewer "
        "must now decide; this platform does not approve deployment.",
        grounds,
    )
