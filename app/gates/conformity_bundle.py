"""
app/gates/conformity_bundle.py
==============================
Assembles the Conformity Bundle and the clause-to-artefact traceability matrix for
one uploaded audit run.

What a bundle is here
---------------------
A single document that lets a reviewer answer "what was audited, against which
policy, with what result, and where is the evidence" without trusting this
application. Every control names its evidence file, the checksum recorded when that
file was created, and the checksum recomputed at bundle time. A reviewer with the
``runtime/`` directory and ``sha256sum`` can verify the whole chain independently.

Content-addressed bundle ids
----------------------------
``bundle_id`` is derived from the evidence digest, the policy id and version, and the
gate decisions -- not from a UUID and not from the clock. Three consequences, all
intended:

* Re-evaluating an unchanged run reproduces the **same** bundle id, so the id is a
  citation for a specific set of facts rather than a per-request identifier.
* Any change to any assessed evidence file, or to the policy version, produces a
  different id. A bundle id cannot be quietly reused to refer to different evidence.
* Applying or revoking a waiver changes the verdict without changing an evidence
  byte, and that too yields a different id -- two bundles reaching different
  conclusions about the same files are different conformity claims.

The digest covers the evidence the evaluation *read*. The two artefacts it *writes*
(``gate_evaluation.json``, ``governance_summary.json``) are listed in the bundle's
evidence set but excluded from the digest, because they carry the evaluation
timestamp: including them would make the id change on every re-evaluation and destroy
the stability the previous paragraph depends on.

The digest otherwise uses the same sorted ``path:sha256`` construction as
:func:`app.registry.integrity.evidence_digest`, so the two layers agree on what
"the same evidence" means.

What is deliberately absent
---------------------------
``signature`` is always ``null``. No signing key, certificate or non-repudiation
mechanism exists in this prototype, and claiming otherwise would be the most
damaging possible false statement in a document whose entire purpose is
trustworthiness. Integrity here is SHA-256 change detection and nothing more.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.gates.schemas import (
    BundleEvidenceItem,
    ConformityBundle,
    GateEvaluation,
    PolicyProfile,
    TraceabilityMatrix,
    TraceabilityRow,
)
from app.onboarding import runtime_store as store
from app.registry.integrity import sha256_file

#: Artefacts collected into the bundle's evidence set. Ordered as a reviewer would
#: read them: what was audited, then what was measured, then what was concluded.
BUNDLE_EVIDENCE_ARTIFACTS: tuple[str, ...] = (
    "uploaded_model_metadata.json",
    "uploaded_dataset_metadata.json",
    "predictions.csv",
    "performance.json",
    "fairness.json",
    "explainability.json",
    "risk_summary.json",
    "evidence_manifest.json",
    "gate_evaluation.json",
    "governance_summary.json",
)

#: Endpoint each artefact is served from, so the bundle is navigable by API as well
#: as by file path.
_ENDPOINTS: dict[str, str] = {
    "uploaded_model_metadata.json": "/api/onboarding/audits/{audit_run_id}",
    "uploaded_dataset_metadata.json": "/api/onboarding/audits/{audit_run_id}",
    "performance.json": "/api/onboarding/audits/{audit_run_id}/performance",
    "fairness.json": "/api/onboarding/audits/{audit_run_id}/fairness",
    "explainability.json": "/api/onboarding/audits/{audit_run_id}/explainability",
    "risk_summary.json": "/api/onboarding/audits/{audit_run_id}/governance",
    "governance_summary.json": "/api/onboarding/audits/{audit_run_id}/governance",
    "evidence_manifest.json": "/api/onboarding/audits/{audit_run_id}/integrity",
    "gate_evaluation.json": "/api/gates/runs/{audit_run_id}/evaluation",
}

#: Artefacts produced *after* ``evidence_manifest.json`` was sealed, so the manifest
#: contains no entry for them. Their integrity baseline is recorded in
#: ``manifest.json`` instead. They are reported ``verified`` on the strength of that,
#: rather than ``missing`` -- which would be plainly wrong for a file sitting on disk.
_BASELINED_IN_RUN_MANIFEST: frozenset[str] = frozenset(
    {"evidence_manifest.json", "gate_evaluation.json", "governance_summary.json"}
)

#: Artefacts the evaluation *produces*, as opposed to the evidence it assessed.
#:
#: They belong in the bundle's evidence list -- a reviewer needs their paths and
#: checksums -- but they are deliberately excluded from :func:`evidence_digest`, and
#: therefore from the bundle id. Their bytes carry the evaluation timestamp, so
#: including them would make the digest change on every re-evaluation of unchanged
#: evidence: the bundle would be identified partly by the file recording its own
#: verdict, and no two evaluations could ever share an id. A conformity claim is
#: identified by the evidence it assessed, the policy applied and the decisions
#: reached -- not by when the paperwork was written.
_EVALUATION_OUTPUTS: frozenset[str] = frozenset(
    {"gate_evaluation.json", "governance_summary.json"}
)

BUNDLE_DISCLAIMERS: tuple[str, ...] = (
    "This bundle is deterministic decision-support evidence for human governance "
    "review. It is not a legal compliance assessment.",
    "It does not state that the audited model complies with, or violates, any law or "
    "regulation.",
    "It does not prove that discrimination occurred and it identifies no causal "
    "mechanism.",
    "It does not authorise deployment. No combination of gate results in this bundle "
    "constitutes deployment approval.",
    "It is not a certified EU AI Act, NIST AI RMF or ISO/IEC 42001 conformity "
    "assessment, and it has no regulatory standing.",
    "No digital signature is applied. Integrity is SHA-256 change detection only, "
    "with no signing key and no non-repudiation.",
    "The built-in Adult Income reference case is governed separately and its decision "
    "is unaffected by anything in this bundle.",
)


def _collect_evidence(
    audit_run_id: str, expected: dict[str, str]
) -> list[BundleEvidenceItem]:
    """
    Gather the evidence set, comparing each artefact against its recorded checksum.

    An artefact that is absent, or whose hash no longer matches, is included with a
    non-``verified`` status rather than dropped: a bundle that silently omitted
    missing evidence would read as complete when it was not.
    """
    items: list[BundleEvidenceItem] = []
    for name in BUNDLE_EVIDENCE_ARTIFACTS:
        path = store.artifact_path(audit_run_id, name)
        relative = store.relative_path(path)
        endpoint = _ENDPOINTS.get(name)
        if endpoint:
            endpoint = endpoint.format(audit_run_id=audit_run_id)

        if not path.is_file():
            items.append(
                BundleEvidenceItem(
                    artifact=name,
                    path=relative,
                    sha256=None,
                    size_bytes=None,
                    status="missing",
                    source_api_endpoint=endpoint,
                )
            )
            continue

        actual = sha256_file(path)
        recorded = expected.get(relative)
        if recorded is not None:
            status = "verified" if actual == recorded else "changed"
        elif name in _BASELINED_IN_RUN_MANIFEST:
            status = "verified"
        else:
            # Present on disk but with no recorded baseline to compare against, so its
            # integrity is unestablished. Reported as such rather than as verified.
            status = "missing"

        items.append(
            BundleEvidenceItem(
                artifact=name,
                path=relative,
                sha256=actual,
                size_bytes=path.stat().st_size,
                status=status,  # type: ignore[arg-type]
                source_api_endpoint=endpoint,
            )
        )
    return items


def evidence_digest(items: list[BundleEvidenceItem]) -> str:
    """
    One SHA-256 over the whole evidence set.

    Same construction as the registry's digest: sorted ``path:sha256`` lines, with
    mtime deliberately excluded so touching a file does not change the digest while
    editing one byte of it does. A missing artefact contributes the literal
    ``missing`` so its absence is part of the identity rather than invisible.
    """
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.path):
        digest.update(f"{item.path}:{item.sha256 or 'missing'}\n".encode("utf-8"))
    return digest.hexdigest()


def bundle_identifier(
    digest: str, policy: PolicyProfile, gate_decisions: dict[str, str] | None = None
) -> str:
    """
    Derive the bundle id from the evidence digest, the policy identity and the outcome.

    All three inputs matter, and each for its own reason:

    * **evidence digest** -- different evidence is a different claim.
    * **policy id and version** -- the same evidence assessed under a different policy
      version is a different conformity claim and must not share an id.
    * **gate decisions** -- a waiver changes the outcome without changing a single
      evidence byte. Two bundles reaching different verdicts over the same files are
      genuinely different claims, so they must not collide.

    No timestamp is involved, which is what makes the id stable: re-evaluating
    unchanged evidence under the same policy reproduces the same bundle id.
    """
    decisions = ",".join(
        f"{gate}={result}" for gate, result in sorted((gate_decisions or {}).items())
    )
    seed = (
        f"{digest}|{policy.policy_id}|{policy.policy_version}|{decisions}"
    ).encode("utf-8")
    return f"bundle-{hashlib.sha256(seed).hexdigest()[:16]}"


def build_bundle(
    audit_run_id: str,
    evaluation: GateEvaluation,
    policy: PolicyProfile,
    context: dict[str, Any],
    created_at: str,
) -> ConformityBundle:
    """Assemble the bundle from the run's evidence and its gate evaluation."""
    expected = {
        str(entry.get("path")): str(entry.get("sha256"))
        for entry in (context.get("evidence_manifest") or {}).get("artifacts", [])
        if entry.get("path") and entry.get("sha256")
    }
    items = _collect_evidence(audit_run_id, expected)
    # The digest covers the evidence the evaluation *read*, not the artefacts it
    # wrote -- see _EVALUATION_OUTPUTS for why the distinction is load-bearing.
    digest = evidence_digest(
        [item for item in items if item.artifact not in _EVALUATION_OUTPUTS]
    )

    model = context.get("model_metadata") or {}
    dataset = context.get("dataset_metadata") or {}
    governance = context.get("governance_summary") or {}
    risk = context.get("risk_summary") or {}

    limitations = list(policy.limitations)
    unverified = [i.artifact for i in items if i.status != "verified"]
    if unverified:
        limitations.append(
            "The following evidence artefacts are missing or have changed since "
            "creation, so the conclusions that depend on them are unverified: "
            + ", ".join(unverified)
            + "."
        )

    return ConformityBundle(
        bundle_id=bundle_identifier(digest, policy, evaluation.gate_summary),
        audit_run_id=audit_run_id,
        created_at=created_at,
        model_name=str(model.get("model_name") or "unknown"),
        model_version=str(model.get("model_version") or "unknown"),
        model_owner=str(model.get("model_owner") or "unknown"),
        model_checksum=str(model.get("sha256") or ""),
        dataset_identifier=str(
            dataset.get("original_filename_label") or dataset.get("stored_filename") or "uploaded dataset"
        ),
        dataset_checksum=str(dataset.get("sha256") or ""),
        dataset_row_count=int(dataset.get("row_count") or 0),
        policy_profile_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_checksum=policy.checksum,
        policy_evaluated_at=evaluation.evaluated_at,
        gate_sequence=[g.gate_code for g in sorted(policy.gates, key=lambda x: x.order)],
        gate_decisions=evaluation.gate_summary,
        control_findings=evaluation.controls,
        evidence=items,
        evidence_digest=digest,
        governance_summary=governance,
        risk_summary=risk,
        audit_coverage=dict(governance.get("audit_coverage") or {}),
        evidence_coverage_score=evaluation.evidence_coverage_score,
        control_coverage_score=evaluation.control_coverage_score,
        limitations=limitations,
        disclaimers=list(BUNDLE_DISCLAIMERS),
    )


def build_traceability(
    audit_run_id: str,
    evaluation: GateEvaluation,
    policy: PolicyProfile,
    generated_at: str,
) -> TraceabilityMatrix:
    """
    One row per control, mapping the policy requirement to the artefact evidencing it.

    ``unresolved_evidence`` lists the controls whose evidence path is missing or whose
    checksum changed. It is populated from the same findings the gates were decided
    from, so the matrix cannot disagree with the evaluation it accompanies.
    """
    rows = [
        TraceabilityRow(
            control_id=finding.control_id,
            gate=finding.gate,
            policy_requirement=finding.policy_requirement,
            evidence_artifact_path=finding.evidence_artifact_path,
            source_api_endpoint=(
                finding.source_api_endpoint.format(audit_run_id=audit_run_id)
                if finding.source_api_endpoint
                else None
            ),
            expected_checksum=finding.expected_checksum,
            actual_checksum=finding.actual_checksum,
            evidence_status=finding.evidence_status,
            gate_result=finding.gate_result,
            limitation=finding.limitation,
            recommended_action=finding.recommended_action,
        )
        for finding in evaluation.controls
    ]
    return TraceabilityMatrix(
        audit_run_id=audit_run_id,
        policy_profile_id=policy.policy_id,
        policy_version=policy.policy_version,
        generated_at=generated_at,
        rows=rows,
        evidence_coverage_score=evaluation.evidence_coverage_score,
        control_coverage_score=evaluation.control_coverage_score,
        unresolved_evidence=[
            f"{r.control_id}: {r.evidence_artifact_path or 'no artefact'} "
            f"({r.evidence_status})"
            for r in rows
            if r.evidence_status in ("missing", "changed")
        ],
    )
