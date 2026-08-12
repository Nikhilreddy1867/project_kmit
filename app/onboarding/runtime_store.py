"""
app/onboarding/runtime_store.py
===============================
The only writable storage in MAAT: the gitignored ``runtime/`` tree.

Layout
------
::

    runtime/
      governance_registry.db          SQLite registry (Phase 8, unchanged)
      uploads/<upload_id>/            one directory per submitted upload bundle
        model_<uuid>.joblib           generated names -- never the user's filename
        dataset_<uuid>.csv
        upload_manifest.json
      audits/<audit_run_id>/          one directory per uploaded-model audit run
        manifest.json  performance.json  fairness.json  explainability.json
        risk_summary.json  governance_summary.json  gate_evaluation.json
        conformity_bundle.json  traceability.json  evidence_manifest.json
        predictions.csv
        uploaded_model_metadata.json  uploaded_dataset_metadata.json

Guarantees this module enforces
-------------------------------
* **Nothing is ever written outside ``runtime/``.** Every path passes through
  :func:`~app.onboarding.security.assert_within` before it is opened.
* **No upload overwrites an existing file.** Writes use ``"xb"`` / ``"x"``
  (exclusive create), so a name collision raises instead of clobbering. Combined
  with UUID filenames this makes overwriting an existing artefact impossible
  rather than merely unlikely.
* **The immutable evidence tree is untouchable from here.** There is no code path
  in this module that can produce a path under ``data/``, ``models/``,
  ``predictions/`` or ``results/``; :func:`assert_not_evidence` asserts it.

Checksums are computed with the Phase 8 helper so uploaded evidence is hashed by
exactly the same code as the reference case.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from app.onboarding.security import UploadRejected, assert_within
from app.registry.integrity import sha256_file

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
RUNTIME_DIR: Final = PROJECT_ROOT / "runtime"
UPLOADS_DIR: Final = RUNTIME_DIR / "uploads"
AUDITS_DIR: Final = RUNTIME_DIR / "audits"

#: Directories that are read-only evidence for the built-in reference case.
IMMUTABLE_DIRS: Final[tuple[str, ...]] = ("data", "models", "predictions", "results")

#: The artefact files an uploaded audit run is expected to produce, in the order
#: they are listed in the run manifest.
AUDIT_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "manifest.json",
    "uploaded_model_metadata.json",
    "uploaded_dataset_metadata.json",
    "predictions.csv",
    "performance.json",
    "fairness.json",
    "explainability.json",
    "risk_summary.json",
    "governance_summary.json",
    "gate_evaluation.json",
    "conformity_bundle.json",
    "traceability.json",
    "evidence_manifest.json",
)


def utc_now() -> str:
    """Current UTC time, ISO-8601, second precision (matches the registry)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def assert_not_evidence(path: Path) -> None:
    """
    Refuse any path that falls inside the immutable evidence tree.

    Redundant with :func:`assert_within` (``runtime/`` and ``results/`` are
    disjoint), and kept deliberately: it states the invariant that matters most in
    the place a future change would be most likely to break it.
    """
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return  # outside the repo entirely (e.g. a temp dir in tests)
    if relative.parts and relative.parts[0] in IMMUTABLE_DIRS:
        raise UploadRejected(
            "immutable_evidence",
            f"Refusing to write to {relative.as_posix()}: data/, models/, "
            "predictions/ and results/ are immutable audit evidence.",
        )


def relative_path(path: Path) -> str:
    """Repo-relative POSIX path, for manifests and checksum records."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def upload_dir(upload_id: str, create: bool = False) -> Path:
    """Directory for one upload bundle."""
    target = assert_within(UPLOADS_DIR / upload_id, RUNTIME_DIR)
    assert_not_evidence(target)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def audit_dir(audit_run_id: str, create: bool = False) -> Path:
    """Directory for one uploaded-model audit run."""
    target = assert_within(AUDITS_DIR / audit_run_id, RUNTIME_DIR)
    assert_not_evidence(target)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def list_audit_ids() -> list[str]:
    """Audit-run ids present on disk, newest first by directory mtime."""
    if not AUDITS_DIR.is_dir():
        return []
    entries = [p for p in AUDITS_DIR.iterdir() if p.is_dir()]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in entries]


# --------------------------------------------------------------------------- #
# Writes -- exclusive-create only
# --------------------------------------------------------------------------- #
def write_bytes(target: Path, payload: bytes) -> Path:
    """
    Write bytes, refusing to overwrite anything.

    ``"xb"`` fails with :class:`FileExistsError` if the path exists, which is
    translated into a policy error rather than allowed to surface as a 500: an
    upload that would overwrite an existing file is a rejected upload, not a bug.
    """
    resolved = assert_within(target, RUNTIME_DIR)
    assert_not_evidence(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise UploadRejected(
            "would_overwrite",
            f"Refusing to overwrite the existing file {relative_path(resolved)}.",
        ) from exc
    return resolved


def write_json(target: Path, payload: Any, *, allow_replace: bool = False) -> Path:
    """
    Write one JSON artefact.

    ``allow_replace`` exists for exactly one case: re-evaluating the policy gates
    for an existing audit run rewrites ``gate_evaluation.json`` and its
    downstream bundle. That is a deliberate, idempotent recomputation of a
    *generated* artefact inside the run's own directory -- never an upload, and
    never anything under the immutable evidence tree. It defaults to ``False`` so
    the safe behaviour is the one you get by accident.
    """
    resolved = assert_within(target, RUNTIME_DIR)
    assert_not_evidence(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    if allow_replace:
        resolved.write_text(text, encoding="utf-8")
        return resolved
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise UploadRejected(
            "would_overwrite",
            f"Refusing to overwrite the existing artefact {relative_path(resolved)}.",
        ) from exc
    return resolved


def write_text(target: Path, text: str) -> Path:
    """Write a text artefact (CSV predictions), refusing to overwrite."""
    return write_bytes(target, text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
class AuditRunNotFound(Exception):
    """No audit run with that id exists under ``runtime/audits/``."""

    def __init__(self, audit_run_id: str, available: list[str] | None = None):
        self.audit_run_id = audit_run_id
        self.available = available or []
        super().__init__(
            f"No uploaded-model audit run '{audit_run_id}' exists in runtime/audits/."
        )


class AuditArtifactMissing(Exception):
    """The run exists but a requested artefact was not produced."""

    def __init__(self, audit_run_id: str, name: str):
        self.audit_run_id = audit_run_id
        self.name = name
        super().__init__(
            f"Audit run '{audit_run_id}' has no artefact '{name}'. It may have been "
            "created by an older version, or the audit did not complete."
        )


def read_json(audit_run_id: str, name: str) -> Any:
    """Read one artefact from an audit run, with specific errors for each failure."""
    directory = audit_dir(audit_run_id)
    if not directory.is_dir():
        raise AuditRunNotFound(audit_run_id, list_audit_ids())
    target = directory / name
    if not target.is_file():
        raise AuditArtifactMissing(audit_run_id, name)
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditArtifactMissing(audit_run_id, f"{name} (unreadable: {exc})") from exc


def artifact_path(audit_run_id: str, name: str) -> Path:
    return audit_dir(audit_run_id) / name


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
def checksum_record(path: Path, group: str) -> dict[str, Any]:
    """
    One evidence-manifest entry: repo-relative path, SHA-256, size and mtime.

    Uses the Phase 8 :func:`sha256_file`, so uploaded evidence is hashed by the
    same streaming implementation as the reference case rather than a parallel
    one that could diverge.
    """
    stat = path.stat()
    return {
        "group": group,
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        .isoformat(timespec="seconds"),
    }
