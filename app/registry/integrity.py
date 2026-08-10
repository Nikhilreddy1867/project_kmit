"""
app/registry/integrity.py
=========================
Artifact discovery and SHA-256 evidence integrity.

This module is the only place checksums are computed. It is deliberately
**root-parameterisable**: every function takes a ``root`` so the integrity logic
can be exercised against a temporary copy of an artefact tree in tests, without
touching the repository's real evidence. Nothing here writes to disk.

What counts as a referenced artefact
------------------------------------
:data:`ARTIFACT_GROUPS` declares the registered evidence set as glob patterns per
audit phase. It covers the raw dataset snapshot, the fitted pipelines, the test
predictions, and every published output under ``results/`` -- including the PNG
figures, because they are published governance outputs and a reader who cites a
chart deserves to know whether it still matches the CSV it was drawn from.

Files matched by a pattern but absent from disk are simply not discovered; a run
records what existed at registration time, and the integrity check later reports
anything that has since gone missing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Glob patterns per phase, relative to the project root. Order is fixed so a
# registered artefact list is stable and diffable.
ARTIFACT_GROUPS: dict[str, tuple[str, ...]] = {
    "dataset": (
        "data/raw/adult_raw.csv",
        "data/raw/adult_metadata.txt",
    ),
    "models": ("models/*.joblib",),
    "predictions": ("predictions/*_test_predictions.csv",),
    "performance": (
        "results/model_metrics.csv",
        "results/classification_report_*.txt",
        "results/confusion_matrix_*.png",
        "results/model_comparison.png",
    ),
    "fairness": (
        "results/fairness/*.csv",
        "results/fairness/*.md",
        "results/fairness/*.png",
    ),
    "explainability": (
        "results/explainability/*.csv",
        "results/explainability/*.md",
        "results/explainability/*.png",
    ),
    "governance": (
        "results/governance/*.csv",
        "results/governance/*.md",
    ),
}

CHUNK_SIZE = 1024 * 1024  # 1 MiB; the largest artefact is a ~100 MB joblib


@dataclass(frozen=True)
class ArtifactRecord:
    """One registered artefact and its checksum at registration time."""

    group: str
    path: str  # repo-relative, POSIX separators
    sha256: str
    size_bytes: int
    modified_utc: str


@dataclass(frozen=True)
class ArtifactVerification:
    """The result of re-checking one registered artefact."""

    group: str
    path: str
    status: str  # "verified" | "missing" | "changed"
    expected_sha256: str
    actual_sha256: str | None
    expected_size_bytes: int
    actual_size_bytes: int | None
    detail: str


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Stream a file through SHA-256. Read-only; never loads it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_mtime(path: Path) -> str:
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        .isoformat(timespec="seconds")
    )


def discover_artifacts(root: Path = PROJECT_ROOT) -> list[ArtifactRecord]:
    """
    Find and checksum every referenced artefact under ``root``.

    Results are sorted by (group order, path) so the artefact list -- and
    therefore the evidence digest derived from it -- is deterministic regardless
    of filesystem ordering.
    """
    records: list[ArtifactRecord] = []
    for group, patterns in ARTIFACT_GROUPS.items():
        matched: set[Path] = set()
        for pattern in patterns:
            matched.update(p for p in root.glob(pattern) if p.is_file())
        for path in sorted(matched):
            records.append(
                ArtifactRecord(
                    group=group,
                    path=path.relative_to(root).as_posix(),
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                    modified_utc=_iso_mtime(path),
                )
            )
    return records


def evidence_digest(records: Iterable[ArtifactRecord]) -> str:
    """
    A single SHA-256 over the whole evidence set.

    Computed from the sorted ``path:sha256`` pairs, so it changes if any artefact
    changes, is added, or is removed -- but not merely because a file was touched
    (mtime is excluded on purpose). This digest is what makes a registry run
    **content-addressed**: identical evidence yields the identical run id, which
    is what makes registration idempotent.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.path):
        digest.update(f"{record.path}:{record.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_artifacts(
    registered: Iterable[ArtifactRecord], root: Path = PROJECT_ROOT
) -> list[ArtifactVerification]:
    """
    Recompute each registered artefact's checksum and classify the outcome.

    * ``verified`` -- present and the digest matches what was registered.
    * ``missing``  -- the registered path no longer exists.
    * ``changed``  -- present but the digest differs from the registered value.

    A size difference alone is not a separate category: any real content change
    shows up in the digest, and the sizes are reported for context.
    """
    results: list[ArtifactVerification] = []
    for record in registered:
        target = root / record.path
        if not target.is_file():
            results.append(
                ArtifactVerification(
                    group=record.group,
                    path=record.path,
                    status="missing",
                    expected_sha256=record.sha256,
                    actual_sha256=None,
                    expected_size_bytes=record.size_bytes,
                    actual_size_bytes=None,
                    detail="Registered artefact is no longer present on disk.",
                )
            )
            continue

        actual = sha256_file(target)
        actual_size = target.stat().st_size
        if actual == record.sha256:
            results.append(
                ArtifactVerification(
                    group=record.group,
                    path=record.path,
                    status="verified",
                    expected_sha256=record.sha256,
                    actual_sha256=actual,
                    expected_size_bytes=record.size_bytes,
                    actual_size_bytes=actual_size,
                    detail="Checksum matches the registered value.",
                )
            )
        else:
            results.append(
                ArtifactVerification(
                    group=record.group,
                    path=record.path,
                    status="changed",
                    expected_sha256=record.sha256,
                    actual_sha256=actual,
                    expected_size_bytes=record.size_bytes,
                    actual_size_bytes=actual_size,
                    detail=(
                        "Content differs from the registered checksum. The evidence "
                        "this run was registered against is no longer what is on disk."
                    ),
                )
            )
    return results


def summarise_verification(results: list[ArtifactVerification]) -> dict:
    """
    Roll per-file outcomes into an overall integrity status.

    * ``verified``               -- every artefact matches.
    * ``incomplete``             -- some artefacts are gone, none altered.
    * ``modified``               -- some artefacts were altered.
    * ``modified_and_incomplete``-- both.

    ``integrity_ok`` is true only for ``verified``. A missing file is not treated
    as benign: a run whose evidence cannot be produced is not a verifiable run.
    """
    verified = [r for r in results if r.status == "verified"]
    missing = [r for r in results if r.status == "missing"]
    changed = [r for r in results if r.status == "changed"]

    if changed and missing:
        status = "modified_and_incomplete"
    elif changed:
        status = "modified"
    elif missing:
        status = "incomplete"
    else:
        status = "verified"

    return {
        "integrity_status": status,
        "integrity_ok": status == "verified",
        "artifacts_checked": len(results),
        "verified_count": len(verified),
        "missing_count": len(missing),
        "changed_count": len(changed),
        "verified_files": [r.path for r in verified],
        "missing_files": [r.path for r in missing],
        "changed_files": [r.path for r in changed],
    }
