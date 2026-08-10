"""
artifact_reader.py
==================
The **only** component in the API that touches the filesystem.

Design contract (enforced by convention and by the tests)
--------------------------------------------------------
1. **Read-only.** Files are opened for reading exclusively. Nothing in the `app/`
   package writes to `data/`, `models/`, `predictions/` or `results/`.
2. **No recomputation.** Values are passed through **verbatim** from the audit
   CSVs. This module performs no arithmetic on any metric -- no re-deriving a
   rate, no rounding, no aggregation of a metric. The audits in `src/` are the
   single source of truth, so the API cannot drift from them.
   The only counting done anywhere is over *rows of the risk register*
   (how many risks are rated Critical), which is metadata about the register,
   not a model metric.
3. **mtime-based caching.** Each artefact is cached and invalidated when its
   modification time changes, so editing a file under `results/` is picked up
   without restarting the server, while repeat requests do not re-read the disk.
4. **Typed failures.** Missing or malformed artefacts raise the exceptions below,
   which `main.py` maps to meaningful HTTP status codes.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Repo root = two levels up from app/services/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FAIRNESS_DIR = RESULTS_DIR / "fairness"
EXPLAIN_DIR = RESULTS_DIR / "explainability"
GOVERNANCE_DIR = RESULTS_DIR / "governance"

# Canonical artefact locations. Every path the API can read is declared here.
ARTIFACTS: dict[str, Path] = {
    "phase1_metrics": RESULTS_DIR / "model_metrics.csv",
    "fairness_by_group": FAIRNESS_DIR / "fairness_metrics_by_group.csv",
    "fairness_summary": FAIRNESS_DIR / "fairness_summary.csv",
    "fairness_report": FAIRNESS_DIR / "fairness_report.md",
    "global_importance": EXPLAIN_DIR / "global_feature_importance.csv",
    "local_explanations": EXPLAIN_DIR / "local_explanations.csv",
    "lr_coefficients": EXPLAIN_DIR / "logistic_regression_coefficients.csv",
    "explainability_report": EXPLAIN_DIR / "explainability_report.md",
    "risk_register": GOVERNANCE_DIR / "governance_risk_register.csv",
    "governance_summary": GOVERNANCE_DIR / "governance_summary.md",
    "model_card": GOVERNANCE_DIR / "model_card.md",
}

# The primary model of the Phase 3 audit. Explainability artefacts cover the
# XGBoost pipeline and the Logistic Regression comparison only -- Random Forest
# was never audited for explainability, and the API says so rather than
# inventing numbers for it.
EXPLAINED_MODELS = ("xgboost", "logistic_regression")
PRIMARY_MODEL = "xgboost"


# --------------------------------------------------------------------------- #
# Typed failures
# --------------------------------------------------------------------------- #
class ArtifactError(Exception):
    """Base class for artefact problems."""


class ArtifactMissingError(ArtifactError):
    """A required audit artefact is not on disk (the pipeline has not been run)."""

    def __init__(self, key: str, path: Path, how_to_fix: str):
        self.key, self.path, self.how_to_fix = key, path, how_to_fix
        super().__init__(f"Missing audit artefact '{key}' at {path}. {how_to_fix}")


class ArtifactMalformedError(ArtifactError):
    """An artefact exists but does not have the expected structure."""

    def __init__(self, key: str, detail: str):
        self.key, self.detail = key, detail
        super().__init__(f"Audit artefact '{key}' is malformed: {detail}")


class ModelNotFoundError(ArtifactError):
    """The requested model is not present in a given artefact."""

    def __init__(self, model: str, available: list[str], scope: str):
        self.model, self.available, self.scope = model, available, scope
        super().__init__(
            f"Model '{model}' has no {scope} data. Available: {', '.join(available)}."
        )


# How to regenerate each artefact -- surfaced in the HTTP error so the message is
# actionable rather than just "500".
_REGEN_HINT = {
    "phase1_metrics": "Run: python src/train.py",
    "fairness_by_group": "Run: python src/fairness_audit.py",
    "fairness_summary": "Run: python src/fairness_audit.py",
    "fairness_report": "Run: python src/fairness_audit.py",
    "global_importance": "Run: python src/explainability_audit.py",
    "local_explanations": "Run: python src/explainability_audit.py",
    "lr_coefficients": "Run: python src/explainability_audit.py",
    "explainability_report": "Run: python src/explainability_audit.py",
    "risk_register": "Phase 4 artefact; expected to be committed in results/governance/.",
    "governance_summary": "Phase 4 artefact; expected to be committed in results/governance/.",
    "model_card": "Phase 4 artefact; expected to be committed in results/governance/.",
}


# --------------------------------------------------------------------------- #
# Caching layer
# --------------------------------------------------------------------------- #
@dataclass
class _CacheEntry:
    mtime_ns: int
    payload: Any


_cache: dict[str, _CacheEntry] = {}
_lock = threading.Lock()  # uvicorn may serve concurrently; keep cache writes atomic


def _resolve(key: str) -> Path:
    if key not in ARTIFACTS:
        raise ArtifactMalformedError(key, "unknown artefact key")
    path = ARTIFACTS[key]
    if not path.exists():
        raise ArtifactMissingError(key, path, _REGEN_HINT.get(key, ""))
    return path


def _load(key: str, loader) -> Any:
    """Return the cached payload, re-reading only when the file's mtime changed."""
    path = _resolve(key)
    mtime = path.stat().st_mtime_ns
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit.mtime_ns == mtime:
            return hit.payload
    payload = loader(path)  # read outside the lock: I/O should not serialise requests
    with _lock:
        _cache[key] = _CacheEntry(mtime_ns=mtime, payload=payload)
    return payload


def clear_cache() -> None:
    """Drop all cached artefacts (used by tests)."""
    with _lock:
        _cache.clear()


# --------------------------------------------------------------------------- #
# Primitive loaders
# --------------------------------------------------------------------------- #
def _sanitise(value: Any) -> Any:
    """
    Make a pandas value JSON-safe **without changing it**.

    NaN/NaT become ``None`` (JSON has no NaN literal), numpy scalars become
    Python scalars. No rounding and no arithmetic: a metric that reaches the API
    is byte-for-byte the number the audit wrote.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar -> python scalar
        try:
            value = item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def read_csv(key: str) -> list[dict[str, Any]]:
    """
    Read an audit CSV as a list of JSON-safe row dicts (cached, verbatim).

    ``float_precision="round_trip"`` is REQUIRED, not cosmetic. Pandas' default
    CSV float parser is a fast approximation that can land one ULP away from the
    written value: ``0.06798687359118791`` in the file parses to
    ``0.0679868735911879``. That would make the API serve a number which differs
    from the audit it claims to report. round_trip guarantees the parsed float is
    exactly the one the audit wrote. Covered by the equality assertions in
    tests/test_api.py.
    """

    def loader(path: Path) -> list[dict[str, Any]]:
        df = pd.read_csv(path, float_precision="round_trip")
        if df.empty:
            raise ArtifactMalformedError(key, "file contains no data rows")
        return [{c: _sanitise(v) for c, v in row.items()} for row in df.to_dict("records")]

    return _load(key, loader)


def read_markdown(key: str) -> str:
    """Read a Markdown artefact as text (cached, verbatim)."""

    def loader(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ArtifactMalformedError(key, "file is empty")
        return text

    return _load(key, loader)


def artifact_status() -> list[dict[str, Any]]:
    """
    Presence/size/mtime of every declared artefact -- powers ``GET /health``.

    Never raises: a missing artefact is reported as ``present: false`` so the
    health endpoint can describe a partially-built project instead of failing.
    """
    out = []
    for key, path in ARTIFACTS.items():
        exists = path.exists()
        stat = path.stat() if exists else None
        out.append(
            {
                "key": key,
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "present": exists,
                "size_bytes": stat.st_size if stat else None,
                "modified_utc": (
                    pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()
                    if stat
                    else None
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Row selectors (filtering only -- no metric is touched)
# --------------------------------------------------------------------------- #
def phase1_rows() -> list[dict[str, Any]]:
    return read_csv("phase1_metrics")


def evaluated_models() -> list[str]:
    """Model names that have Phase 1 metrics, in the file's own order."""
    return [str(r["model"]) for r in phase1_rows()]


def phase1_row(model: str) -> dict[str, Any]:
    for row in phase1_rows():
        if str(row["model"]) == model:
            return row
    raise ModelNotFoundError(model, evaluated_models(), "Phase 1 performance")


def fairness_rows(model: str) -> list[dict[str, Any]]:
    rows = [r for r in read_csv("fairness_by_group") if str(r["model"]) == model]
    if not rows:
        available = sorted({str(r["model"]) for r in read_csv("fairness_by_group")})
        raise ModelNotFoundError(model, available, "fairness audit")
    return rows


def fairness_summary_rows(model: str) -> list[dict[str, Any]]:
    return [r for r in read_csv("fairness_summary") if str(r["model"]) == model]


def importance_rows() -> list[dict[str, Any]]:
    return read_csv("global_importance")


def local_explanation_rows() -> list[dict[str, Any]]:
    return read_csv("local_explanations")


def risk_rows() -> list[dict[str, Any]]:
    return read_csv("risk_register")


def assert_explained(model: str) -> None:
    """Guard for the explainability endpoint."""
    if model not in EXPLAINED_MODELS:
        raise ModelNotFoundError(model, list(EXPLAINED_MODELS), "explainability audit")
