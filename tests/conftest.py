"""
tests/conftest.py
=================
Shared fixtures for the MAAT model-intake tests.

Two things are deliberate here.

**The intake tests run against the real ``runtime/`` tree, not a temp directory.**
Several of the behaviours under test are *about* that tree -- that every generated
artefact lands under ``runtime/``, that nothing lands under ``data/``, ``models/``,
``predictions/`` or ``results/``, that the SQLite registry is the only state store.
Redirecting the store to ``tmp_path`` would replace the thing being asserted with a
stand-in. Instead, :func:`created_runs` records every run the tests create and
removes it afterwards, so the suite is repeatable without being self-fulfilling.

**The models the tests upload are built here, in-process, with scikit-learn.** They
are trusted by construction: the fixtures fit a real estimator and serialise it with
``joblib.dump`` into an in-memory buffer. No fixture file is committed, nothing is
downloaded, and no upload in this suite originates outside the test process -- which
is the only defensible way to exercise a code path whose documented risk is that
joblib deserialisation executes arbitrary code.
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The immutable evidence tree. Snapshotted and re-verified by the intake tests.
IMMUTABLE_DIRS = ("data", "models", "predictions", "results")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Synthetic dataset
#
# A deliberate group effect is injected on `sex` so the fairness screen has
# something real to report. That makes the *audit* meaningful; it is not a claim
# about any real population, and the tests assert only that the numbers served
# match the numbers computed, never that a disparity is or is not lawful.
# --------------------------------------------------------------------------- #
def build_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, size=n)
    hours = rng.integers(12, 58, size=n)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    region = rng.choice(["north", "south", "east"], size=n)
    score = (
        0.04 * (age - 40)
        + 0.05 * (hours - 34)
        + np.where(sex == "Male", 0.7, -0.3)
        + rng.normal(0, 0.7, size=n)
    )
    return pd.DataFrame(
        {
            "age": age,
            "hours_per_week": hours,
            "sex": sex,
            "region": region,
            "income": np.where(score > 0.4, ">50K", "<=50K"),
        }
    )


def _dump(estimator: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    return buffer.getvalue()


def _pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["age", "hours_per_week"]),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["sex", "region"],
                        ),
                    ]
                ),
            ),
            ("clf", classifier),
        ]
    )


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return build_frame()


@pytest.fixture(scope="session")
def dataset_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture(scope="session")
def model_bytes(frame: pd.DataFrame) -> bytes:
    """A trusted temporary sklearn Pipeline with ``predict_proba``."""
    pipeline = _pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(frame.drop(columns=["income"]), frame["income"])
    return _dump(pipeline)


@pytest.fixture(scope="session")
def predict_only_model_bytes(frame: pd.DataFrame) -> bytes:
    """
    A fitted classifier with **no** ``predict_proba``.

    ``SVC(probability=False)`` is the honest way to reach the no-probability branch:
    the estimator really cannot produce a score, so ROC-AUC really is unavailable.
    """
    pipeline = _pipeline(SVC(probability=False))
    pipeline.fit(frame.drop(columns=["income"]), frame["income"])
    return _dump(pipeline)


@pytest.fixture(scope="session")
def mismatched_model_bytes(frame: pd.DataFrame) -> bytes:
    """A model expecting a feature the submitted CSV does not contain."""
    trained = frame.rename(columns={"hours_per_week": "weekly_hours"})
    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["age", "weekly_hours"]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )
    pipeline.fit(trained.drop(columns=["income"]), trained["income"])
    return _dump(pipeline)


@pytest.fixture(scope="session")
def unsupported_explainability_model_bytes(frame: pd.DataFrame) -> bytes:
    """
    A fitted object that is not a supported classifier at all.

    A bare :class:`~sklearn.preprocessing.StandardScaler` loads cleanly from joblib
    and has no ``predict``, which is exactly the shape the validator must refuse
    rather than attempt to audit.
    """
    scaler = StandardScaler().fit(frame[["age", "hours_per_week"]])
    return _dump(scaler)


# --------------------------------------------------------------------------- #
# Multipart helpers
# --------------------------------------------------------------------------- #
def intake_files(
    model_bytes: bytes,
    dataset_bytes: bytes,
    *,
    model_name: str = "trusted_test_model.joblib",
    dataset_name: str = "test_data.csv",
) -> dict[str, tuple[str, bytes, str]]:
    return {
        "model_file": (model_name, model_bytes, "application/octet-stream"),
        "dataset_file": (dataset_name, dataset_bytes, "text/csv"),
    }


def intake_form(**overrides: Any) -> dict[str, str]:
    """
    Base multipart form for the intake endpoints.

    ``sensitive_columns`` is a JSON array rather than a comma-joined string so a
    column name containing a comma cannot be silently split in two.
    """
    form: dict[str, str] = {
        "target_column": "income",
        "positive_class": ">50K",
        "decision_threshold": "0.5",
        "sensitive_columns": '["sex", "region"]',
        "security_acknowledged": "true",
        "model_name": "trusted-test-logreg",
        "model_version": "1.0.0-test",
        "model_owner": "MAAT test suite",
        "intended_use": "Automated verification of the intake path. Not a real model.",
        "decision_context": "Synthetic data. No real people and no real decisions.",
    }
    form.update({k: v for k, v in overrides.items() if v is not None})
    for key in [k for k, v in overrides.items() if v is None]:
        form.pop(key, None)
    return form


# --------------------------------------------------------------------------- #
# Runtime cleanup
# --------------------------------------------------------------------------- #
@pytest.fixture
def created_runs() -> Iterator[list[str]]:
    """
    Collect audit-run ids created by a test and remove all trace of them afterwards.

    Both halves of a run's footprint are cleaned: the directory under
    ``runtime/audits/`` **and** the run's rows in the runtime SQLite state (registry
    entry, artefact checksums, events, waivers). Leaving the rows behind would not
    merely be untidy -- the registry's newest run would become a run whose files no
    longer exist, which is a state no real user can produce and which changes what
    the dashboard's registry page shows.

    Best-effort by design: this only ever touches ``runtime/``, and a failure to
    clean up must not turn a passing assertion into a failing test.
    """
    from app.onboarding import runtime_store as store

    ids: list[str] = []
    yield ids
    for audit_run_id in ids:
        target = store.AUDITS_DIR / audit_run_id
        if target.is_dir() and target.resolve().is_relative_to(store.RUNTIME_DIR):
            shutil.rmtree(target, ignore_errors=True)
    forget_runtime_rows(ids)


def forget_runtime_rows(audit_run_ids: Sequence[str]) -> None:
    """
    Delete the runtime SQLite rows belonging to the given uploaded audit runs.

    Only rows keyed by an id in ``audit_run_ids`` are touched, so a reference-case
    run in the same database is untouchable from here no matter what a test does.
    """
    if not audit_run_ids:
        return
    from app.gates import service as gates_service
    from app.registry.db import connect

    try:
        connection = connect(create=False)
    except FileNotFoundError:
        return  # No registry was created, so there is nothing to forget.
    keys = tuple(audit_run_ids)
    placeholders = ",".join("?" for _ in keys)
    try:
        gates_service.ensure_waiver_table(connection)
        for table, column in (
            ("governance_waivers", "audit_run_id"),
            ("registry_events", "run_id"),
            ("run_artifacts", "run_id"),
            ("audit_runs", "run_id"),
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE {column} IN ({placeholders})", keys
            )
        connection.commit()
    except sqlite3.Error:
        pass  # Cleanup must never convert a passing assertion into a failure.
    finally:
        connection.close()


@pytest.fixture(autouse=True)
def _sweep_uploads() -> Iterator[None]:
    """
    Remove upload directories a test created, unless a surviving run needs them.

    Autouse because the property is universal: no test should leave a stored
    submission behind. Every intake request creates a
    ``runtime/uploads/<upload_id>/`` directory, and a test that only *validates*
    never gets an audit-run id to clean up by, so per-test bookkeeping would miss
    exactly the cases that accumulate fastest. Left alone, a full run of the suite
    stores several hundred model and dataset copies.

    A new upload directory is deleted only when **no** run under
    ``runtime/audits/`` still cites it: a completed audit run records the
    ``runtime/uploads/...`` paths of the model and dataset it audited, and its DG-02
    control re-hashes the stored dataset. Deleting an upload out from under a run
    that is being kept would manufacture a missing-evidence finding that no user
    action produced -- so the reference check, not the fixture ordering, is what
    makes this safe.
    """
    from app.onboarding import runtime_store as store

    def upload_dirs() -> set[str]:
        root = store.UPLOADS_DIR
        return {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()

    before = upload_dirs()
    yield
    new = upload_dirs() - before
    if not new:
        return
    cited = _cited_upload_ids()
    for upload_id in new - cited:
        target = store.UPLOADS_DIR / upload_id
        if target.is_dir() and target.resolve().is_relative_to(store.RUNTIME_DIR):
            shutil.rmtree(target, ignore_errors=True)


def _cited_upload_ids() -> set[str]:
    """Upload ids named by the metadata of any run still present on disk."""
    from app.onboarding import runtime_store as store

    cited: set[str] = set()
    if not store.AUDITS_DIR.is_dir():
        return cited
    for run_dir in store.AUDITS_DIR.iterdir():
        for name in ("uploaded_model_metadata.json", "uploaded_dataset_metadata.json"):
            path = run_dir / name
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # An unreadable record is treated as citing everything it might
                # have cited: keep the uploads rather than risk breaking a run.
                return cited | {p.name for p in store.UPLOADS_DIR.iterdir() if p.is_dir()}
            if record.get("upload_id"):
                cited.add(str(record["upload_id"]))
    return cited


@pytest.fixture
def audit_run(
    client: TestClient,
    model_bytes: bytes,
    dataset_bytes: bytes,
    created_runs: list[str],
) -> dict[str, Any]:
    """One complete audit run over the trusted temporary model, cleaned up after."""
    response = client.post(
        "/api/onboarding/audits",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    created_runs.append(body["audit_run_id"])
    return body


# --------------------------------------------------------------------------- #
# Immutable-evidence snapshot
# --------------------------------------------------------------------------- #
def snapshot_evidence() -> dict[str, tuple[int, str]]:
    """
    Path -> (size, sha256) for every file under the immutable evidence tree.

    Hashed rather than merely stat-ed: an mtime comparison would miss a rewrite
    that happened to preserve the timestamp, and the whole point of the assertion
    is that the bytes are unchanged.
    """
    from app.registry.integrity import sha256_file

    snapshot: dict[str, tuple[int, str]] = {}
    for directory in IMMUTABLE_DIRS:
        root = PROJECT_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                key = path.relative_to(PROJECT_ROOT).as_posix()
                snapshot[key] = (path.stat().st_size, sha256_file(path))
    return snapshot


@pytest.fixture
def evidence_unchanged() -> Iterator[None]:
    """
    Fail the test if any file under the immutable evidence tree changed.

    Takes a checksum snapshot before the test body and re-verifies it after, so a
    write that a test *causes* is caught by that same test rather than by whichever
    unlucky test runs next.
    """
    before = snapshot_evidence()
    assert before, "No immutable evidence found -- the snapshot fixture is not working."
    yield
    after = snapshot_evidence()
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    removed = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    assert not changed, f"Immutable evidence was modified: {changed}"
    assert not removed, f"Immutable evidence was deleted: {removed}"
    assert not added, (
        "Files were created under data/, models/, predictions/ or results/: "
        f"{added}. Every generated artefact belongs under runtime/."
    )
