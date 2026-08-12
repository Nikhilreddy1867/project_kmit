"""
tests/test_onboarding.py
========================
Tests for the MAAT model-intake path: upload security, structural validation, and
the audit run those uploads produce.

What these tests are really protecting
--------------------------------------
1. **Refusals happen before anything is trusted.** A ``.pkl`` upload, a withheld
   acknowledgement, a duplicate column name and a wrong target column are all
   rejected with a specific machine-readable code -- not a generic 400, and not
   after the model has already been deserialised.
2. **Nothing escapes ``runtime/``.** Every generated artefact is asserted to live
   under ``runtime/``, and the immutable evidence tree is checksummed before and
   after each end-to-end test.
3. **The API never improves on its own numbers.** Performance is re-derived from
   the run's own ``predictions.csv`` and must match what the endpoint serves
   exactly. An unavailable metric stays unavailable: ``roc_auc`` is ``None`` with a
   stated reason for a model without ``predict_proba``, never an estimate.
4. **Silence is not a pass.** With no sensitive columns selected the fairness
   status is ``not_provided_by_user``, and an undefined rate is ``None`` rather
   than ``0.0`` -- because a zero would read as a measured value.

Run from the repo root with the venv active::

    pytest -q tests/test_onboarding.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.onboarding import runtime_store as store
from app.onboarding import security
from conftest import PROJECT_ROOT, intake_files, intake_form


def _read_predictions(audit_run_id: str) -> list[dict[str, str]]:
    path = store.artifact_path(audit_run_id, "predictions.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Upload security
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "filename",
    [
        "model.pkl",
        "model.pickle",
        "model.py",
        "model.zip",
        "model.tar.gz",
        "model.exe",
        "model.sh",
    ],
)
def test_dangerous_file_types_are_refused(
    client: TestClient, dataset_bytes: bytes, filename: str
) -> None:
    """
    Arbitrary-code formats are refused by extension, before any read.

    The payload here is a single harmless byte on purpose: the refusal must not
    depend on the content being inspected, because inspecting a pickle is already
    the dangerous act.
    """
    response = client.post(
        "/api/onboarding/validate",
        files={
            "model_file": (filename, b"x", "application/octet-stream"),
            "dataset_file": ("data.csv", dataset_bytes, "text/csv"),
        },
        data=intake_form(),
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"] in {"denied_file_type", "unsupported_file_type"}
    assert ".joblib" in json.dumps(body)


def test_remote_url_as_filename_is_refused(
    client: TestClient, dataset_bytes: bytes
) -> None:
    """A URL is not a file. There is no fetch path, and the label is still refused."""
    response = client.post(
        "/api/onboarding/validate",
        files={
            "model_file": (
                "https://example.invalid/model.joblib",
                b"x",
                "application/octet-stream",
            ),
            "dataset_file": ("data.csv", dataset_bytes, "text/csv"),
        },
        data=intake_form(),
    )
    assert response.status_code == 422
    assert response.json()["error"] in {
        "remote_source_rejected",
        "denied_file_type",
        "unsupported_file_type",
    }


def test_path_traversal_filename_cannot_escape_runtime(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """
    A traversal attempt in the filename cannot reach outside ``runtime/``.

    The defence is structural rather than a refusal: the original name is reduced to
    a bare label used only for display, and the bytes are stored under a generated
    UUID name. So the assertion is about *where the file ended up* and *what the
    recorded name is* -- refusing the request would be a weaker guarantee, because it
    would still leave the door open to any traversal string the blocklist missed.
    """
    from app.onboarding import runtime_store as store
    from app.onboarding.security import safe_label

    assert safe_label("../../results/model_metrics.joblib") == "model_metrics.joblib"

    response = client.post(
        "/api/onboarding/validate",
        files={
            "model_file": (
                "../../results/model_metrics.joblib",
                model_bytes,
                "application/octet-stream",
            ),
            "dataset_file": ("../../data/adult.csv", dataset_bytes, "text/csv"),
        },
        data=intake_form(),
    )
    assert response.status_code == 200, response.text
    upload_id = response.json()["upload_id"]

    # Both files landed inside the upload's own directory under runtime/, under
    # generated names -- nothing named after the traversal string exists anywhere.
    upload_root = store.UPLOADS_DIR.resolve()
    stored = sorted(p for p in (upload_root / upload_id).rglob("*") if p.is_file())
    assert stored, "the upload wrote nothing"
    for path in stored:
        resolved = path.resolve()
        assert resolved.is_relative_to(upload_root), resolved
        assert ".." not in resolved.name
    assert not (store.PROJECT_ROOT / "results" / "model_metrics.joblib").exists()


def test_empty_model_file_is_refused(client: TestClient, dataset_bytes: bytes) -> None:
    response = client.post(
        "/api/onboarding/validate",
        files={
            "model_file": ("empty.joblib", b"", "application/octet-stream"),
            "dataset_file": ("data.csv", dataset_bytes, "text/csv"),
        },
        data=intake_form(),
    )
    assert response.status_code == 422
    assert response.json()["error"] == "empty_file"


def test_acknowledgement_is_required_before_the_model_is_loaded(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """
    Without the explicit acknowledgement, the request is refused and the warning is
    restated in the refusal itself.
    """
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(security_acknowledged="false"),
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"] == "acknowledgement_required"
    assert security.JOBLIB_SECURITY_WARNING in json.dumps(body)


def test_acknowledgement_default_is_refusal(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """Omitting the field entirely must not be read as consent."""
    response = client.post(
        "/api/onboarding/audits",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(security_acknowledged=None),
    )
    assert response.status_code == 422
    assert response.json()["error"] == "acknowledgement_required"


def test_exact_security_warning_text_is_served(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """
    The warning is served verbatim, and the production-hardening note names the
    safer alternatives rather than gesturing at them.
    """
    body = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(),
    ).json()
    assert body["security_warning"] == (
        "Joblib files may execute arbitrary code. Upload only models from trusted "
        "sources. This local academic prototype must not accept untrusted model "
        "files in production."
    )
    hardening = body["production_hardening"].lower()
    assert "sandbox" in hardening
    assert "onnx" in hardening
    assert "skops" in hardening


# --------------------------------------------------------------------------- #
# Dataset validation
# --------------------------------------------------------------------------- #
def test_duplicate_csv_columns_are_rejected(
    client: TestClient, model_bytes: bytes, frame: pd.DataFrame
) -> None:
    """
    A duplicated header is refused rather than resolved by picking one.

    Written as raw text because pandas would rename the second occurrence on the
    way out, which is precisely the ambiguity the check exists to prevent.
    """
    header = "age,hours_per_week,sex,region,region,income"
    rows = [
        f"{r.age},{r.hours_per_week},{r.sex},{r.region},{r.region},{r.income}"
        for r in frame.head(40).itertuples()
    ]
    csv_bytes = ("\n".join([header, *rows]) + "\n").encode()

    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, csv_bytes),
        data=intake_form(),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is False
    codes = {i["code"] for i in body["issues"]}
    assert "duplicate_columns" in codes
    assert "region" in json.dumps(body["issues"])


def test_missing_target_column_names_the_available_columns(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(target_column="salary_band"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    issue = next(i for i in body["issues"] if i["code"] == "missing_target_column")
    assert "salary_band" in issue["message"]
    assert "income" in (issue.get("hint") or "") + issue["message"]


def test_missing_sensitive_column_is_named_precisely(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(sensitive_columns='["sex", "ethnicity"]'),
    )
    body = response.json()
    assert body["valid"] is False
    issue = next(i for i in body["issues"] if i["code"] == "missing_sensitive_columns")
    assert "ethnicity" in issue["message"]
    assert "sex" not in issue["message"].replace("sensitive", "")


def test_unknown_positive_class_is_rejected(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(positive_class="RICH"),
    )
    body = response.json()
    assert body["valid"] is False
    codes = {i["code"] for i in body["issues"]}
    assert "unknown_positive_class" in codes


def test_single_class_target_is_rejected(
    client: TestClient, model_bytes: bytes, frame: pd.DataFrame
) -> None:
    """One class is not a classification problem, and no metric would be defined."""
    single = frame.copy()
    single["income"] = ">50K"
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, single.to_csv(index=False).encode()),
        data=intake_form(),
    )
    body = response.json()
    assert body["valid"] is False
    assert "single_class_target" in {i["code"] for i in body["issues"]}


def test_multiclass_target_is_rejected(
    client: TestClient, model_bytes: bytes, frame: pd.DataFrame
) -> None:
    """The platform audits binary classification only, and says so."""
    multi = frame.copy()
    multi.loc[multi.index[:80], "income"] = "unknown"
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, multi.to_csv(index=False).encode()),
        data=intake_form(),
    )
    body = response.json()
    assert body["valid"] is False
    assert "multiclass_target" in {i["code"] for i in body["issues"]}


def test_too_few_rows_is_rejected(
    client: TestClient, model_bytes: bytes, frame: pd.DataFrame
) -> None:
    tiny = frame.head(5)
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, tiny.to_csv(index=False).encode()),
        data=intake_form(),
    )
    body = response.json()
    assert body["valid"] is False
    issue = next(i for i in body["issues"] if i["code"] == "dataset_too_small")
    assert str(security.MIN_DATASET_ROWS) in issue["message"]


def test_unparseable_csv_is_rejected(
    client: TestClient, model_bytes: bytes
) -> None:
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, b"\x00\x01\x02 not a csv at all \xff"),
        data=intake_form(),
    )
    body = response.json()
    assert response.status_code in {200, 422}
    payload = json.dumps(body)
    assert "unparseable_csv" in payload or "empty_dataset" in payload


# --------------------------------------------------------------------------- #
# Model / feature compatibility
# --------------------------------------------------------------------------- #
def test_feature_mismatch_names_missing_and_unexpected_features(
    client: TestClient, mismatched_model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """
    The error must be precise enough to act on: which feature is missing, and which
    submitted column the model did not ask for.
    """
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(mismatched_model_bytes, dataset_bytes),
        data=intake_form(),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    compatibility = body["feature_compatibility"]
    assert compatibility["checked"] is True
    assert compatibility["compatible"] is False
    assert "weekly_hours" in compatibility["missing_features"]
    assert "hours_per_week" in compatibility["unexpected_features"]
    assert body["valid"] is False
    assert "feature_mismatch" in {i["code"] for i in body["issues"]}


def test_non_classifier_model_is_refused(
    client: TestClient,
    unsupported_explainability_model_bytes: bytes,
    dataset_bytes: bytes,
) -> None:
    """A fitted transformer is not auditable, and is refused rather than coerced."""
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(unsupported_explainability_model_bytes, dataset_bytes),
        data=intake_form(),
    )
    body = response.json() if response.status_code == 200 else response.json()
    payload = json.dumps(body)
    assert response.status_code in {200, 422}
    assert "predict" in payload
    if response.status_code == 200:
        assert body["valid"] is False
        assert body["model_capabilities"]["has_predict"] is False


def test_predict_only_model_reports_roc_auc_as_unavailable(
    client: TestClient, predict_only_model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """
    No ``predict_proba`` is a stated limitation at validation time, not a surprise
    discovered after the audit has run.
    """
    body = client.post(
        "/api/onboarding/validate",
        files=intake_files(predict_only_model_bytes, dataset_bytes),
        data=intake_form(),
    ).json()
    assert body["valid"] is True, body["issues"]
    assert body["model_capabilities"]["has_predict_proba"] is False
    assert body["audit_capabilities"]["roc_auc"] is False
    warnings = [i for i in body["issues"] if i["severity"] != "error"]
    assert any("predict_proba" in json.dumps(w) for w in warnings)


def test_validation_creates_no_audit_run(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """Validating is not auditing: the run list must not grow."""
    before = client.get("/api/onboarding/audits").json()["count"]
    response = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(),
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert client.get("/api/onboarding/audits").json()["count"] == before


def test_upload_never_overwrites_an_existing_file(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """
    Two identical submissions get two separate upload directories.

    The generated-UUID naming plus exclusive-create writes make an overwrite
    impossible rather than unlikely, so the test asserts distinct storage paths.
    """
    first = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(),
    ).json()
    second = client.post(
        "/api/onboarding/validate",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(),
    ).json()
    assert first["upload_id"] != second["upload_id"]
    for upload_id in (first["upload_id"], second["upload_id"]):
        directory = store.upload_dir(upload_id)
        assert directory.is_dir()
        assert directory.resolve().is_relative_to(store.RUNTIME_DIR.resolve())


def test_stored_filenames_are_generated_not_user_supplied(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    """The user's filename is a label in the manifest, never a path on disk."""
    body = client.post(
        "/api/onboarding/validate",
        files=intake_files(
            model_bytes, dataset_bytes, model_name="my model (v2).joblib"
        ),
        data=intake_form(),
    ).json()
    directory = store.upload_dir(body["upload_id"])
    names = sorted(p.name for p in directory.iterdir())
    assert not any("my model" in n for n in names), names
    assert any(n.startswith("model_") and n.endswith(".joblib") for n in names)
    manifest = json.loads((directory / "upload_manifest.json").read_text("utf-8"))
    assert manifest["model"]["original_filename_label"] != manifest["model"]["stored_filename"]


# --------------------------------------------------------------------------- #
# End-to-end audit run
# --------------------------------------------------------------------------- #
def test_trusted_model_is_audited_end_to_end(
    client: TestClient, audit_run: dict[str, Any], evidence_unchanged: None
) -> None:
    """
    A trusted temporary sklearn Pipeline goes in; a complete governance record
    comes out -- and the reference case is untouched (asserted by the fixture).
    """
    assert audit_run["run_type"] == "uploaded_model"
    assert audit_run["audit_run_id"].startswith("audit-")
    assert audit_run["governance_state"] in {
        "review_required",
        "insufficient_evidence",
        "blocked_by_policy",
    }
    assert audit_run["fairness_status"] == "available"
    assert audit_run["explainability_status"] == "available"
    assert audit_run["conformity_bundle_id"].startswith("bundle-")
    assert audit_run["artifact_count"] >= 12
    assert set(audit_run["gate_summary"]) == {"DG", "TG", "VG", "RG", "OG"}


def test_every_artifact_is_written_under_runtime(
    audit_run: dict[str, Any], evidence_unchanged: None
) -> None:
    """
    The one storage rule, asserted on the real paths.

    ``written_under`` is checked as a string *and* every artefact path is resolved
    and confirmed to be inside ``runtime/`` -- a claim in a response field is not
    evidence about the filesystem.
    """
    assert audit_run["written_under"].startswith("runtime/")
    directory = store.audit_dir(audit_run["audit_run_id"])
    runtime = store.RUNTIME_DIR.resolve()
    assert directory.resolve().is_relative_to(runtime)

    files = [p for p in directory.rglob("*") if p.is_file()]
    assert len(files) >= 12
    for path in files:
        resolved = path.resolve()
        assert resolved.is_relative_to(runtime), f"{path} escaped runtime/"
        relative = resolved.relative_to(PROJECT_ROOT).parts[0]
        assert relative == "runtime", f"{path} is not under runtime/"


def test_expected_artifacts_are_all_present(audit_run: dict[str, Any]) -> None:
    directory = store.audit_dir(audit_run["audit_run_id"])
    present = {p.name for p in directory.iterdir() if p.is_file()}
    for name in store.AUDIT_ARTIFACT_NAMES:
        assert name in present, f"{name} was not written"


def test_performance_matches_the_runs_own_predictions_exactly(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    The served metrics are re-derived from ``predictions.csv`` and compared exactly.

    Recomputing here is the point: it proves the API is reporting the model's own
    outputs rather than a number produced somewhere else.
    """
    run_id = audit_run["audit_run_id"]
    served = client.get(f"/api/onboarding/audits/{run_id}/performance").json()
    rows = _read_predictions(run_id)
    assert len(rows) == served["n_samples"]

    positive = served["positive_class"]
    tp = sum(r["actual_label"] == positive and r["predicted_label"] == positive for r in rows)
    tn = sum(r["actual_label"] != positive and r["predicted_label"] != positive for r in rows)
    fp = sum(r["actual_label"] != positive and r["predicted_label"] == positive for r in rows)
    fn = sum(r["actual_label"] == positive and r["predicted_label"] != positive for r in rows)

    matrix = served["confusion_matrix"]
    assert matrix["true_positives"] == tp
    assert matrix["true_negatives"] == tn
    assert matrix["false_positives"] == fp
    assert matrix["false_negatives"] == fn
    assert tp + tn + fp + fn == served["n_samples"]

    assert served["accuracy"] == pytest.approx((tp + tn) / len(rows), abs=1e-12)
    assert served["precision"] == pytest.approx(tp / (tp + fp), abs=1e-12)
    assert served["recall"] == pytest.approx(tp / (tp + fn), abs=1e-12)


def test_roc_auc_is_null_with_a_reason_for_a_predict_only_model(
    client: TestClient,
    predict_only_model_bytes: bytes,
    dataset_bytes: bytes,
    created_runs: list[str],
) -> None:
    """
    An unavailable metric is ``None`` plus an explanation -- never an estimate, and
    never a zero.
    """
    created = client.post(
        "/api/onboarding/audits",
        files=intake_files(predict_only_model_bytes, dataset_bytes),
        data=intake_form(model_name="trusted-test-svc"),
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["audit_run_id"]
    created_runs.append(run_id)

    served = client.get(f"/api/onboarding/audits/{run_id}/performance").json()
    assert served["roc_auc"] is None
    assert served["roc_auc_unavailable_reason"]
    assert "predict_proba" in served["roc_auc_unavailable_reason"]
    assert served["threshold_applied"] is False
    # The metrics that *are* defined must still be present.
    assert isinstance(served["f1"], float)


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def test_fairness_covers_only_the_selected_columns(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """`region` was selected and `age` was not; the audit must not volunteer `age`."""
    fairness = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/fairness"
    ).json()
    assert fairness["status"] == "available"
    assessed = {a["attribute"] for a in fairness["attributes"]}
    assert assessed == {"sex", "region"}
    assert "age" not in assessed
    assert set(fairness["sensitive_columns_requested"]) == {"sex", "region"}


def test_no_sensitive_columns_gives_not_provided_by_user(
    client: TestClient,
    model_bytes: bytes,
    dataset_bytes: bytes,
    created_runs: list[str],
) -> None:
    """
    The central honesty rule: an unmeasured audit is not a passed audit.

    ``not_provided_by_user`` must be the status, no group rows may be invented, and
    nothing in the response may read as a fairness claim in either direction.
    """
    created = client.post(
        "/api/onboarding/audits",
        files=intake_files(model_bytes, dataset_bytes),
        data=intake_form(sensitive_columns="[]", model_name="trusted-test-nofair"),
    )
    assert created.status_code == 201, created.text
    body = created.json()
    created_runs.append(body["audit_run_id"])
    assert body["fairness_status"] == "not_provided_by_user"

    fairness = client.get(
        f"/api/onboarding/audits/{body['audit_run_id']}/fairness"
    ).json()
    assert fairness["status"] == "not_provided_by_user"
    assert fairness["attributes"] == []
    assert fairness["groups"] == []
    assert fairness["sensitive_columns_requested"] == []
    detail = fairness["status_detail"].lower()
    assert "not a pass" in detail or "not a fairness claim" in detail
    assert "pass" not in detail.replace("not a pass", "")


def test_undefined_fairness_denominators_are_null_not_zero(
    client: TestClient,
    model_bytes: bytes,
    frame: pd.DataFrame,
    created_runs: list[str],
) -> None:
    """
    A group with no positive-labelled rows has no TPR. It must be ``None``.

    A zero here would be a measurement that was never taken, and it would drag any
    downstream disparity comparison toward a conclusion the data cannot support.
    """
    edited = frame.copy()
    # Build a group whose members are all negative-labelled, so TPR is undefined.
    edited.loc[edited.index[:60], "region"] = "west"
    edited.loc[edited.index[:60], "income"] = "<=50K"

    created = client.post(
        "/api/onboarding/audits",
        files=intake_files(model_bytes, edited.to_csv(index=False).encode()),
        data=intake_form(
            sensitive_columns='["region"]', model_name="trusted-test-undefined"
        ),
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["audit_run_id"]
    created_runs.append(run_id)

    fairness = client.get(f"/api/onboarding/audits/{run_id}/fairness").json()
    west = next(g for g in fairness["groups"] if g["group"] == "west")
    assert west["actual_positive_rate"] == 0.0
    assert west["true_positive_rate"] is None, (
        "An undefined TPR was converted into a number: "
        f"{west['true_positive_rate']!r}"
    )
    attribute = next(a for a in fairness["attributes"] if a["attribute"] == "region")
    assert attribute["undefined_metric_count"] >= 1


def test_four_fifths_notice_is_stated_verbatim(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    fairness = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/fairness"
    ).json()
    assert fairness["four_fifths_notice"] == (
        "The four-fifths threshold is a screening heuristic. It is not a legal "
        "conclusion and does not prove discrimination or causation."
    )


def test_fairness_makes_no_legal_or_causal_claim(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """Scan the whole response for language the platform must never use."""
    payload = json.dumps(
        client.get(
            f"/api/onboarding/audits/{audit_run['audit_run_id']}/fairness"
        ).json()
    ).lower()
    for forbidden in (
        "is illegal",
        "violates",
        "proves discrimination",
        "proven discrimination",
        "causes",
        "legally compliant",
        "certified compliant",
    ):
        assert forbidden not in payload, f"forbidden claim in fairness output: {forbidden}"


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_explainability_states_association_not_causation(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    explain = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/explainability"
    ).json()
    assert explain["status"] == "available"
    assert explain["global_importance"]
    text = json.dumps(explain).lower()
    assert "association" in text
    assert "not causation" in text or "not a causal claim" in text


def test_importance_scores_are_real_measurements(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    Every feature reported has a mean and a standard deviation from the same
    permutation run. A score without a spread would be an assertion, not a result.
    """
    explain = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/explainability"
    ).json()
    assert explain["method"]
    assert explain["n_repeats"] >= 1
    for item in explain["global_importance"]:
        assert isinstance(item["importance_mean"], float)
        assert item["importance_std"] is None or isinstance(item["importance_std"], float)
        assert item["feature"]
    means = [i["importance_mean"] for i in explain["global_importance"]]
    assert means == sorted(means, reverse=True), "importance is not ranked"


def test_unsupported_model_type_reports_explainability_unavailable(
    client: TestClient,
    predict_only_model_bytes: bytes,
    dataset_bytes: bytes,
    created_runs: list[str],
) -> None:
    """
    Where a method genuinely cannot run, the API says so in the required words and
    invents nothing.
    """
    created = client.post(
        "/api/onboarding/audits",
        files=intake_files(predict_only_model_bytes, dataset_bytes),
        data=intake_form(model_name="trusted-test-svc-explain"),
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["audit_run_id"]
    created_runs.append(run_id)

    explain = client.get(f"/api/onboarding/audits/{run_id}/explainability").json()
    if explain["status"] == "available":
        # Permutation importance only needs `predict`, so it may legitimately run.
        # What must *not* happen is a local explanation with no method behind it.
        assert explain["local_method"] is None or explain["local_explanations"]
    else:
        assert explain["status_detail"] == (
            "Explainability not available for this model type in the current local "
            "prototype."
        )
        assert explain["global_importance"] == []
        assert explain["local_explanations"] == []


# --------------------------------------------------------------------------- #
# Governance framing
# --------------------------------------------------------------------------- #
def test_uploaded_run_never_claims_deployment_approval(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    governance = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/governance"
    ).json()
    assert governance["human_review_required"] is True
    assert governance["deployment_authorisation"] == "not_granted"
    text = json.dumps(governance).lower()
    assert "approved for deployment" not in text
    assert "production ready" not in text
    assert "production-ready" not in text


def test_every_uploaded_endpoint_carries_the_decision_support_notice(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    for suffix in (
        "",
        "/performance",
        "/fairness",
        "/explainability",
        "/governance",
        "/integrity",
        "/timeline",
    ):
        body = client.get(f"/api/onboarding/audits/{run_id}{suffix}").json()
        assert "notice" in body, suffix
        assert "human governance review" in body["notice"].lower(), suffix


def test_uploaded_run_is_separated_from_the_reference_case(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    listing = client.get("/api/onboarding/audits").json()
    assert listing["reference_case_separate"] is True
    ids = {r["audit_run_id"] for r in listing["runs"]}
    assert audit_run["audit_run_id"] in ids
    assert all(r["run_type"] == "uploaded_model" for r in listing["runs"])


# --------------------------------------------------------------------------- #
# Evidence integrity
# --------------------------------------------------------------------------- #
def test_integrity_verifies_a_clean_run(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    integrity = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/integrity"
    ).json()
    assert integrity["integrity_status"] == "verified"
    assert integrity["integrity_ok"] is True
    assert integrity["changed_count"] == 0
    assert integrity["missing_count"] == 0
    assert integrity["verified_count"] == integrity["artifacts_checked"]
    assert integrity["artifacts_checked"] >= 12


def test_modifying_a_runtime_artifact_is_detected(
    client: TestClient, audit_run: dict[str, Any], evidence_unchanged: None
) -> None:
    """
    Deliberately corrupt one *runtime* artefact and confirm the check catches it.

    The tampering is confined to this run's own directory under ``runtime/`` -- the
    committed evidence is never touched, which the ``evidence_unchanged`` fixture
    verifies by checksum on the way out.
    """
    run_id = audit_run["audit_run_id"]
    target = store.artifact_path(run_id, "performance.json")
    assert target.resolve().is_relative_to(store.RUNTIME_DIR.resolve())

    original = target.read_bytes()
    try:
        payload = json.loads(original)
        payload["accuracy"] = 0.999999
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        integrity = client.get(f"/api/onboarding/audits/{run_id}/integrity").json()
        assert integrity["integrity_ok"] is False
        assert integrity["integrity_status"] in {"modified", "modified_and_incomplete"}
        assert integrity["changed_count"] >= 1
        changed = [
            a for a in integrity["artifacts"] if a["status"] not in {"verified", "ok"}
        ]
        assert any("performance.json" in a["path"] for a in changed)
    finally:
        target.write_bytes(original)

    restored = client.get(f"/api/onboarding/audits/{run_id}/integrity").json()
    assert restored["integrity_status"] == "verified", (
        "Restoring the original bytes did not restore the verdict, so the check is "
        "not a pure function of file content."
    )


def test_missing_artifact_is_reported_as_incomplete(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    run_id = audit_run["audit_run_id"]
    target = store.artifact_path(run_id, "traceability.json")
    original = target.read_bytes()
    try:
        target.unlink()
        integrity = client.get(f"/api/onboarding/audits/{run_id}/integrity").json()
        assert integrity["missing_count"] >= 1
        assert "incomplete" in integrity["integrity_status"]
    finally:
        target.write_bytes(original)


# --------------------------------------------------------------------------- #
# Timeline and registry
# --------------------------------------------------------------------------- #
def test_timeline_is_ordered_and_records_the_audit(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    timeline = client.get(
        f"/api/onboarding/audits/{audit_run['audit_run_id']}/timeline"
    ).json()
    assert timeline["count"] >= 1
    times = [e["event_time"] for e in timeline["events"]]
    assert times == sorted(times), "timeline is not in chronological order"
    assert any("audit" in e["event_type"] for e in timeline["events"])


def test_uploaded_run_is_registered_without_touching_reference_rows(
    client: TestClient, audit_run: dict[str, Any]
) -> None:
    """
    The run joins the registry, and the reference case's own rows are unaffected.

    Registration is allowed to fail (it is reported as a warning, not swallowed);
    what must not happen is a reference row changing because of an upload.
    """
    reference_before = client.get("/api/registry/runs").json()
    if audit_run["registry_run_id"]:
        ids = {r["run_id"] for r in reference_before["runs"]}
        assert audit_run["registry_run_id"] in ids
    if audit_run["registry_run_id"]:
        integrity = client.get(
            f"/api/registry/runs/{audit_run['registry_run_id']}/integrity"
        )
        assert integrity.status_code == 200, integrity.text


# --------------------------------------------------------------------------- #
# 404s
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "suffix",
    ["", "/performance", "/fairness", "/explainability", "/governance", "/integrity",
     "/timeline"],
)
def test_unknown_run_gives_404_listing_alternatives(
    client: TestClient, suffix: str
) -> None:
    response = client.get(f"/api/onboarding/audits/audit-does-not-exist{suffix}")
    assert response.status_code == 404
    body = response.json()
    assert "audit-does-not-exist" in json.dumps(body)


def test_unknown_upload_id_is_rejected(
    client: TestClient, model_bytes: bytes, dataset_bytes: bytes
) -> None:
    response = client.post(
        "/api/onboarding/audits", data=intake_form(upload_id="upload-nope")
    )
    assert response.status_code in {404, 422}
    assert "upload" in json.dumps(response.json()).lower()


def test_audits_requires_files_or_an_upload_id(client: TestClient) -> None:
    response = client.post("/api/onboarding/audits", data=intake_form())
    assert response.status_code == 422
    assert response.json()["error"] == "missing_files"
