"""
app/onboarding/upload_service.py
================================
Accepts uploaded files and validates the **dataset** structurally.

Ordering is the point of this module
------------------------------------
Validation runs in a fixed order, cheapest and least dangerous first:

1. extension / size / remote-source checks  (nothing is written yet)
2. store both files under generated UUID names in ``runtime/uploads/<id>/``
3. checksum them immediately
4. parse the CSV header and reject duplicate column names
5. parse the CSV body and check the target / positive class / sensitive columns

Only after all of that does :mod:`app.onboarding.model_validator` deserialise the
model -- and only if the security warning was acknowledged. So a submission that
was never going to be auditable (wrong target column, single-class labels) is
rejected *without the uploaded pickle ever being executed*. Getting this order
wrong would be the single most consequential mistake in the intake path, which is
why the sequence is stated here and asserted by tests.

Duplicate column names
----------------------
``pandas`` silently de-duplicates repeated headers into ``income`` / ``income.1``.
That would let two different columns masquerade as one, so the header is read with
the :mod:`csv` module *before* pandas sees it and duplicates are rejected outright.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd

from app.onboarding import runtime_store as store
from app.onboarding import security
from app.onboarding.schemas import DatasetProfile, ValidationIssue

#: Values treated as null when reading an uploaded CSV. Mirrors the reference
#: pipeline's handling of the Adult dataset's " ?" placeholder.
NA_VALUES = ["", " ", "?", " ?", "NA", "N/A", "na", "null", "NULL", "None", "nan"]

MAX_SENSITIVE_COLUMNS = 6
MAX_GROUPS_PER_ATTRIBUTE = 50


def _issue(
    code: str, message: str, field: str | None = None, hint: str | None = None,
    severity: str = "error",
) -> ValidationIssue:
    return ValidationIssue(
        code=code, severity=severity, field=field, message=message, hint=hint
    )


# --------------------------------------------------------------------------- #
# Storing uploads
# --------------------------------------------------------------------------- #
def store_uploads(
    model_bytes: bytes,
    model_filename: str,
    dataset_bytes: bytes,
    dataset_filename: str,
    *,
    security_acknowledged: bool,
) -> dict[str, Any]:
    """
    Validate the file envelopes and persist both uploads under a new upload id.

    Nothing is deserialised here -- the model is stored as opaque bytes. The
    acknowledgement flag is recorded alongside so the audit run can evidence that
    it was given before any load occurred.
    """
    security.check_extension(model_filename, security.MODEL_EXTENSION, "model file")
    security.check_size(len(model_bytes), security.MAX_MODEL_BYTES, "model file")
    security.check_extension(dataset_filename, security.DATASET_EXTENSION, "dataset file")
    security.check_size(len(dataset_bytes), security.MAX_DATASET_BYTES, "dataset file")

    upload_id = security.new_upload_id()
    directory = store.upload_dir(upload_id, create=True)

    model_path = store.write_bytes(
        directory / security.generated_filename("model", security.MODEL_EXTENSION),
        model_bytes,
    )
    dataset_path = store.write_bytes(
        directory / security.generated_filename("dataset", security.DATASET_EXTENSION),
        dataset_bytes,
    )

    model_record = store.checksum_record(model_path, "uploaded_model")
    dataset_record = store.checksum_record(dataset_path, "uploaded_dataset")

    manifest = {
        "upload_id": upload_id,
        "uploaded_at": store.utc_now(),
        "security_acknowledged": bool(security_acknowledged),
        "security_warning": security.JOBLIB_SECURITY_WARNING,
        "production_hardening": security.PRODUCTION_HARDENING_NOTE,
        "model": {
            **model_record,
            # The original name is metadata only -- it was never used as a path.
            "original_filename_label": security.safe_label(model_filename, "model"),
            "stored_filename": model_path.name,
        },
        "dataset": {
            **dataset_record,
            "original_filename_label": security.safe_label(dataset_filename, "dataset"),
            "stored_filename": dataset_path.name,
        },
    }
    store.write_json(directory / "upload_manifest.json", manifest)
    return manifest


def load_upload_manifest(upload_id: str) -> dict[str, Any]:
    """Re-read a stored upload bundle, so validate and create need not re-upload."""
    directory = store.upload_dir(upload_id)
    target = directory / "upload_manifest.json"
    if not target.is_file():
        raise security.UploadRejected(
            "unknown_upload",
            f"No stored upload bundle '{upload_id}' was found.",
            hint="Upload the model and dataset again.",
        )
    import json

    return json.loads(target.read_text(encoding="utf-8"))


def upload_paths(manifest: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve the stored model and dataset paths from an upload manifest."""
    directory = store.upload_dir(str(manifest["upload_id"]))
    return (
        directory / str(manifest["model"]["stored_filename"]),
        directory / str(manifest["dataset"]["stored_filename"]),
    )


# --------------------------------------------------------------------------- #
# Dataset structure
# --------------------------------------------------------------------------- #
def read_header(dataset_bytes: bytes) -> list[str]:
    """
    Read the raw CSV header before pandas can rename anything.

    Returns the header exactly as written, duplicates included, so
    :func:`validate_dataset` can reject them.
    """
    text = dataset_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        return [h.strip() for h in next(reader)]
    except StopIteration:
        return []


def duplicate_columns(header: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def validate_dataset(
    dataset_bytes: bytes,
    target_column: str,
    positive_class: str,
    sensitive_columns: list[str],
) -> tuple[DatasetProfile | None, pd.DataFrame | None, list[ValidationIssue]]:
    """
    Parse and check the uploaded CSV.

    Returns ``(profile, dataframe, issues)``. The dataframe is ``None`` whenever an
    error-severity issue was raised, so no caller can accidentally proceed to
    inference on a dataset that failed validation.
    """
    issues: list[ValidationIssue] = []

    header = read_header(dataset_bytes)
    if not header:
        return None, None, [_issue("empty_dataset", "The uploaded CSV has no header row.")]

    duplicates = duplicate_columns(header)
    if duplicates:
        # Fatal and returned immediately: with duplicate headers we cannot know
        # which column the user's target/sensitive selections refer to.
        return (
            None,
            None,
            [
                _issue(
                    "duplicate_columns",
                    "The CSV has duplicate column name(s): "
                    + ", ".join(sorted(duplicates))
                    + ". Column names must be unique.",
                    field="dataset",
                    hint="pandas would silently rename these (e.g. 'income.1'), which "
                    "would make the target and sensitive-column selections ambiguous.",
                )
            ],
        )

    try:
        frame = pd.read_csv(
            io.BytesIO(dataset_bytes),
            skipinitialspace=True,
            na_values=NA_VALUES,
            keep_default_na=True,
            float_precision="round_trip",
        )
    except Exception as exc:  # pandas raises a variety of parser errors
        return (
            None,
            None,
            [
                _issue(
                    "unparseable_csv",
                    f"The CSV could not be parsed: {type(exc).__name__}: {exc}",
                    field="dataset",
                )
            ],
        )

    frame.columns = [str(c).strip() for c in frame.columns]
    columns = list(frame.columns)

    profile = DatasetProfile(
        row_count=int(len(frame)),
        column_count=len(columns),
        columns=columns,
        duplicate_columns=[],
    )

    if len(frame) < security.MIN_DATASET_ROWS:
        issues.append(
            _issue(
                "dataset_too_small",
                f"The dataset has {len(frame)} rows; at least "
                f"{security.MIN_DATASET_ROWS} are required for the group metrics to "
                "mean anything.",
                field="dataset",
            )
        )
    if len(frame) > security.MAX_DATASET_ROWS:
        issues.append(
            _issue(
                "dataset_too_large",
                f"The dataset has {len(frame)} rows; the limit is "
                f"{security.MAX_DATASET_ROWS} for this local prototype.",
                field="dataset",
            )
        )

    # -- target column -------------------------------------------------------- #
    if target_column not in columns:
        issues.append(
            _issue(
                "missing_target_column",
                f"The target column '{target_column}' is not in the dataset.",
                field="target_column",
                hint="Available columns: " + ", ".join(columns[:40]),
            )
        )
        return profile, None, issues

    profile.target_column = target_column
    target = frame[target_column]
    if target.isna().all():
        issues.append(
            _issue(
                "empty_target",
                f"The target column '{target_column}' is entirely null.",
                field="target_column",
            )
        )
        return profile, None, issues

    labels = target.dropna().astype("string").str.strip()
    counts = labels.value_counts()
    profile.target_classes = [str(v) for v in counts.index.tolist()]
    profile.target_class_counts = {str(k): int(v) for k, v in counts.items()}

    if len(profile.target_classes) < 2:
        issues.append(
            _issue(
                "single_class_target",
                f"The target column '{target_column}' has only one distinct value "
                f"({profile.target_classes[0] if profile.target_classes else 'none'}). "
                "Binary classification metrics are undefined.",
                field="target_column",
            )
        )
    elif len(profile.target_classes) > 2:
        issues.append(
            _issue(
                "multiclass_target",
                f"The target column '{target_column}' has "
                f"{len(profile.target_classes)} distinct values. This prototype "
                "audits binary classification only.",
                field="target_column",
                hint="Values found: " + ", ".join(profile.target_classes[:12]),
            )
        )

    # -- positive class ------------------------------------------------------- #
    wanted = str(positive_class).strip()
    if wanted not in profile.target_classes:
        issues.append(
            _issue(
                "unknown_positive_class",
                f"The positive class '{wanted}' does not occur in "
                f"'{target_column}'.",
                field="positive_class",
                hint="Values present: " + ", ".join(profile.target_classes[:12]),
            )
        )
    else:
        profile.positive_class = wanted
        profile.positive_class_count = int(counts.get(wanted, 0))

    # -- sensitive columns ---------------------------------------------------- #
    missing_sensitive = [c for c in sensitive_columns if c not in columns]
    if missing_sensitive:
        issues.append(
            _issue(
                "missing_sensitive_columns",
                "Selected sensitive column(s) not in the dataset: "
                + ", ".join(missing_sensitive)
                + ".",
                field="sensitive_columns",
                hint="Available columns: " + ", ".join(columns[:40]),
            )
        )
    if target_column in sensitive_columns:
        issues.append(
            _issue(
                "target_as_sensitive",
                f"'{target_column}' is the target column and cannot also be a "
                "sensitive attribute.",
                field="sensitive_columns",
            )
        )
    if len(sensitive_columns) > MAX_SENSITIVE_COLUMNS:
        issues.append(
            _issue(
                "too_many_sensitive_columns",
                f"{len(sensitive_columns)} sensitive columns selected; the limit is "
                f"{MAX_SENSITIVE_COLUMNS}.",
                field="sensitive_columns",
            )
        )
    for column in sensitive_columns:
        if column in columns:
            distinct = int(frame[column].astype("string").nunique(dropna=True))
            if distinct > MAX_GROUPS_PER_ATTRIBUTE:
                issues.append(
                    _issue(
                        "high_cardinality_sensitive_column",
                        f"'{column}' has {distinct} distinct values. Group metrics "
                        "over that many groups are dominated by sampling noise.",
                        field="sensitive_columns",
                        severity="warning",
                    )
                )
    profile.sensitive_columns = [c for c in sensitive_columns if c in columns]

    if not sensitive_columns:
        issues.append(
            _issue(
                "no_sensitive_columns",
                "No sensitive columns were selected, so no fairness assessment will "
                "be performed. This is recorded as 'not_provided_by_user' -- it is "
                "not a pass and not a fairness claim.",
                field="sensitive_columns",
                severity="warning",
            )
        )

    fatal = [i for i in issues if i.severity == "error"]
    return profile, (None if fatal else frame), issues
