"""
app/onboarding/security.py
==========================
Upload security policy for the MAAT model-intake path.

This module is the single place that decides **what a user is allowed to upload**
and **where it may be written**. Everything else in ``app/onboarding`` asks this
module rather than re-deciding, so the policy cannot drift between the validate
and the create-audit paths.

The threat this guards against
------------------------------
``joblib.load`` is built on ``pickle``, and unpickling **executes code contained
in the file**. A malicious ``.joblib`` is therefore remote code execution, not
merely bad data. No amount of extension checking changes that: the file extension
is chosen by whoever produced the file.

This prototype accepts ``.joblib`` regardless, because auditing a real fitted
scikit-learn estimator is the entire point of the exercise. What it does instead
is refuse to pretend the risk is absent:

* the risk is stated verbatim to the user (:data:`JOBLIB_SECURITY_WARNING`);
* the user must **explicitly acknowledge** it before any deserialisation happens
  (:func:`require_acknowledgement`), and the acknowledgement is recorded in the
  audit run's manifest;
* deserialisation is ordered *after* structural validation, so a file that was
  never going to be auditable is rejected without ever being unpickled;
* the safer production alternatives are named rather than left implicit
  (:data:`PRODUCTION_HARDENING_NOTE`).

Formats that are pure code-execution vectors with no offsetting purpose here --
``.py``, ``.pkl``, ``.pickle``, archives, executables -- are refused outright, as
are remote URLs. That is not security theatre: it removes the cases where a user
could get code executed *without* ever seeing the joblib warning.

Path safety
-----------
Original filenames are never used as filesystem paths. :func:`generated_filename`
issues a UUID-based name, and :func:`assert_within` re-checks after resolution
that a target really is inside the runtime tree -- so a crafted name containing
``..`` or a drive letter cannot escape ``runtime/``.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Allowed / denied formats
# --------------------------------------------------------------------------- #
MODEL_EXTENSION: Final = ".joblib"
DATASET_EXTENSION: Final = ".csv"

#: Extensions refused before anything is read. Every one of these is either a
#: direct code-execution vector (``.py``, ``.pkl``) or a container that could
#: smuggle one past an extension check (archives).
DENIED_EXTENSIONS: Final[tuple[str, ...]] = (
    ".py", ".pyc", ".pyo", ".pyw", ".pyz",
    ".pkl", ".pickle", ".dill", ".cloudpickle",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".com", ".msi",
    ".ps1", ".sh", ".vbs", ".js", ".jar",
    ".h5", ".hdf5", ".pt", ".pth", ".onnx", ".pmml", ".bin",
)

#: Size ceilings. Generous enough for a real sklearn pipeline, small enough that a
#: single request cannot exhaust the disk of a local prototype.
MAX_MODEL_BYTES: Final = 200 * 1024 * 1024   # 200 MiB
MAX_DATASET_BYTES: Final = 50 * 1024 * 1024  # 50 MiB
MAX_DATASET_ROWS: Final = 200_000
MIN_DATASET_ROWS: Final = 20

#: Shown to the user verbatim, and required to be acknowledged before load.
JOBLIB_SECURITY_WARNING: Final = (
    "Joblib files may execute arbitrary code. Upload only models from trusted "
    "sources. This local academic prototype must not accept untrusted model files "
    "in production."
)

PRODUCTION_HARDENING_NOTE: Final = (
    "A production implementation must not deserialise user-supplied pickles in the "
    "application process. It should run model loading and inference inside an "
    "isolated sandbox (a separate container or VM, no network egress, read-only "
    "filesystem, dropped privileges, CPU/memory limits) and prefer formats that do "
    "not carry executable payloads -- ONNX for the computation graph, or skops, "
    "which reconstructs scikit-learn estimators from an allow-list of types instead "
    "of executing arbitrary opcodes. This prototype does neither: it loads the "
    "uploaded model in-process, which is acceptable only because the operator is "
    "also the person supplying the file."
)

ACKNOWLEDGEMENT_ERROR: Final = (
    "The model-upload security warning must be explicitly acknowledged before the "
    "uploaded .joblib file can be deserialised. No model has been loaded."
)

_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class UploadRejected(Exception):
    """
    An upload violates the intake policy.

    Carries a machine-readable ``code`` alongside the human message so the API can
    map it to a stable error identifier and the UI can react to specific cases
    (e.g. prompting for the acknowledgement) rather than string-matching.
    """

    def __init__(self, code: str, message: str, hint: str | None = None):
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)


class AcknowledgementRequired(UploadRejected):
    """
    The joblib security warning was not acknowledged.

    The warning itself is repeated in the hint rather than merely referred to. A
    client that reaches this refusal without having rendered the warning -- a
    script, a direct API call, a UI whose first request failed -- would otherwise be
    told to acknowledge something it has never been shown.
    """

    def __init__(self) -> None:
        super().__init__(
            "acknowledgement_required",
            ACKNOWLEDGEMENT_ERROR,
            hint=(
                "Send security_acknowledged=true to confirm you accept this risk: "
                f"{JOBLIB_SECURITY_WARNING}"
            ),
        )


# --------------------------------------------------------------------------- #
# Filename / extension checks
# --------------------------------------------------------------------------- #
def looks_like_url(value: str) -> bool:
    """True for anything that is a remote reference rather than a local filename."""
    lowered = (value or "").strip().lower()
    return bool(
        re.match(r"^[a-z][a-z0-9+.\-]*://", lowered)
        or lowered.startswith("\\\\")  # UNC share
    )


def safe_label(original_name: str, fallback: str = "upload") -> str:
    """
    Reduce a user-supplied filename to a short, inert display label.

    The result is **only ever displayed or stored as metadata** -- never used to
    build a path. Directory components are dropped first so that
    ``../../etc/passwd`` cannot survive as anything but ``passwd``.
    """
    tail = re.split(r"[\\/]", (original_name or "").strip())[-1]
    cleaned = _SAFE_LABEL.sub("_", tail).strip("._-")
    return (cleaned or fallback)[:120]


def check_extension(original_name: str, expected: str, kind: str) -> None:
    """
    Enforce the extension allow-list for one upload.

    Order matters: URL and denied-format checks run before the allow-list so the
    user gets the specific reason ("Python files are not accepted") rather than a
    generic "expected .csv".
    """
    name = (original_name or "").strip()
    if not name:
        raise UploadRejected(
            "missing_filename", f"The uploaded {kind} has no filename."
        )
    if looks_like_url(name):
        raise UploadRejected(
            "remote_source_rejected",
            f"Remote sources are not accepted for the {kind}: '{name}'.",
            hint="Download the file yourself, inspect it, then upload the local file.",
        )

    suffixes = [s.lower() for s in Path(safe_label(name)).suffixes]
    for denied in DENIED_EXTENSIONS:
        if denied in suffixes and denied != expected:
            raise UploadRejected(
                "denied_file_type",
                f"'{denied}' files are not accepted. The {kind} must be a "
                f"'{expected}' file.",
                hint=(
                    "Archives, pickles, scripts and executables are refused because "
                    "they can execute code or hide a payload behind an accepted "
                    "extension."
                ),
            )

    if not suffixes or suffixes[-1] != expected:
        raise UploadRejected(
            "unsupported_file_type",
            f"The {kind} must be a '{expected}' file; got '{name}'.",
            hint=f"Only {expected} is supported for the {kind} in this prototype.",
        )


def check_size(size_bytes: int, limit: int, kind: str) -> None:
    """Reject empty and oversized uploads."""
    if size_bytes <= 0:
        raise UploadRejected("empty_file", f"The uploaded {kind} is empty (0 bytes).")
    if size_bytes > limit:
        raise UploadRejected(
            "file_too_large",
            f"The {kind} is {size_bytes / 1048576:.1f} MiB, which exceeds the "
            f"{limit / 1048576:.0f} MiB limit for this prototype.",
        )


def require_acknowledgement(acknowledged: bool) -> None:
    """
    Gate deserialisation on the user's explicit acknowledgement.

    Called immediately before ``joblib.load``. Nothing in the intake path may
    unpickle an uploaded file without passing through here first.
    """
    if not acknowledged:
        raise AcknowledgementRequired()


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def new_upload_id() -> str:
    """A fresh, unguessable id for one upload bundle."""
    return uuid.uuid4().hex


def generated_filename(kind: str, extension: str) -> str:
    """
    The name an upload is actually stored under.

    Generated, never derived from user input: the original filename is recorded as
    metadata only. This removes path traversal, collision and case-folding
    problems in one step, and guarantees an upload can never land on the name of
    an existing file.
    """
    return f"{kind}_{uuid.uuid4().hex}{extension}"


def assert_within(target: Path, root: Path) -> Path:
    """
    Confirm ``target`` resolves inside ``root``, and return the resolved path.

    A belt-and-braces check. Names are already generated rather than taken from
    the user, so this should be unreachable -- which is exactly why it raises
    loudly instead of silently correcting: reaching it means an assumption
    upstream has broken.
    """
    resolved_root = root.resolve()
    resolved = (root / target).resolve() if not target.is_absolute() else target.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise UploadRejected(
            "path_escape",
            "Refusing to write outside the runtime directory.",
            hint=f"Target {resolved} is not inside {resolved_root}.",
        )
    return resolved
