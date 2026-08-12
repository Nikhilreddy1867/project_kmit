"""
app/onboarding
==============
Model intake for MAAT: accepting a trusted local ``.joblib`` model and a labelled
CSV, validating them, and producing a new audit run under ``runtime/``.

Modules
-------
``security.py``        the single place that decides what may be uploaded and where it
                       may be written: extension allow-list, size limits, UUID
                       filenames, path containment, and the joblib acknowledgement gate.
``runtime_store.py``   the only writable storage. Exclusive-create writes under
                       ``runtime/``; no path it can produce touches ``data/``,
                       ``models/``, ``predictions/`` or ``results/``.
``schemas.py``         Pydantic models for intake, validation and audit results.
``upload_service.py``  stores uploads and validates the dataset structurally.
``model_validator.py`` the only caller of ``joblib.load``; establishes capabilities by
                       inspection and checks feature compatibility.
``audit_service.py``   runs inference and computes performance, fairness,
                       explainability, risk and the run's artefacts.

Ordering guarantee
------------------
The uploaded pickle is deserialised **last**: after the dataset has parsed and
validated, and only once the security warning has been explicitly acknowledged. A
submission that could never be audited is therefore rejected without executing it.

Submodules are not imported here on purpose. ``audit_service`` depends on
:mod:`app.gates`, which in turn reads :mod:`app.onboarding.runtime_store` -- importing
eagerly would make that a package-initialisation cycle for no benefit. Import the
submodule you need directly.
"""

__all__ = [
    "audit_service",
    "model_validator",
    "runtime_store",
    "schemas",
    "security",
    "upload_service",
]
