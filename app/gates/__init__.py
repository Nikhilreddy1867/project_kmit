"""
app/gates
=========
Governance-as-Code: versioned policy profiles, the five pipeline gates, waivers, the
Conformity Bundle and clause-to-artefact traceability.

Modules
-------
``policies/``          versioned policy JSON. The source of every threshold, control,
                       owner and limitation. Code here never hard-codes a threshold.
``schemas.py``         Pydantic models for policies, gate results, waivers, bundles
                       and traceability.
``policy_engine.py``   pure evaluation: (evidence, policy, waivers) -> gate results.
``conformity_bundle.py`` assembles the evidence package and the traceability matrix.
``service.py``         I/O and orchestration: load evidence, write the four
                       governance artefacts, manage the waiver register.

Statuses
--------
``PASS`` / ``WAIVE`` / ``BLOCK`` / ``NOT_EVALUATED``. ``NOT_EVALUATED`` means the
evidence a control needs does not exist -- it is not a soft pass. ``WAIVE`` records
that a named human accepted an unmet requirement until a stated expiry -- it is not a
pass either.

The Release and Operations gates can only be ``BLOCK`` or ``NOT_EVALUATED``. No
computation in this package can state that a user-uploaded model is production-ready.
"""

from app.gates import conformity_bundle, policy_engine, schemas, service

__all__ = ["conformity_bundle", "policy_engine", "schemas", "service"]
