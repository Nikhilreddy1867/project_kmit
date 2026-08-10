"""
app/registry/cli.py
===================
Command-line entry point for the governance audit registry.

    python -m app.registry.cli register     # create or refresh the current run
    python -m app.registry.cli list         # list registered runs
    python -m app.registry.cli integrity    # verify the active run's checksums

``register`` is **idempotent**: running it repeatedly against unchanged evidence
refreshes the same content-addressed run rather than creating duplicates. It writes
only to the registry database under ``runtime/`` and opens the evidence read-only.
"""

from __future__ import annotations

import argparse
import sys

from app.registry import db as registry_db
from app.registry import service


def _cmd_register(args: argparse.Namespace) -> int:
    result = service.register_run(db_path=args.db, model_name=args.model)
    print("=" * 74)
    print("GOVERNANCE AUDIT REGISTRY - register")
    print("=" * 74)
    print(f"  run id          : {result['run_id']}")
    print(f"  action          : {result['action']}")
    print(f"  message         : {result['message']}")
    print(f"  created at      : {result['created_at']}")
    print(f"  refreshed at    : {result['refreshed_at']}  (refresh #{result['refresh_count']})")
    print(f"  artefacts       : {result['artifact_count']}")
    print(f"  evidence digest : {result['evidence_digest']}")
    if result["superseded_run_ids"]:
        print(f"  superseded      : {', '.join(result['superseded_run_ids'])}")
    print(f"  database        : {result['database']}")
    print()
    print("Evidence under data/, models/, predictions/ and results/ was read only.")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        payload = service.list_runs(db_path=args.db, status=args.status)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{payload['count']} run(s) in {payload['database']}"
          f" | active: {payload['active_run_id'] or 'none'}")
    for run in payload["runs"]:
        print(
            f"  {run['run_id']}  {run['status']:<11} {run['created_at']}  "
            f"{run['model_name']} {run['model_version']}  "
            f"{run['artifact_count']} artefacts"
        )
    return 0


def _cmd_integrity(args: argparse.Namespace) -> int:
    try:
        run_id = args.run_id
        if not run_id:
            listing = service.list_runs(db_path=args.db)
            run_id = listing["active_run_id"]
            if not run_id:
                print("ERROR: no active run. Run `register` first.", file=sys.stderr)
                return 1
        result = service.check_integrity(run_id, db_path=args.db)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except service.RunNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"run {result['run_id']} - integrity: {result['integrity_status'].upper()}")
    print(
        f"  {result['verified_count']} verified | {result['changed_count']} changed | "
        f"{result['missing_count']} missing  (of {result['artifacts_checked']})"
    )
    for path in result["changed_files"]:
        print(f"  CHANGED : {path}")
    for path in result["missing_files"]:
        print(f"  MISSING : {path}")
    return 0 if result["integrity_ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.registry.cli",
        description=(
            "Local governance audit registry. Records the current audit as one "
            "review run with SHA-256 checksums of every referenced artefact. "
            "Read-only with respect to the evidence."
        ),
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Registry database path (default: runtime/governance_registry.db, or "
        f"${registry_db.DB_ENV_VAR}).",
    )
    sub = parser.add_subparsers(dest="command")

    register = sub.add_parser(
        "register", help="Create or refresh the current audit-run record (idempotent)."
    )
    register.add_argument(
        "--model",
        default=service.SUBJECT_MODEL,
        help=f"Subject model for the run (default: {service.SUBJECT_MODEL}).",
    )
    register.set_defaults(func=_cmd_register)

    listing = sub.add_parser("list", help="List registered runs.")
    listing.add_argument("--status", default=None, choices=registry_db.VALID_STATUSES)
    listing.set_defaults(func=_cmd_list)

    integrity = sub.add_parser("integrity", help="Verify a run's artefact checksums.")
    integrity.add_argument(
        "run_id", nargs="?", default=None, help="Run id (default: the active run)."
    )
    integrity.set_defaults(func=_cmd_integrity)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # Bare invocation does the useful thing rather than printing usage.
        args = parser.parse_args(["register", *(argv or [])])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
