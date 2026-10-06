"""``python -m strands_robots iot <verb>``: provision, re-provision and tear down AWS IoT identities.

Every verb is a thin wrapper over the functions in :mod:`strands_robots.mesh.iot.provision`
and prints the ``export`` lines a process needs afterwards. Exit codes: 0 on success,
1 when AWS refused or the Thing does not exist, 2 on a usage error (argparse).

::

    strands-robots iot provision-robot so101-arm-01
    strands-robots iot provision-operator ops-console-1
    strands-robots iot reprovision so101-arm-01        # rotate to a CSR certificate with CN=<thing>
    strands-robots iot reprovision watchdog-1 --estop-publish keep   # a safety authority keeps its grant
    strands-robots iot withdraw-estop-publish --keep watchdog-1      # dry run: who still holds the grant
    strands-robots iot withdraw-estop-publish --keep watchdog-1 --apply
    strands-robots iot clear-retained-safety --apply   # delete stored stops/releases after the rollout
    strands-robots iot teardown so101-arm-01
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strands-robots iot", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb, help_text in (
        ("provision-robot", "create a robot Thing, its CSR certificate and the strands-robot policy"),
        ("provision-operator", "create an operator Thing, its CSR certificate and the strands-operator policy"),
        (
            "reprovision",
            "rotate an existing Thing's certificate in place to one issued from a local CSR with CN=<thing>; "
            "keeps the Thing, its attributes and its policy attachments",
        ),
        ("teardown", "delete the Thing, its certificates and the local files"),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument("thing_name", help="the Thing name, which is the mesh peer id")
        p.add_argument("--region", default=None, help="AWS region (default: the boto3 session's)")
        p.add_argument("--cert-dir", default=None, help="where the PEM files live (default ~/.strands_robots/iot)")
        if verb == "provision-robot":
            p.add_argument(
                "--estop-publish",
                action="store_true",
                help=(
                    "attach the strands-robot policy: this robot may ORIGINATE and clear a fleet-wide stop. "
                    "Only for a designated safety authority; the default strands-robot-no-estop obeys stops"
                ),
            )
        if verb == "reprovision":
            p.add_argument(
                "--estop-publish",
                choices=("keep", "drop"),
                default=None,
                help=(
                    "what to do when the current certificate carries strands-robot (the fleet-stop publish grant): "
                    "keep it for a designated safety authority, or drop it and rotate onto strands-robot-no-estop. "
                    "Without this flag such a rotation is refused; a certificate without the grant needs no flag"
                ),
            )
    w = sub.add_parser(
        "withdraw-estop-publish",
        help=(
            "move every certificate on strands-robot to strands-robot-no-estop, except the Things named with --keep; "
            "a dry run unless --apply is given"
        ),
    )
    w.add_argument(
        "--keep", action="append", default=[], metavar="THING", help="a safety authority that keeps the grant"
    )
    w.add_argument("--apply", action="store_true", help="change the account; without it the verb only reports")
    w.add_argument("--region", default=None, help="AWS region (default: the boto3 session's)")
    c = sub.add_parser(
        "clear-retained-safety",
        help="delete every retained message under strands/safety/ (a dry run unless --apply is given)",
    )
    c.add_argument("--apply", action="store_true", help="clear them; without it the verb only reports")
    c.add_argument("--region", default=None, help="AWS region (default: the boto3 session's)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one verb; returns the exit code."""
    args = _parser().parse_args(argv)
    from strands_robots.mesh.iot import provision as prov

    try:
        if args.verb in ("withdraw-estop-publish", "clear-retained-safety"):
            report: prov.FleetStopGrantReport | prov.RetainedSafetyReport
            if args.verb == "withdraw-estop-publish":
                report = prov.withdraw_fleet_stop_grant(
                    region=args.region, safety_authorities=args.keep, apply=bool(args.apply)
                )
            else:
                report = prov.clear_retained_safety_messages(region=args.region, apply=bool(args.apply))
            for line in report.lines():
                print(line)
            return 0
        if args.verb == "teardown":
            prov.teardown_thing(args.thing_name, region=args.region, cert_dir=args.cert_dir)
            print(f"{args.thing_name}: Thing, certificates and local files removed")
            return 0
        if args.verb == "provision-robot":
            result = prov.provision_robot(
                args.thing_name,
                region=args.region,
                cert_dir=args.cert_dir,
                allow_estop_publish=bool(args.estop_publish),
            )
        elif args.verb == "provision-operator":
            result = prov.provision_operator(args.thing_name, region=args.region, cert_dir=args.cert_dir)
        else:
            decision = {"keep": True, "drop": False, None: None}[args.estop_publish]
            result = prov.reprovision_thing(
                args.thing_name, region=args.region, cert_dir=args.cert_dir, estop_publish=decision
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - botocore raises its own hierarchy; the message is the report
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"{result.thing_name}: certificate {result.cert_id[:12]} (CN={result.subject_cn}), policy {result.policy_name}"
    )
    print(f"files: {result.cert_path}, {result.key_path}")
    if result.stale_certificates:
        print(
            f"warning: {len(result.stale_certificates)} old certificate(s) could not be removed and are still "
            f"active: {', '.join(result.stale_certificates)} (see the log for the commands)",
            file=sys.stderr,
        )
    if args.verb == "reprovision":
        print("restart the peer that used the old certificate: its MQTT session ended with it")
    for line in result.export_lines():
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry
    sys.exit(main())
