"""Repro: Teleoperator('<leader>') missing-port refusal has no serial-scan hint.

Teleoperator and Robot both front lerobot's serial-bus dataclasses, but their
missing-``port`` refusals disagree on who the audience is.

Robot('so101', mode='real', driver='strands')   # Feetech bus, no port
    ^-- ValueError that NAMES this host's 7 serial devices and the ones that
        look like a servo bus, teaches "port is a position not an identity",
        and spells out the fully-formed call to make.

Teleoperator('so101_leader')                     # Feetech bus, no port
    ^-- ValueError that re-wraps lerobot's raw
            "SOLeaderTeleopConfig.__init__() missing 1 required positional
             argument: 'port'. Config: {}"
        and stops. No scan. No candidate list. Not even the docs-canonical form
        Teleoperator('so101_leader', port='/dev/ttyACM0').

Both call sites are inside the same package; the Robot side already imports
    scan_serial_devices + describe_serial_candidates
from strands_robots._serial_discovery and uses them for exactly this failure
mode at robot.py:537.

The two refusals are two sides of the same sentence: the Teleoperator leader
IS a serial device of the same family as the follower arm the Robot refusal
teaches, so the same candidate scan belongs on both sides.

Upstream cites
--------------
- strands_robots/teleoperator.py:215-219  (Teleoperator config construction,
  single ``except (TypeError, ValueError)`` branch that re-wraps the raw lerobot
  error without augmentation)
- strands_robots/robot.py:532-539          (the sibling site that DOES scan)
- strands_robots/_serial_discovery.py:177-202 (reusable helper, never-empty,
  tolerant of hosts with no serial)
- docs/learn/hardware/teleoperation.md:52  (docs form the user is copying:
  ``leader = Teleoperator('so101_leader', port='/dev/ttyACM1', id='blue')``)
- docs/learn/hardware/feetech-arms.md:32   (quickstart row)

Run
---
$ python3 bugbash_repros/teleoperator_missing_port_no_scan_repro.py
"""

from __future__ import annotations

import sys


def _print_refusal(label: str, exc: BaseException) -> None:
    print(f"--- {label} ---")
    print(f"{type(exc).__name__}: {exc}")
    print()


def main() -> int:
    # Robot side: documented servo-bus call missing port. Already scans.
    print("=" * 72)
    print("Robot('so101', mode='real', driver='strands')  <-- Feetech, no port")
    print("=" * 72)
    from strands_robots import Robot

    robot_msg = ""
    try:
        Robot("so101", mode="real", driver="strands")
    except ValueError as exc:
        robot_msg = str(exc)
        _print_refusal("Robot refusal (scans the host)", exc)

    scan_hint_in_robot = (
        "serial device" in robot_msg or "servo-bus" in robot_msg or "serial devices are present" in robot_msg
    )
    print(f"[Robot] scan-hint present: {scan_hint_in_robot}")
    print()

    # Teleoperator side: documented sibling call missing port. No scan.
    print("=" * 72)
    print("Teleoperator('so101_leader')  <-- Feetech leader, no port")
    print("=" * 72)
    from strands_robots import Teleoperator

    teleop_msg = ""
    try:
        Teleoperator("so101_leader")
    except ValueError as exc2:
        teleop_msg = str(exc2)
        _print_refusal("Teleoperator refusal (no scan)", exc2)

    scan_hint_in_teleop = (
        "serial device" in teleop_msg or "servo-bus" in teleop_msg or "serial devices are present" in teleop_msg
    )
    print(f"[Teleoperator] scan-hint present: {scan_hint_in_teleop}")
    print()

    # Static proof of code-level asymmetry
    print("=" * 72)
    print("Static proof: same helper imported on one side, missing on the other")
    print("=" * 72)
    import strands_robots.robot as robot_mod
    import strands_robots.teleoperator as teleop_mod

    robot_src = (robot_mod.__file__ or "")
    teleop_src = (teleop_mod.__file__ or "")

    import pathlib

    robot_imports_scan = "describe_serial_candidates" in pathlib.Path(robot_src).read_text()
    teleop_imports_scan = "describe_serial_candidates" in pathlib.Path(teleop_src).read_text()

    print(f"strands_robots/robot.py        imports describe_serial_candidates: {robot_imports_scan}")
    print(f"strands_robots/teleoperator.py imports describe_serial_candidates: {teleop_imports_scan}")
    print()

    # Asymmetry assertion (bug: pre-patch this is True)
    bug_present = scan_hint_in_robot and not scan_hint_in_teleop
    print("=" * 72)
    print(f"BUG PRESENT (asymmetric scan-hint on sibling missing-port refusals): {bug_present}")
    print("=" * 72)

    # Also show that the bare lerobot message is what the user sees
    print()
    print("Verbatim end-user teleop message (what the docs call out as 'required'):")
    print(f"    {teleop_msg}")
    print()
    print("Compared to the Robot side the user would have seen on the follower:")
    print(f"    {robot_msg}")

    return 0 if bug_present else 1


if __name__ == "__main__":
    sys.exit(main())
