"""Repro: SpotDriver.connect_eagerly reports a BOSDYN_CLIENT_* credentials
red-herring instead of the real cause (no [spot] extra installed), because
the credentials gate runs BEFORE the SDK import gate.

Reproduce on `pip install strands-robots` (no extras, nothing in the env):

    $ pip install strands-robots
    $ unset BOSDYN_CLIENT_USERNAME BOSDYN_CLIENT_PASSWORD
    $ python bugbash_repros/spot_connect_eagerly_wrong_reason_cred_before_sdk.py

Expected: a message naming the install path, mirroring every sibling
hardware driver (xarm/crazyflie/rby1/ur all do this today):

    "the Spot SDK is not importable (No module named 'bosdyn'). Install it
     with: pip install 'strands-robots[spot]'"

Actual (strands-labs/robots@92d1d5136, strands_robots/drivers/spot.py:220-243):

    "SpotDriver: set BOSDYN_CLIENT_USERNAME and BOSDYN_CLIENT_PASSWORD to
     the robot's credentials"

The user flips the two env vars that the message names, re-runs, and only
THEN learns they need the extra -- a two-step dance where sibling drivers
make the one-step hint right away. The useful message exists in the file
(line 107 of spot.py) but is only reachable after the credentials pass.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def main() -> int:
    # Scrub the env so this script is reproducible regardless of what the
    # user has in their shell.
    for v in ("BOSDYN_CLIENT_USERNAME", "BOSDYN_CLIENT_PASSWORD"):
        os.environ.pop(v, None)
    os.environ.setdefault("STRANDS_SKIP_ZENOH", "1")

    assert importlib.util.find_spec("bosdyn") is None, (
        "This repro assumes bosdyn is NOT installed. Uninstall it (or run on "
        "a fresh env) to see the papercut."
    )

    from strands_robots import Robot

    r = Robot("spot", mode="real", port="192.168.80.3")
    msg = r.connect_eagerly()
    print(f"SpotDriver.connect_eagerly() -> {msg!r}")

    # Compare with a sibling driver on the same conditions.
    r_sibling = Robot("xarm7", mode="real", port="192.168.1.100")
    print(f"XArmDriver.connect_eagerly() -> {r_sibling.connect_eagerly()!r}")

    # Assert the defect: without [spot] installed and no env creds, we expect
    # Spot to cite the install path (like every sibling does), not the env vars.
    if "pip install" in msg and "[spot]" in msg:
        print("PASS: Spot cites the install path -- defect is fixed.")
        return 0
    print(
        "FAIL: Spot pointed at env vars that cannot help; the user has no "
        "bosdyn SDK yet, so BOSDYN_CLIENT_* is a red-herring.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
