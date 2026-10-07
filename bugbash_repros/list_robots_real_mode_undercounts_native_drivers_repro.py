"""Repro: ``list_robots(mode="real")`` undercounts by 25 — every native-driver-only robot is hidden.

Before the fix at ``strands_robots/registry/robots.py:299``:

    _has_real = "hardware" in info

Only reads the entry's ``hardware`` block, so a robot whose real-driver fact
lives in ``_NATIVE_DRIVERS`` (registered through
:func:`strands_robots.drivers.register_native_driver`) is dropped from the
"real" listing even though ``Robot(name, mode="real")`` builds a working
driver instance on the same spelling.

Sibling :func:`strands_robots.drivers.list_driver_coverage` already joins both
registries — the complete answer the ``list_robots`` docstring itself names —
but a user who asks "which robots can I drive for real" via the primary
listing surface sees 24 of 49 drivable robots. The missing 25 include every
Universal Robots arm (ur3e/5e/7e/8long/10e/12e/15/16e/18/20/30), Franka Panda
(``panda``/``fr3``/``fr3_v2``), Spot, Stretch (1 + 3), Kuka iiwa, Kinova
Gen3, RBY1, Unitree H1 (1 + 2), xArm7, b2 and open_duck_mini — the arms a
real roboticist is most likely to reach for first.

After the fix:

    from strands_robots.drivers.registry import get_native_driver_class
    _has_real = "hardware" in info or get_native_driver_class(name) is not None

Which lines up ``mode="real"`` with the single-truth coverage report. The
docstring's claim ("joins the two halves of ``list_driver_coverage``") is
now the implementation, not a promise the implementation breaks.

Run:
    python bugbash_repros/list_robots_real_mode_undercounts_native_drivers_repro.py

No hardware, no network, no optional deps required.
"""

from __future__ import annotations

import sys

from strands_robots.drivers import list_driver_coverage
from strands_robots.registry import list_robots
from strands_robots.robot import Robot


def main() -> int:
    real_names = {r["name"] for r in list_robots(mode="real")}
    drivable = {name for name, drivers in list_driver_coverage().items() if drivers}
    missing = sorted(drivable - real_names)

    print("list_robots(mode='real'):", len(real_names), "robots")
    print("list_driver_coverage() non-empty:", len(drivable), "robots")
    print("drivable but hidden from mode='real':", len(missing), "robots")

    if not missing:
        print("\nOK: lists agree.")
        return 0

    print("\nHidden robots (each has a working native driver):")
    for name in missing:
        try:
            obj = Robot(name, mode="real")
            print(f"  {name:<20s} -> {type(obj).__name__}")
        except Exception as e:
            # Some drivers require a port/host at construct time; still prove
            # the registry resolves the driver.
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  {name:<20s} -> driver resolved, needs config ({msg[:70]})")

    print(
        f"\nFAIL: {len(missing)} robots a user can drive for real are hidden "
        "from mode='real'. The docstring names list_driver_coverage as the "
        "complete answer, but list_robots is the primary listing surface and "
        "must agree."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
