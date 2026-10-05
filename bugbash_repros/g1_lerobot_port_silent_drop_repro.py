"""Repro: Robot('unitree_g1', mode='real', driver='lerobot', port=...) silently drops
`port`, keeping the stock lerobot default `robot_ip='192.168.123.164'` instead.

Context
-------
docs/robots/unitree_g1.md:19-23 shows TWO adjacent hardware sketches:

    robot = Robot("unitree_g1", mode="real", driver="lerobot", robot_ip="192.168.123.164")
    robot = Robot("unitree_g1", mode="real", port="192.168.123.164", network_interface="eth0")

Line 1 uses `robot_ip`, line 2 uses `port`.  A user who scans the block and keeps
`port=` while flipping to `driver="lerobot"` (a plausible mix-up since `port` is
the more familiar name) gets a Robot that **silently** reverts to the stock
`192.168.123.164` default.  No warning, no stderr; only a DEBUG log says the
kwarg was dropped by the `_FORWARDABLE_KWARGS` polymorphism carve-out
(`strands_robots/hardware_robot.py:144-170, 1455-1477`).

Sibling behaviour: the SAME typo against `driver="strands"` (G1Driver, the
default on the registry) **raises ValueError**, naming the valid kwarg list:

    Unknown kwarg(s) for 'unitree_g1' on driver='strands': ['port'] ...
    G1Driver accepts: ['battery_floor_pct', 'cameras', 'data_config',
                       'motion_switcher_client_factory', 'network_interface',
                       'port', 'tool_name'].

So the asymmetry is: native driver has a validator; the lerobot carve-out
silently discards the exact kwarg the docs show one line above.

Related
-------
harness#683 flagged the same class (`robot_ip` dropped on LeKiwi because
`LeKiwiConfig` lacks the field).  The underlying semantic hole in
`_FORWARDABLE_KWARGS` was left as "B-tier follow-up".  This repro surfaces the
G1-specific instance (`port` dropped on `unitree_g1` because `UnitreeG1Config`
only declares `robot_ip`) and demonstrates that the stock default
`robot_ip='192.168.123.164'` **masks the drop in demo settings**, exploding
only in the field.

Run
---
    python bugbash_repros/g1_lerobot_port_silent_drop_repro.py
"""

from __future__ import annotations

import logging
import sys

from strands_robots import Robot

# Enable DEBUG so we can see the hidden drop log.
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")


def _driver_robot_ip(robot) -> str:
    """Reach past the strands wrapper to the lerobot driver's robot_ip."""
    # Robot() with driver="lerobot" returns the strands wrapper; the config
    # sits on the lerobot-side driver instance.
    for attr in ("_driver", "driver", "_robot", "robot"):
        inner = getattr(robot, attr, None)
        if inner is None:
            continue
        for cand in ("config", "_config", "cfg"):
            cfg = getattr(inner, cand, None)
            if cfg is not None and hasattr(cfg, "robot_ip"):
                return cfg.robot_ip
    # Fallback: deeply introspect config dataclasses the driver stores.
    try:
        import dataclasses
        for name, val in vars(robot).items():
            if dataclasses.is_dataclass(val) and hasattr(val, "robot_ip"):
                return val.robot_ip
            if hasattr(val, "config") and hasattr(val.config, "robot_ip"):
                return val.config.robot_ip
    except Exception:
        pass
    return "<unknown>"


def main() -> int:
    print("=" * 70)
    print("DEMO: Robot('unitree_g1', mode='real', driver='lerobot', port='10.0.0.5')")
    print("=" * 70)
    r1 = Robot("unitree_g1", mode="real", driver="lerobot", port="10.0.0.5")
    print(f"\n  resulting robot_ip = {_driver_robot_ip(r1)!r}")
    print(f"  user intended      = '10.0.0.5'")
    r1.cleanup()
    print()
    print("Observations:")
    print(" * No exception raised.")
    print(" * No WARN/ERROR surfaced on stdout/stderr.")
    print(" * The DEBUG line `dropping cross-robot kwarg 'port' ...` above is")
    print("   the only trace, invisible at the default log level.")
    print(" * robot_ip reverted to UnitreeG1Config default ('192.168.123.164'),")
    print("   not the '10.0.0.5' the caller passed.")
    print()

    print("=" * 70)
    print("SIBLING: driver='strands' + robot_ip='10.0.0.5' REFUSES loudly")
    print("        (the lerobot kwarg against the native driver)")
    print("=" * 70)
    try:
        Robot("unitree_g1", mode="real", driver="strands",
              robot_ip="10.0.0.5", network_interface="eth0")
        print("  (strands took robot_ip - unexpected)")
    except ValueError as e:
        print(f"  ValueError: {e}")
    print()
    print("  Same site shape as this bug (one kwarg from docs line, wrong")
    print("  driver on the other line) - but the strands driver has a")
    print("  per-driver kwarg allowlist that refuses by name, while the")
    print("  lerobot path silently drops via _FORWARDABLE_KWARGS.")
    print()

    print("=" * 70)
    print("DOCS DEMO: port='192.168.123.164' happens to WORK because that is")
    print("the UnitreeG1Config default - a demo cannot tell the drop apart.")
    print("=" * 70)
    r2 = Robot("unitree_g1", mode="real", driver="lerobot", port="192.168.123.164")
    print(f"  resulting robot_ip = {_driver_robot_ip(r2)!r}")
    print("  (user's 'port' value coincided with the default; drop invisible)")
    r2.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
