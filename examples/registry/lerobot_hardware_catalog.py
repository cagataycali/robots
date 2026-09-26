#!/usr/bin/env python3
"""End-to-end: every robot a driver can build is reachable through ``strands_robots.Robot``.

``strands_robots`` is a thin natural-language + policy layer over two sets of
hardware drivers: LeRobot's, and this package's own native ones. This example
walks the join of both - ``list_driver_coverage()`` - and reports, per robot,
which ``driver=`` values can build it and the LeRobot ``robot_type`` where one
exists. A robot reached only by a native driver has no LeRobot type at all, so
the join is wider than the registry's own ``hardware`` declarations. It then
builds a real Unitree G1 in simulation (no hardware, no GPU) to prove the same
factory drives the catalog's most complex robot.

Everything here goes through the ``strands_robots`` public API - the registry
read helpers, the driver-coverage join and the ``Robot()`` factory - never
``import lerobot`` directly. That is the whole point: users program against
``Robot("g1")``, not the driver.

Run:
    python examples/registry/lerobot_hardware_catalog.py        # hardware catalog
    python examples/registry/lerobot_hardware_catalog.py --g1   # focus: Unitree G1 in sim
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


def show_catalog() -> int:
    """List every robot a driver can build, and which driver builds it."""
    from strands_robots.drivers import list_driver_coverage, list_native_drivers
    from strands_robots.registry import format_robot_table, get_hardware_type

    coverage = {name: drivers for name, drivers in list_driver_coverage().items() if drivers}
    native = list_native_drivers()
    print(f"strands_robots reaches hardware for {len(coverage)} robot(s): the join of")
    print("LeRobot's robot types and this package's own native drivers.\n")

    header = f"{'name':<16} {'driver=':<18} {'lerobot_type':<24} native driver"
    print(header)
    print("-" * len(header))
    for name, drivers in coverage.items():
        print(f"{name:<16} {' '.join(drivers):<18} {get_hardware_type(name) or '-':<24} {native.get(name, '-')}")

    print("\nDrive a robot with a LeRobot type for real with, e.g.:")
    print("    from strands_robots import Robot")
    print("    arm = Robot('so100', mode='real', port='/dev/ttyACM0')")
    print("    arm('pick up the red cube', policy_port=8080)")
    print("\nA robot listed 'strands' only has no LeRobot type, so name its driver:")
    print("    arm = Robot('vx300s', mode='real', driver='strands', port='/dev/ttyUSB0')")
    print("\nFull registry (sim + real). Its Real column is the registry's own hardware")
    print("declaration, which is narrower than the join above:\n")
    print(format_robot_table())
    return 0


def show_g1() -> int:
    """Build a Unitree G1 in simulation through the same ``Robot()`` factory.

    The G1 is the catalog's most complex robot - a 29-DOF humanoid. Its registry
    entry declares ``hardware.driver = "strands"``, so ``mode='real'`` builds the
    native CycloneDDS driver and motion goes through its FSM-gated tool bundle;
    here we use the default ``mode='sim'`` so it runs in MuJoCo with no hardware
    and no GPU.
    """
    from strands_robots import Robot
    from strands_robots.registry import get_hardware_type, get_robot, resolve_name

    print("=== Unitree G1 (29-DOF humanoid) ===\n")
    canonical = resolve_name("g1")  # alias -> canonical
    info = get_robot(canonical)
    print(f"  alias 'g1' resolves to:  {canonical}")
    print(f"  description:             {info['description']}")
    print(f"  category:                {info['category']}")
    print(f"  lerobot hardware type:   {get_hardware_type(canonical)}")
    print(f"  sim asset (MuJoCo):      {info.get('asset', {}).get('model_xml')}")

    print("\n  Building it in simulation via Robot('g1') ...")
    sim = Robot("g1", mesh=False)
    try:
        print(f"  -> {type(sim).__name__} ready (MuJoCo backend, no hardware).")
    finally:
        sim.destroy()

    print("\n  Drive it for real over CycloneDDS:")
    print("    g1 = Robot('g1', mode='real', port='192.168.123.164')")
    print("    # network_interface='eth0' by default; motion is FSM-gated, see")
    print("    # docs/hardware/unitree-g1.md.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g1", action="store_true", help="Focus on the Unitree G1 humanoid (builds it in sim).")
    args = p.parse_args()

    try:
        return show_g1() if args.g1 else show_catalog()
    except ImportError as e:
        print(
            f"This example needs the simulation extra: pip install 'strands-robots[sim-mujoco]'\n  {e}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
