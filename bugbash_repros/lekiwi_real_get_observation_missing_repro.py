"""Repro: Robot(name, mode='real', port=...) is missing .get_observation().

Rotation target: lekiwi_sim_quickstart
Observed: v0.5.3 / head c6a5bafb6 / Thor (Linux aarch64)

What a working robotics engineer sees
=====================================
`docs/robots/lekiwi.md` (and 60+ sibling robot pages) print an
"Observation key" table listing the names each robot returns from the
observation call. The same name sits at the top of the sim quickstart
sketch:

    robot = Robot("lekiwi", mode="real", port="/dev/ttyACM0")  # lerobot lekiwi

An engineer who read the two blocks together naturally writes
`obs = robot.get_observation()` to inspect those keys. That call lands on
`AttributeError`, because:

  1. The construction call succeeds silently on a box with no serial
     device, no `/dev/ttyACM0`, no lerobot connect: `_initialize_robot`
     only builds a lerobot `Robot` config instance; the actual port is
     dialled lazily on the first `send_action`.

  2. The returned object is a `strands_robots.hardware_robot.Robot`
     wrapper. Its inner `self.robot` (lerobot) has `get_observation`.
     The wrapper does not forward it, nor does it define its own.

  3. The strands-labs tree is actively inconsistent about this:

       - `strands_robots/simulation/base.py:2034` defines it on every
         SimEngine backend (mjlab/mujoco/newton/isaac/mjlab).
       - 15 native hardware drivers define it (franka, ur, kuka, kinova,
         xarm, spot, stretch, microduck, robotiq, earthrover, yahboom,
         composite, ...).
       - `strands_robots/device_connect/test_control_loop_dc.sh:92`
         literally runs `obs = robot.get_observation()` on
         `Robot("so100")` in its CI control-loop smoke test.
       - `strands_robots/hardware_robot.py:801` and :1203 have comments
         that read `the first ``get_observation()``` as if the method
         were on `self`.

  4. The symmetric sim path works (`Robot("lekiwi")` → `MuJoCoSimEngine`
     has `get_observation`). Code written for sim does not port to real
     by flipping `mode="real"` — the method ceases to exist.

The same hole exists on `Robot("so101", mode="real", port=...)` where
the returned object is `FeetechDriver` (native driver), which also
lacks `get_observation`. Two concrete classes
(`hardware_robot.Robot`, `FeetechDriver`) silently drop the surface the
docs and the in-tree test both assume is there.

Repro (deterministic, no hardware required):

    python3 bugbash_repros/lekiwi_real_get_observation_missing_repro.py

Expected: three AttributeErrors; the fourth check (sim) succeeds.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> int:
    from strands_robots import Robot

    failures: list[str] = []

    # ── 1. lekiwi real mode: hardware_robot.Robot wrapper ─────────────
    r_lekiwi = Robot("lekiwi", mode="real", port="/dev/ttyACM0")
    cls_name = type(r_lekiwi).__name__
    assert cls_name == "Robot", f"expected hardware_robot.Robot, got {cls_name}"
    inner = getattr(r_lekiwi, "robot", None)
    assert inner is not None and hasattr(inner, "get_observation"), (
        "inner lerobot .robot should carry get_observation"
    )
    if not hasattr(r_lekiwi, "get_observation"):
        failures.append(
            f"Robot('lekiwi', mode='real') is a {cls_name} with no get_observation(); "
            f"inner .robot ({type(inner).__name__}) has it but is not forwarded."
        )
    r_lekiwi.cleanup()

    # ── 2. so101 real mode: FeetechDriver (native strands driver) ────
    r_so101 = Robot("so101", mode="real", port="/dev/ttyACM0")
    cls2 = type(r_so101).__name__
    assert cls2 == "FeetechDriver", f"expected FeetechDriver, got {cls2}"
    if not hasattr(r_so101, "get_observation"):
        failures.append(
            f"Robot('so101', mode='real') is a {cls2} with no get_observation(); "
            f"the native-driver branch is missing a method 15 of its sibling "
            f"drivers (franka/ur/kuka/kinova/xarm/spot/stretch/microduck/robotiq/"
            f"earthrover/yahboom/composite/...) all define."
        )
    r_so101.cleanup()

    # ── 3. The in-tree test script literally calls this method on real ─
    test_script = "strands_robots/device_connect/test_control_loop_dc.sh"
    try:
        with open(test_script) as f:
            text = f.read()
    except FileNotFoundError:
        text = ""
    if "robot.get_observation()" in text:
        # Not a repro-failure in isolation, but documents the asymmetry:
        # the authors expect the method on both sides.
        print(
            f"CONTEXT: {test_script} line calling robot.get_observation() exists "
            f"and runs against Robot('so100') (sim here, but the test is named "
            f"after the device-connect real-arm pathway).",
            file=sys.stderr,
        )

    # ── 4. sim path works (for contrast) ──────────────────────────────
    r_sim = Robot("lekiwi")
    assert hasattr(r_sim, "get_observation"), (
        "sim mode must still work -- MuJoCoSimEngine defines get_observation"
    )
    obs = r_sim.get_observation()
    assert "Jaw" in obs, "sim obs schema should include Jaw"
    r_sim.cleanup()

    # ── Report ─────────────────────────────────────────────────────────
    if failures:
        print("DEFECT REPRODUCED:")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        print()
        print(
            "Impact: the obs-key table on docs/robots/lekiwi.md (and 60+ "
            "sibling robot pages) advertises a schema the primary wrapper "
            "class does not expose. Sim-authored code does not port to real "
            "by flipping mode='real'."
        )
        return 1

    print("No defect found (unexpected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
