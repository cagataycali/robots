"""Repro: docs/robots/lekiwi.md quickstart emits a stderr burial WARNING
on every first-touch run; the generator at docs/hooks/robot_pages.py:678
does not propagate the position= value that the warning itself prints.

Fire #58 — rotation target: lekiwi_sim_quickstart
Upstream HEAD at repro: 778876b (v0.5.3 line)

End-user path (copy-paste from docs/robots/lekiwi.md, lines 15-20):

    from strands_robots import Robot
    robot = Robot("lekiwi")
    print(robot.robot_joint_names("lekiwi"))
    robot.cleanup()

Expected: three lines of joint names, no warnings.
Actual:
    WARNING strands_robots.simulation.mujoco.simulation:
      'lekiwi' starts 34.6 mm inside the ground, so the contact solver
       pushes it from the first step. Pass position=[0.0, 0.0, 0.0346]
       to spawn it resting on the ground.

The engine *already knows* the exact position= to pass (computed in
strands_robots/simulation/mujoco/simulation.py:3410-3417, in
_spawn_burial_warning). The docs generator renders the quickstart that
trips the warning but does not consume the fix.

Impact — 13 of 66 sim-capable robots trigger this on their own
docs-generated quickstart:

    ur10e(27.0mm), aero_hand(108.3mm), allegro_hand(44.2mm),
    shadow_hand(5.0mm), asimov_v0(4.7mm), open_duck_mini(26.2mm),
    rby1(2.6mm), aliengo(126.5mm), anymal_c(10.5mm), go1(4.0mm),
    lekiwi(34.6mm), unitree_a1(120.0mm), unitree_go2(3.0mm)

Fix path (docs-side, declarative):
  1. Add optional "spawn_position": [x, y, z] to the 13 affected
     registry entries in strands_robots/registry/robots.json.
  2. In docs/hooks/robot_pages.py:671-680, read spec.get("spawn_position")
     and emit:
        robot = Robot("{name}", position=[x, y, z])

Behavioural fix (sim-side, out of scope) would be to auto-seat buried
assets on add_robot; maintainers chose not to, per the docstring at
simulation.py:3384: "Moving it would change the spawn every caller
already relies on".

Run with:
    cd <repo>
    pip install -e .                   # end-user: pip install strands-robots
    python bugbash_repros/lekiwi_quickstart_burial_repro.py

Exit code is 0 on CURRENT (reproduces the defect — warning on stderr);
assert the capture to make it fail-loud:
    EXPECT_CLEAN=1 python bugbash_repros/lekiwi_quickstart_burial_repro.py
"""

from __future__ import annotations

import io
import logging
import os
import sys


def main() -> int:
    # No surprise GL path on headless Thor.
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Scheduler/agent env leak guard.
    os.environ.pop("SYSTEM_PROMPT", None)

    # Capture WARNING+ from any logger — mimics a user eyeballing stderr.
    captured: list[tuple[str, str]] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                captured.append((record.name, record.getMessage()))

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(_Capture())

    # Exactly what docs/robots/lekiwi.md teaches.
    from strands_robots import Robot

    robot = Robot("lekiwi")
    names = robot.robot_joint_names("lekiwi")
    print(names)
    robot.cleanup()

    print("\n--- captured WARNING+ output ---")
    for logger_name, msg in captured:
        print(f"{logger_name}: {msg}")

    burial = [m for _, m in captured if "inside the ground" in m]
    if not burial:
        print("\nOK — quickstart is clean.")
        return 0

    print(
        "\nDEFECT: docs quickstart emits burial WARNING on the first-touch call.\n"
        "        Generator at docs/hooks/robot_pages.py:678 does not emit\n"
        "        the position= the warning itself prescribes."
    )

    if os.environ.get("EXPECT_CLEAN") == "1":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
