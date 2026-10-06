"""Repro: `spawn_position` is a dead-letter registry field — 13 robots
declare it, zero code reads it, so the default `Robot(name)` call spawns
them inside the ground and the warning tells the user to retype the
number by hand.

Run:
    MUJOCO_GL=egl python bugbash_repros/spawn_position_dead_letter_repro.py

Expected (ideal): `Robot("lekiwi")` consults registry `spawn_position` and
spawns the robot resting on the ground with no warning. The documented
quickstart line `Robot("lekiwi", position=[0.0, 0.0, 0.0346])` becomes
redundant.

Actual on HEAD (`strands-labs/robots@main` as of 2026-11-25):
  * `strands_robots/registry/robots.json` carries `spawn_position` on 13
    robots (lekiwi, ur10e, aero_hand, allegro_hand, shadow_hand,
    asimov_v0, open_duck_mini, rby1, aliengo, anymal_c, go1, unitree_a1,
    unitree_go2).
  * `grep -rn spawn_position strands_robots/` finds **zero** readers in
    the Python source — only `docs/hooks/robot_pages.py:703` consumes it
    to print the `position=` arg in the generated per-robot page.
  * `Robot("lekiwi")` (default) spawns at `[0, 0, 0]` and emits
    `"'lekiwi' starts 34.6 mm inside the ground, [...] Pass
    position=[0.0, 0.0, 0.0346] to spawn it resting on the ground."` —
    exactly the number already in the registry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(REPO))

from strands_robots import Robot  # noqa: E402


def main() -> int:
    # ---------- Step 1: the dead-letter field in the registry ----------
    with (REPO / "strands_robots" / "registry" / "robots.json").open() as f:
        registry = json.load(f)
    with_spawn = sorted(
        name for name, spec in registry["robots"].items() if "spawn_position" in spec
    )
    print(
        f"Registry declares `spawn_position` on {len(with_spawn)} robots: "
        f"{with_spawn}"
    )

    # ---------- Step 2: zero readers in the Python source ----------
    res = subprocess.run(
        ["grep", "-rn", "spawn_position", "strands_robots/"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    source_hits = [
        line
        for line in res.stdout.strip().split("\n")
        if line
        and not line.endswith(".pyc")
        and "/registry/robots.json" not in line
        and ".bak" not in line
    ]
    print(f"Readers in strands_robots/*.py: {len(source_hits)}")
    assert source_hits == [], f"Unexpected readers found: {source_hits}"

    # ---------- Step 3: default call hits the burial warning ----------
    print()
    print("--- Default Robot('lekiwi') (no explicit position=) ---")
    r = Robot("lekiwi")
    obs = r.get_observation()
    default_z = obs.get("base_pos", [None, None, None])[2]
    print(f"  base_pos.z = {default_z}")
    r.cleanup()

    # ---------- Step 4: user retypes the registry value by hand ----------
    print()
    print("--- Docs quickstart Robot('lekiwi', position=[0,0,0.0346]) ---")
    registry_spawn = registry["robots"]["lekiwi"]["spawn_position"]
    r = Robot("lekiwi", position=registry_spawn)
    obs = r.get_observation()
    fixed_z = obs.get("base_pos", [None, None, None])[2]
    print(f"  registry spawn_position = {registry_spawn}")
    print(f"  base_pos.z = {fixed_z}  (matches registry)")
    r.cleanup()

    print()
    print("=== Assertions ===")
    assert default_z == 0.0, f"Default z expected 0.0 (dead-letter), got {default_z}"
    assert abs(fixed_z - registry_spawn[2]) < 1e-6, "Explicit position should match"
    print("  PASS: default spawn ignores registry.spawn_position")
    print(
        "  PASS: explicit position= with the registry value spawns correctly, "
        "proving the field is a textual duplicate the user is asked to copy."
    )
    print()
    print(
        "Mechanism: `spawn_position` is a registry field read ONLY by "
        "`docs/hooks/robot_pages.py` to compose the quickstart example. "
        "`Robot()` / `add_robot()` never consult it, so the default call "
        "hits `_spawn_burial_warning` and asks the user to retype the "
        "same number the registry already carries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
