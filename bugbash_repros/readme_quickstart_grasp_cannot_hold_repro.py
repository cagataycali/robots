"""
README quickstart (`pick up the red cube`) fails on physics alone — the so100
jaw's `forcerange=-3.5..3.5 N·m` + default friction + default `0.1 kg` cube
produce a grasp that is reported `Closed on 'red_cube' (N contacts)` by
`set_gripper("close")` but drops the cube on the first `move_to` lift. The
simulator correctly diagnoses it (`'red_cube' was in the fingers when the move
started and was left behind … the grasp does not hold it`) and names the
`attach_bodies(parent='so100/Fixed_Jaw', child='red_cube', mode='weld')`
grasp-assist as the remedy — but the README that promised the hero demo does
not: a new user sees "pick up the red cube" and gets a cube on the ground.

Reproduces on main @ 53488bab9 — fresh clone of strands-labs/robots, fresh
`uv venv --python 3.12 && uv pip install "strands-robots[sim-mujoco]"`,
then this script. Verified both with the README's `size=[0.05]*3` cube and
with the smaller `size=[0.025]*3` cube from docs/start/first-robot.md — same
"left behind" verdict in both cases.

Run:
    MUJOCO_GL=egl python bugbash_repros/readme_quickstart_grasp_cannot_hold_repro.py
"""
from __future__ import annotations

import sys

from strands_robots import Robot


def _run_pickup_sequence(cube_size: list[float], label: str) -> dict:
    """Execute the README scripted pick-and-place; return cube/lift report."""
    r = Robot("so100")
    r.add_object(
        name="red_cube",
        shape="box",
        size=cube_size,
        position=[0.0, -0.2, 0.025],
        color=[1.0, 0.0, 0.0],
    )
    r.add_camera(
        name="front",
        position=[0.3, -0.7, 0.45],
        target=[0.0, -0.2, 0.03],
    )

    cube_z_before = r.get_body_state(body_name="red_cube")["content"][0]["text"]
    r.set_gripper(state="open")
    r.move_to(position=[0.0, -0.2, 0.08])          # approach above
    r.move_to(position=[0.0, -0.2, 0.025])         # lower to cube
    close = r.set_gripper(state="close")           # grip
    lift = r.move_to(position=[0.0, -0.2, 0.25])   # lift
    cube_after = r.get_body_state(body_name="red_cube")["content"][0]["text"]
    r.destroy()

    return {
        "label": label,
        "cube_size": cube_size,
        "close_text": close["content"][0]["text"],
        "lift_text": lift["content"][0]["text"],
        "cube_z_before": cube_z_before,
        "cube_after": cube_after,
    }


def main() -> int:
    print("=" * 72)
    print("Scenario A — README verbatim: size=[0.05, 0.05, 0.05]")
    print("=" * 72)
    a = _run_pickup_sequence([0.05, 0.05, 0.05], "README (5 cm cube)")
    print(f"  close: {a['close_text']}")
    print(f"  lift : {a['lift_text']}")
    # Extract post-lift z of cube from body-state text
    assert "left behind" in a["lift_text"], (
        "Expected 'left behind' verdict in lift message — did the physics "
        "or the message change? Lift text was:\n  " + a["lift_text"]
    )

    print()
    print("=" * 72)
    print("Scenario B — docs/start/first-robot.md cube: size=[0.025]*3")
    print("=" * 72)
    b = _run_pickup_sequence([0.025, 0.025, 0.025], "docs (2.5 cm cube)")
    print(f"  close: {b['close_text']}")
    print(f"  lift : {b['lift_text']}")
    assert "left behind" in b["lift_text"], (
        "Expected 'left behind' verdict for the smaller cube too — this repro "
        "pins the papercut on BOTH cube sizes the project ships in its public "
        "quickstart demos."
    )

    print()
    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    print(
        "`Agent(tools=[robot])('pick up the red cube')` as advertised in README.md:57\n"
        "cannot succeed through physics alone with the shipped so100 defaults:\n"
        "  - 5 cm cube : lift reports 'left behind, the grasp does not hold it'\n"
        "  - 2.5 cm cube: same verdict (4 contacts, still slides out)\n"
        "The simulator's remedy sentence names `attach_bodies(...mode='weld')`\n"
        "— a grasp-assist weld — but the README doesn't mention it, so the LLM\n"
        "has to FAIL FIRST, read the error, then weld. First-run UX is a cube on\n"
        "the ground."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
