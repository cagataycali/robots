"""
Defect: README quickstart `Robot("so100")` grasp sequence silently fails because
`move_to(position=cube_center)` physically sweeps the cube out of range before
the gripper ever closes. `move_to` returns `reached=True` (EE did arrive at the
requested XYZ), then `set_gripper(state="close")` reports **"Closed on nothing"**
and advises *"move_to the object first"* — advice the user already followed.

Difference from harness#818: #818 is the LIFT phase (close succeeds, lift
leaves the cube behind — fixed with weld). This defect is the APPROACH phase
(close never catches the cube at all). On so100 the cube is pushed hard enough
that even one `set_gripper(state="close")` has zero fingertip contacts and the
set_gripper advice sentence ("move_to the object first (get_body_state gives
its position)") is a false lead: the user did call move_to, with the right
position.

Rotation target: so101_sim_quickstart (same hero snippet).
Upstream cite:
  - README.md:50-57 — hero snippet calling move_to+set_gripper directly through
    the Agent.
  - strands_robots/simulation/mujoco/motion_primitives.py:470 — move_to is a
    straight servo-descent, not collision-aware (per its own docstring,
    "NOT collision-aware: the straight servo descent can sweep through
    obstacles"); the primitive reports reached=True on EE arrival regardless.
  - strands_robots/simulation/mujoco/physics.py — set_gripper's "Closed on
    nothing" refusal sentence names `move_to` as the fix without a path that
    detects "the body you targeted was displaced during the previous move_to".

Repro: builds the README literal, inspects cube before/after move_to, reports
displacement, then closes and prints the misleading advice.

Run:
    MUJOCO_GL=egl python bugbash_repros/move_to_sweeps_cube_before_grasp_repro.py
"""

from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def _get_cube_pos(r: object) -> list[float]:
    state = r.get_body_state(body_name="red_cube")
    text = state["content"][0]["text"]
    m = re.search(r"pos: \[([^\]]+)\]", text)
    if not m:
        raise RuntimeError(f"no pos in body-state: {text[:200]}")
    return [float(x) for x in m.group(1).split(",")]


def main() -> int:
    print("=" * 70)
    print("README quickstart literal — Robot('so100') + add_object cube + grasp")
    print("=" * 70)
    r = Robot("so100")
    r.add_object(
        name="red_cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.0, -0.2, 0.025],
        color=[1.0, 0.0, 0.0],
    )
    r.add_camera(
        name="front",
        position=[0.3, -0.7, 0.45],
        target=[0.0, -0.2, 0.03],
    )

    # Open gripper — the scripted equivalent of what any sensible agent emits
    # before approaching the cube.
    r.set_gripper(state="open")

    cube_before = _get_cube_pos(r)
    print(f"\nCube BEFORE move_to: {cube_before}  (target: [0.0, -0.2, 0.025])")

    # Approach the cube center (the README's spawn position).
    res_move = r.move_to(position=[0.0, -0.2, 0.025])
    rj = res_move["content"][1]["json"]
    print("\n--- move_to result ---")
    print(f"  reached            : {rj['reached']}")
    print(f"  position_error_m   : {rj['position_error_m']:.4f}")
    print(f"  ee_position        : {rj['ee_position']}")

    cube_after = _get_cube_pos(r)
    disp_m = sum((a - b) ** 2 for a, b in zip(cube_before, cube_after)) ** 0.5
    print("\n--- cube displaced during move_to ---")
    print(f"  Cube AFTER move_to : {cube_after}")
    print(f"  displacement       : {disp_m * 100:.1f} cm  (shoved by swept arm)")

    body = r.get_body_state(body_name="red_cube")
    txt = body["content"][0]["text"]
    for key in ("linvel", "angvel"):
        m = re.search(rf"{key}: \[([^\]]+)\]", txt)
        if m:
            print(f"  {key:9s}        : [{m.group(1)}]")

    # Close the gripper — this is where the physics tells us the truth.
    res_close = r.set_gripper(state="close")
    close_text = res_close["content"][0]["text"]
    print("\n--- set_gripper(state='close') ---")
    print(close_text)

    # Assertions: the defect fingerprint.
    assert rj["reached"] is True, "regression: move_to no longer claims reached=True"
    assert disp_m > 0.01, (
        "regression: cube no longer displaced by approach "
        f"(disp={disp_m * 100:.2f} cm, expected > 1 cm)"
    )
    assert "Closed on nothing" in close_text, (
        "regression: close no longer reports 'Closed on nothing' — the papercut "
        "may be fixed (upgrade this repro)"
    )
    assert "move_to the object first" in close_text, (
        "regression: close's advice sentence changed — update defect wording "
        "if the advice now mentions get_body_state-after-displacement"
    )

    print("\n" + "=" * 70)
    print("CONFIRMED DEFECT:")
    print("  * move_to(position=cube_center) swept the cube ~2.4 cm out of range")
    print("  * move_to.reached is still True (silent-wrong: EE arrived, task failed)")
    print("  * set_gripper's 'Closed on nothing' points user back to move_to —")
    print("    which the user already called with the correct position")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
