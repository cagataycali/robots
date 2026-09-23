#!/usr/bin/env python3
"""Checkpoint the sim, apply a disturbance, measure, then restore exactly.

Goal: Show the reproducible-experiment toolkit - the pattern for perturbation
and robustness testing where you need to return to an exact starting state:

  - ``save_state`` / ``load_state`` - snapshot the full sim state (qpos/qvel/time)
    under a name and restore it byte-for-byte later.
  - ``apply_force``                 - latch an external force on a body (a push,
                                       a disturbance). MuJoCo re-applies it on
                                       every step until the next ``apply_force``
                                       on that body or a ``reset()``.
  - ``raycast``                     - cast a ray and get the first geom hit +
                                       distance (a range/contact sensor).

The example: checkpoint the scene, latch an upward force on a cube, hold it for
``--steps`` steps and measure how far it moved, then load_state to prove the
restore returns the cube to its exact original height (and clears the latch with
it). Finally raycast down onto the cube to read its surface distance.

Runs on CPU, no GPU/checkpoint/hardware.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: the cube's height after the held push (2 N for 20 steps lifts a
50 g cube ~23 mm; a one-step impulse of the same force would leave it ~7 mm
lower, having fallen), after the restore (equal to the original), and the raycast
hit. Runtime: ~3 seconds on CPU.
"""

from __future__ import annotations

import argparse

from strands_robots import Robot


def _dispatch(sim, action: str, params: dict) -> dict:
    """Dispatch a sim action and fail loud on error."""
    result = sim._dispatch_action(action, params)
    if isinstance(result, dict) and result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"{action} failed: {msg}")
    return result


def _json(result: dict) -> dict:
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def _cube_z(sim) -> float:
    """Current z-height of the cube's body origin."""
    state = _json(_dispatch(sim, "get_body_state", {"body_name": "cube"}))
    return float(state["position"][2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", type=float, default=2.0, help="upward force (N) to latch on the cube")
    parser.add_argument("--steps", type=int, default=20, help="steps to hold the latched force for")
    args = parser.parse_args()

    sim = Robot("so100", mesh=False)
    sim.add_object(
        name="cube",
        shape="box",
        size=[0.03, 0.03, 0.03],
        position=[0.2, 0.0, 0.05],
        color=[1, 0, 0, 1],
        mass=0.05,
    )
    for _ in range(3):
        sim.step()

    z_start = _cube_z(sim)
    print(f"cube z at start          : {z_start:.4f} m")

    # Checkpoint the exact state we want to return to.
    _dispatch(sim, "save_state", {"name": "before_push"})

    # Latch an upward force on the cube. It is re-applied on every step below,
    # so the cube accelerates for the whole window - not once.
    _dispatch(sim, "apply_force", {"body_name": "cube", "force": [0.0, 0.0, args.force]})
    for _ in range(args.steps):
        sim.step()
    z_pushed = _cube_z(sim)
    print(
        f"cube z after {args.force} N held  : {z_pushed:.4f} m  "
        f"(moved {z_pushed - z_start:+.4f} over {args.steps} steps)"
    )

    # Restore the checkpoint: the cube returns to its exact starting height, and
    # the saved state carries the latched wrench, so the push stops with it.
    _dispatch(sim, "load_state", {"name": "before_push"})
    z_restored = _cube_z(sim)
    print(f"cube z after load_state  : {z_restored:.4f} m  (delta from start {z_restored - z_start:+.6f})")

    # Raycast straight down onto the cube to read its surface distance.
    ray = _json(_dispatch(sim, "raycast", {"origin": [0.2, 0.0, 0.5], "direction": [0.0, 0.0, -1.0]}))
    if ray.get("hit"):
        print(f"raycast down             : hit {ray['geom_name']} at {ray['distance']:.4f} m")
    else:
        print("raycast down             : no hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
