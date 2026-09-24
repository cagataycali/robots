#!/usr/bin/env python3
"""Build each procedural terrain, drop a quadruped on it, and confirm the ground.

Goal: Show that ``create_world(terrain=...)`` lays down a deterministic
heightfield ground - rough bumps, stairs, a pyramid, or a slope - instead of
the flat plane, so a locomotion scene tests robustness to ground the robot can
trip on. The terrain curriculum knob is the ``difficulty`` scalar: it scales the
peak elevation without changing the terrain kind, so the same world gets harder
across resets.

For each terrain the example inspects the generated heightfield (its distinct
signature is the proof each kind is different: ``stairs`` is five discrete
plateaus, ``slope`` is a monotonic ramp, ``rough`` is continuous noise,
``pyramid`` is symmetric concentric plateaus), then rebuilds the world, spawns
the robot, steps physics until the robot has actually stopped moving, and
saves a rendered frame. Note the default ~8 cm terrain is deliberately subtle
in the image - raise ``--difficulty`` to make the relief obvious.

Settling is MEASURED, not assumed. A floating base spawns seated with its feet
just clear of the surface, so it drops, and with nothing driving its joints its
legs then fold under gravity - which takes the better part of a second. So this
example steps until the base's own reported speed falls under
``SETTLED_SPEED_MPS``, and refuses rather than capture a frame if it never does.
The shipped Go2 is still falling at ~0.8 m/s after 40 steps (0.08 s) and comes
to rest after roughly 400.

No policy, no checkpoint, no GPU, no Hugging Face credentials: the heightfield
generator is pure Python and MuJoCo steps on CPU. It is the ground-generation
primitive a locomotion curriculum builds on; the shipped locomotion examples
(``examples/locomotion/``) run their WBC policy on the flat default world.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: one line per terrain (kind, peak elevation, distinct
heightfield levels, settle steps used) and one PNG per terrain under the output
dir. Runtime: ~5 seconds on CPU.
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v3 as iio

from strands_robots import Robot
from strands_robots.simulation.terrain import (
    SUPPORTED_TERRAINS,
    generate_heightfield,
    terrain_elevation,
)

# A base reporting less than this is at rest for the purpose of a still frame:
# the Go2 falls at ~0.8 m/s while settling and holds a few mm/s of contact jitter
# once its legs have folded, so the two states are two orders of magnitude apart
# and the threshold does not need tuning per robot.
SETTLED_SPEED_MPS = 0.01

# Steps between speed samples while settling. One sample per 20 steps (0.04 s of
# a 500 Hz sim) stops promptly without reading the observation on every step.
SETTLE_SAMPLE_STEPS = 20


def _base_speed(sim: Robot, robot: str) -> float:
    """Magnitude of ``robot``'s reported base linear velocity, in m/s."""
    return sum(float(v) ** 2 for v in sim.get_observation(robot)["base_lin_vel"]) ** 0.5


def settle(sim: Robot, robot: str, max_steps: int) -> int:
    """Step until ``robot``'s base stops moving, and report the steps it took.

    How long a floating base takes to come to rest is a property of the model,
    not a constant - the Go2's legs fold in ~400 steps, a longer-legged base
    takes thousands - so the settle is measured from the base's own reported
    speed rather than assumed from a step count.

    Args:
        sim: Live simulation whose world already holds ``robot``.
        robot: Name of the spawned robot to settle.
        max_steps: Give up after this many steps rather than looping forever.

    Returns:
        Physics steps stepped before the base came to rest.

    Raises:
        RuntimeError: The base was still moving after ``max_steps``, so no frame
            captured here would show a settled robot.
    """
    stepped = 0
    while stepped < max_steps:
        for _ in range(min(SETTLE_SAMPLE_STEPS, max_steps - stepped)):
            sim.step()
            stepped += 1
        if _base_speed(sim, robot) <= SETTLED_SPEED_MPS:
            return stepped
    raise RuntimeError(
        f"{robot} was still moving at {_base_speed(sim, robot):.3f} m/s after {max_steps} "
        f"settle steps (at rest is <= {SETTLED_SPEED_MPS} m/s); raise --steps"
    )


def _check(result: dict, what: str) -> None:
    """Raise if a sim action returned an error dict - never continue silently."""
    if isinstance(result, dict) and result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"{what} failed: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="go2", help="registered robot to spawn (default: go2 quadruped)")
    parser.add_argument("--difficulty", type=float, default=1.0, help="terrain curriculum scale (1.0 = full height)")
    parser.add_argument(
        "--steps", type=int, default=3000, help="give up settling the robot after this many physics steps"
    )
    parser.add_argument("--out", default="/tmp/strands_terrain", help="directory for the per-terrain PNG frames")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sim by default - no hardware. mesh=False keeps this a single local process.
    sim = Robot(args.robot, mesh=False)

    for kind in SUPPORTED_TERRAINS:
        # The heightfield is pure data, generated deterministically before any
        # physics: its distinct-level count is the signature that each kind is a
        # different ground (stairs -> a few discrete plateaus, slope -> a
        # monotonic ramp, rough -> continuous noise, pyramid -> concentric rings).
        heightfield = generate_heightfield(kind)
        distinct_levels = len({round(h, 4) for h in heightfield})

        # Tear down the current world (the factory built a flat one, and
        # create_world refuses to overwrite a live world), then lay this
        # heightfield ground. Each step returns a status dict - check it rather
        # than assume success. difficulty scales the terrain's peak elevation.
        _check(sim.destroy(), f"destroy before {kind}")
        _check(sim.create_world(terrain=kind, difficulty=args.difficulty), f"create_world(terrain={kind!r})")
        _check(sim.add_robot(args.robot), f"add_robot({args.robot!r}) on {kind}")
        # Elevated 3/4 camera: terrain relief reads best from above and to the side.
        sim.add_camera(name="view", position=[4.0, -4.0, 3.0], target=[0.0, 0.0, 0.2])

        # Settle the robot so the captured frame really shows it at rest on the
        # heightfield, and report how long that took on this model.
        settle_steps = settle(sim, args.robot, args.steps)

        frame = sim.get_observation(args.robot)["view"]
        path = out_dir / f"terrain_{kind}.png"
        iio.imwrite(path, frame)
        print(
            f"{kind:8} peak={terrain_elevation(args.difficulty):.3f} m  "
            f"levels={distinct_levels:>4}  settled in {settle_steps:>4} steps  ->  {path}"
        )

    print(f"Saved {len(SUPPORTED_TERRAINS)} terrain frames under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
