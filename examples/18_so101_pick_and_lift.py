#!/usr/bin/env python3
"""Get an SO-101 from a cube on the table to a cube in the air - no hardware, no RL.

Goal: a minimal, always-works *reference pick* for the SO-101 in MuJoCo, using
only the public simulation surface. It reaches the cube with :meth:`move_to`,
closes the fingers with :meth:`set_gripper`, holds the cube with the supported
grasp-assist primitive :meth:`attach_bodies` (``mode="weld"``), lifts it, and
places it down again with :meth:`detach_bodies`.

Why grasp-assist and not a friction pinch
-----------------------------------------
A two-finger *friction* grasp does not hold the cube on the shipped ``so101``
model: the gripper reaches and closes on the cube, but it is not lifted. This
reproduces across a scripted pinch (0 mm lift with the fingers in contact), the
``so101_curobo`` example's scripted physical-grasp path (``success=False``,
displacement 6 mm), and an RL study over 1.5M steps (0% success). The cause is
the gripper's convex-hull collision geometry - the advertised tool site
``so101/gripper`` sits on the static jaw's hull boundary and the free channel
between the pads is offset from it - so no reward tuning of the pinch makes the
cube lift. See strands-labs/robots#2167 and #2145.

The kinematics are not the blocker: 5-DOF IK reaches the grasp pose to well
under a millimetre. What is missing is a *reference* that composes the public
primitives into a lift. That is what this file is.

``attach_bodies(mode="weld")`` adds an equality constraint holding the cube's
current pose relative to the gripper - the library's supported grasp-assist.
It is NOT a physical grasp: no contact force holds the cube. A dataset recorded
during a welded carry contains an idealized transport segment; label or gate
such episodes accordingly (the same caveat the ``attach_bodies`` docstring
carries).

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: "PICK OK - cube lifted NNN mm" (NNN is ~150).
Runtime: ~3 seconds on CPU (a few more if --video is passed).

Usage:
    python examples/18_so101_pick_and_lift.py
    python examples/18_so101_pick_and_lift.py --video /tmp/so101_pick.mp4
"""

from __future__ import annotations

import argparse

from strands_robots import Robot

# The cube starts here on the table (metres, in the robot's base frame). Well
# within the SO-101's reach; see examples/15_robot_catalog.py for the workspace.
CUBE_XY = (0.02, -0.34)
CUBE_HALF = 0.025  # 25 mm cube edge (full extent passed to add_object)
LIFT_HEIGHT = 0.18  # how far above the table to raise the cube (metres)


def _cube_z(sim) -> float:
    """Current world z of the cube's body origin (metres)."""
    return sim.get_body_state("cube")["content"][-1]["json"]["position"][2]


def run_pick(video_path: str | None = None) -> dict:
    """Scripted SO-101 pick that lifts the cube, using only the public API.

    Args:
        video_path: When set, render the rollout from a side camera and write an
            MP4 there (needs a GL backend; run headless with ``MUJOCO_GL=egl``).

    Returns:
        A summary dict: ``{"status", "lifted_mm", "success", "frames"}``.
        ``success`` is True when the cube rises at least 80 mm.
    """
    # Robot("so101") defaults to mode="sim": the factory has already called
    # create_world() and added the "so101" arm, so the scene is ready.
    sim = Robot("so101", mesh=False)
    rest_on_table = CUBE_HALF / 2  # a 25 mm cube rests with its centre 12.5 mm up
    sim.add_object(
        name="cube",
        shape="box",
        position=[CUBE_XY[0], CUBE_XY[1], rest_on_table],
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        color=[0.9, 0.2, 0.2, 1],
        mass=0.02,
    )

    frames: list = []
    record = video_path is not None
    if record:
        import imageio.v3 as iio
        import numpy as np

        sim.add_camera(
            name="side",
            position=[0.32, -0.55, 0.30],
            target=[CUBE_XY[0], CUBE_XY[1], 0.08],
            fov=52,
            width=480,
            height=368,
        )

        def snap():
            png = sim.render(camera_name="side")["content"][1]["image"]["source"]["bytes"]
            frames.append(np.asarray(iio.imread(png, extension=".png")))
    else:

        def snap():
            return None

    # Let the cube settle onto the table before we read its rest height.
    sim.step(200)
    snap()
    z_rest = _cube_z(sim)

    # --- pick sequence, public API only -------------------------------------
    sim.set_gripper(robot_name="so101", state="open")
    snap()
    # 1. approach: hover 10 cm above the cube
    sim.move_to(robot_name="so101", position=[CUBE_XY[0], CUBE_XY[1], z_rest + 0.10], tol=0.02)
    snap()
    # 2. descend to the cube (tol 2 cm: a 5-DOF arm cannot also pin orientation)
    sim.move_to(robot_name="so101", position=[CUBE_XY[0], CUBE_XY[1], z_rest + 0.005], tol=0.02)
    snap()
    # 3. close the fingers on the cube
    sim.set_gripper(robot_name="so101", state="close")
    snap()
    # 4. grasp-assist: hold the cube's current pose relative to the gripper.
    #    NOT a physical grasp - see the module docstring.
    sim.attach_bodies(parent="so101/gripper", child="cube", mode="weld")
    snap()
    # 5. lift
    sim.move_to(robot_name="so101", position=[CUBE_XY[0], CUBE_XY[1], z_rest + LIFT_HEIGHT], tol=0.02)
    snap()
    z_top = _cube_z(sim)
    lifted_mm = 1000.0 * (z_top - z_rest)

    # 6. place back down and release, so the scene ends clean
    sim.move_to(robot_name="so101", position=[CUBE_XY[0] + 0.06, CUBE_XY[1], z_rest + 0.02], tol=0.02)
    snap()
    sim.detach_bodies(parent="so101/gripper", child="cube")
    sim.set_gripper(robot_name="so101", state="open")
    sim.step(100)
    snap()

    if record and frames:
        import imageio.v3 as iio
        import numpy as np

        iio.imwrite(video_path, np.stack(frames), fps=6, codec="libx264")

    return {
        "status": "success",
        "lifted_mm": round(lifted_mm, 1),
        "success": lifted_mm >= 80.0,
        "frames": len(frames),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=None, help="write an MP4 of the rollout here")
    args = ap.parse_args()

    result = run_pick(video_path=args.video)
    if result["success"]:
        print(f"PICK OK - cube lifted {result['lifted_mm']} mm")
    else:
        print(f"PICK FAILED - cube lifted only {result['lifted_mm']} mm")
    if args.video:
        print(f"wrote {result['frames']} frames -> {args.video}")


if __name__ == "__main__":
    main()
