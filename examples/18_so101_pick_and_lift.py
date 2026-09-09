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

Every primitive here returns a tool envelope, and this example checks each one:
a refused step is reported with the refusal's own text rather than carried past.
Without that check the ``[sim-mujoco]`` IK solver being absent - which refuses
all three ``move_to`` calls, each naming the install that fixes it - was
indistinguishable from the friction limitation described above, because the run
still summarised ``status="success"`` and printed "PICK FAILED - cube lifted
only 0.0 mm".

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


class _Refused(RuntimeError):
    """A simulation primitive answered with an error envelope.

    Raised by :func:`_ok` and caught once in :func:`run_pick`, which renders it
    as the error summary the caller receives. The step name is carried because
    that summary is the only thing a caller sees.
    """

    def __init__(self, step: str, result: dict) -> None:
        self.step = step
        self.detail = "; ".join(
            item["text"] for item in result.get("content", []) if isinstance(item, dict) and item.get("text")
        )
        super().__init__(f"{step}: {self.detail}")


def _ok(step: str, result: dict) -> dict:
    """Return *result*, or raise :class:`_Refused` when it is an error envelope.

    Args:
        step: What was attempted, named as it should read in the summary.
        result: The envelope the primitive returned.

    Returns:
        *result* unchanged, so a call site reads as the primitive it wraps.

    Raises:
        _Refused: When ``result["status"]`` is ``"error"``.
    """
    if result.get("status") == "error":
        raise _Refused(step, result)
    return result


def _cube_z(sim) -> float:
    """Current world z of the cube's body origin (metres)."""
    return sim.get_body_state("cube")["content"][-1]["json"]["position"][2]


def run_pick(video_path: str | None = None) -> dict:
    """Scripted SO-101 pick that lifts the cube, using only the public API.

    Args:
        video_path: When set, render the rollout from a side camera and write an
            MP4 there (needs a GL backend; run headless with ``MUJOCO_GL=egl``).

    Returns:
        A summary dict. On a completed run: ``{"status": "success", "lifted_mm",
        "success", "frames"}``, where ``success`` is True when the cube rises at
        least 80 mm. If any primitive refuses, ``status`` is ``"error"`` and the
        summary also carries ``step`` (which one) and ``detail`` (the refusal's
        own text, which names the remedy) - so a refusal is never reported as a
        completed pick that merely failed to lift.
    """
    # Robot("so101") defaults to mode="sim": the factory has already called
    # create_world() and added the "so101" arm, so the scene is ready.
    sim = Robot("so101", mesh=False)
    rest_on_table = CUBE_HALF / 2  # a 25 mm cube rests with its centre 12.5 mm up

    frames: list = []
    record = video_path is not None

    try:
        _ok(
            "add_object(cube)",
            sim.add_object(
                name="cube",
                shape="box",
                position=[CUBE_XY[0], CUBE_XY[1], rest_on_table],
                size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
                color=[0.9, 0.2, 0.2, 1],
                mass=0.02,
            ),
        )

        if record:
            import imageio.v3 as iio
            import numpy as np

            _ok(
                "add_camera(side)",
                sim.add_camera(
                    name="side",
                    position=[0.32, -0.55, 0.30],
                    target=[CUBE_XY[0], CUBE_XY[1], 0.08],
                    fov=52,
                    width=480,
                    height=368,
                ),
            )

            def snap() -> None:
                rendered = _ok("render(side)", sim.render(camera_name="side"))
                png = rendered["content"][1]["image"]["source"]["bytes"]
                frames.append(np.asarray(iio.imread(png, extension=".png")))

        else:

            def snap() -> None:
                return None

        # Let the cube settle onto the table before we read its rest height.
        _ok("step(settle)", sim.step(200))
        snap()
        z_rest = _cube_z(sim)

        # --- pick sequence, public API only ---------------------------------
        _ok("set_gripper(open)", sim.set_gripper(robot_name="so101", state="open"))
        snap()
        # 1. approach: hover 10 cm above the cube
        _ok(
            "move_to(hover)",
            sim.move_to(robot_name="so101", position=[CUBE_XY[0], CUBE_XY[1], z_rest + 0.10], tol=0.02),
        )
        snap()
        # 2. descend to the cube (tol 2 cm: a 5-DOF arm cannot also pin orientation)
        _ok(
            "move_to(descend)",
            sim.move_to(robot_name="so101", position=[CUBE_XY[0], CUBE_XY[1], z_rest + 0.005], tol=0.02),
        )
        snap()
        # 3. close the fingers on the cube
        _ok("set_gripper(close)", sim.set_gripper(robot_name="so101", state="close"))
        snap()
        # 4. grasp-assist: hold the cube's current pose relative to the gripper.
        #    NOT a physical grasp - see the module docstring.
        _ok("attach_bodies(weld)", sim.attach_bodies(parent="so101/gripper", child="cube", mode="weld"))
        snap()
        # 5. lift
        _ok(
            "move_to(lift)",
            sim.move_to(robot_name="so101", position=[CUBE_XY[0], CUBE_XY[1], z_rest + LIFT_HEIGHT], tol=0.02),
        )
        snap()
        z_top = _cube_z(sim)
        lifted_mm = 1000.0 * (z_top - z_rest)

        # 6. place back down and release, so the scene ends clean
        _ok(
            "move_to(place)",
            sim.move_to(robot_name="so101", position=[CUBE_XY[0] + 0.06, CUBE_XY[1], z_rest + 0.02], tol=0.02),
        )
        snap()
        _ok("detach_bodies", sim.detach_bodies(parent="so101/gripper", child="cube"))
        _ok("set_gripper(release)", sim.set_gripper(robot_name="so101", state="open"))
        _ok("step(settle)", sim.step(100))
        snap()
    except _Refused as refused:
        # The refusal carries its own remedy; a summary that dropped it would
        # leave "lifted 0.0 mm" as the only evidence, which is the friction
        # limitation this example exists to work around - a different cause
        # with a different fix.
        return {
            "status": "error",
            "step": refused.step,
            "detail": refused.detail,
            "lifted_mm": 0.0,
            "success": False,
            "frames": len(frames),
        }

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
    if result["status"] == "error":
        print(f"PICK REFUSED at {result['step']}: {result['detail']}")
        raise SystemExit(1)
    if result["success"]:
        print(f"PICK OK - cube lifted {result['lifted_mm']} mm")
    else:
        print(f"PICK FAILED - cube lifted only {result['lifted_mm']} mm")
    if args.video:
        print(f"wrote {result['frames']} frames -> {args.video}")


if __name__ == "__main__":
    main()
