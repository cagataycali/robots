"""
Repro: SimEngine.send_action violates the "never partially applied" invariant.

docs/start/first-robot.md:71 promises:
    "A key the robot does not have is not silently dropped: it returns
     status=error naming the valid keys and labels."

The docs are silent on what happens to the VALID keys in a mixed batch.
Sibling command surfaces (ROS bridge, RTPS bridge) treat refusal as whole-batch
rejection — see tests/test_inbound_command_refuses_a_non_finite_position.py:48
and tests/test_hardware_ros_bridge.py:508 ("never partially applied").

SimEngine.send_action is the outlier: it writes the valid keys, advances
physics n_substeps times, and THEN returns status="error" naming the invalid
keys. The code in strands_robots/simulation/mujoco/simulation.py:1304-1312
even documents the design intent:

    # Refused before a single ctrl value is written, because a refusal that
    # arrived after the write would leave the robot commanded and the world
    # un-advanced - the one state this surface must never report an error from.

That block guards n_substeps but the invariant is violated by the unresolved-keys
path 50 lines below. Caller reading status="error" has no way to know the
world has moved. Retrying on error double-strokes the valid keys.

Same pattern repeats on newton/simulation.py:1353 and isaac/simulation.py:6680.
"""

from strands_robots import Robot


def main() -> None:
    robot = Robot("so101")
    try:
        baseline = robot.get_robot_state()["content"][1]["json"]["state"]["1"]["position"]
        print(f"baseline joint 1 pos: {baseline:.4f}")

        # Mixed valid + invalid in one call
        result = robot.send_action({"1": 0.5, "wrist_pan": 0.1}, n_substeps=200)

        print(f"status={result['status']}")
        print(f"applied={result['content'][1]['json']['applied']}")

        after = robot.get_robot_state()["content"][1]["json"]["state"]["1"]["position"]
        print(f"joint 1 after: {after:.4f}")

        moved = abs(after - baseline) > 1e-3
        print()
        print(f"status is 'error' but joint 1 moved: {moved}")
        print()
        if moved and result["status"] == "error":
            print("DEFECT CONFIRMED:")
            print("  - status reports error (as docs promise)")
            print("  - world has advanced and ctrl was written for valid keys")
            print("  - user retrying on error will double-write")
            print("  - sibling surfaces (ROS/RTPS bridges) reject whole-batch")
    finally:
        robot.cleanup()


if __name__ == "__main__":
    main()
