"""Repro for README L101 'ROS 2 ... expose a running sim' claim.

The ROS 2 row of the README's 'What you get' table (README.md:101) names three
capabilities on a single row, implying each works over the same pip-only path:

    | **ROS 2** - observe and command any graph (`use_ros`),
                  act as a node without rclpy (`use_rtps`),
                  expose a running sim                     | [ROS 2] |

docs/learn/ros2.md:38-43 shows that a *real* arm reaches the ROS 2 graph without
rclpy by selecting ``ros2_transport="rtps"``:

    Robot("so101", mode="real", ..., ros2_bridge=True, ros2_transport="rtps")

The ``[ros2]`` extra installs the (pip-only) ``cyclonedds`` wheel; no sourced
distro, no system ROS 2 install. ``HardwareRobot`` passes ``ros2_transport`` to
``_check_ros2_bridge_deps`` (strands_robots/hardware_robot.py:775), which uses
``HardwareRtpsBridge`` (strands_robots/hardware_rtps_bridge.py:107) - the
rclpy-free sibling of ``HardwareRosBridge``.

But the sim side has **no** ``ros2_transport`` kwarg at all:

    * SimEngine (strands_robots/simulation/base.py:1170-1234) only takes
      ``ros2_bridge: bool`` and hard-wires ``SimRosBridge`` (which imports
      rclpy).
    * No ``SimRtpsBridge`` class exists.
    * MuJoCo's constructor (strands_robots/simulation/mujoco/simulation.py:913,
      :1071) forwards only ``ros2_bridge`` and ``ros2_domain`` into
      ``_init_ros_bridge``.

So of the three ROS 2 capabilities the README row advertises:

    use_ros     - rclpy-free? NO (needs sourced distro) - documented
    use_rtps    - rclpy-free? YES (pip install [ros2])  - documented
    expose sim  - rclpy-free? NO  (ros2_bridge requires rclpy, no rtps path)

The sibling capability for hardware (``mode="real"`` with ``ros2_transport="rtps"``)
exists and is documented at docs/learn/ros2.md:38-54. The README row's parallel
phrasing - "act as a node without rclpy (``use_rtps``), expose a running sim" -
places the sim claim one comma over from the "without rclpy" promise, with no
scoping or extra needed.  A user on macOS/CI/a Jetson without a sourced
ROS 2 distro follows the row and hits an rclpy ImportError on the sim path.

Run::

    python readme_ros2_expose_sim_rclpy_only_repro.py

in an environment with::

    pip install 'strands-robots[sim-mujoco,ros2]'   # README-friendly: no rclpy

Expected (per README L101 parallel + hardware precedent): same pip-only path
publishes sim joint_states / image_raw over DDS/RTPS the way the real arm does.
Actual: ``ImportError: 'rclpy' is required for the ROS 2 telemetry bridge``.
"""

from __future__ import annotations

import sys


def main() -> int:
    # README-faithful call. The user has `pip install 'strands-robots[sim-mujoco,ros2]'`.
    # ``[ros2]`` installs cyclonedds (which `use_rtps` and the hardware RTPS
    # bridge speak); ``[sim-mujoco]`` installs the sim backend. No sourced ROS 2.
    from strands_robots import Robot

    # Mirror the hardware pattern from docs/learn/ros2.md:43 - just with no
    # ``mode="real"``, since the README promises the sim path too.
    try:
        robot = Robot(
            "so100",
            ros2_bridge=True,
            # ros2_transport="rtps",  # ← has no effect on sim; kwarg does not exist
        )
    except ImportError as e:
        print("STATUS: fail")
        print(f"RAISED: ImportError: {e}")
        print()
        print("Hardware siblings (strands_robots/hardware_robot.py:559-575) accept")
        print("``ros2_transport='rtps'`` and route to HardwareRtpsBridge, which uses")
        print("only the pip-installable cyclonedds; the sim constructor has no such")
        print("knob (strands_robots/simulation/base.py:1170-1234 only takes")
        print("``ros2_bridge: bool`` and ``ros2_domain: int``).")
        return 1

    # If we somehow got here, show the type.
    print(f"STATUS: constructed {type(robot).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
