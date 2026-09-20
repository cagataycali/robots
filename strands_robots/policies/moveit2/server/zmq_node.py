"""ROS 2 sidecar that exposes ``moveit_py`` planning over ZMQ + msgpack.

This is a **reference implementation** matching the wire protocol the
client (:class:`strands_robots.policies.moveit2.MoveIt2Policy`) expects.
It is intentionally minimal - production deployments should fork this
file and harden it for their own collision world / planner pipeline /
auth posture.

Run it with::

    source /opt/ros/jazzy/setup.bash         # or your distro, with moveit_py
    pip install 'strands-robots[moveit2]'    # pyzmq + msgpack, the only non-ROS deps
    python -m strands_robots.policies.moveit2.server.zmq_node \\
        --port 5556 --planning-group panda_arm

``--moveit-config-package`` / ``--robot-name`` default to the panda config
MoveIt 2 itself ships, so that command plans out of the box; point them at your
own config package and pass its group name.

The sidecar is single-threaded REQ/REP - one in-flight plan request at a
time. That matches the ``MoveItPy.plan()`` API which is itself
single-threaded. Because REQ/REP is lockstep, every request that is
received gets exactly one reply: a request that cannot be served is
answered with a failure response, never by dropping the reply and
exiting. A planning failure is reported in the ``plan`` response
(``success=False`` plus a ``status`` naming the stage); anything else is
reported as ``{"error": ...}``, which the client raises as
``RuntimeError``.

Wire protocol::

    request  = {"endpoint": "plan",
                "data": {"joint_state": list[float] | None,
                         "planning_group": str,
                         "target_pose": [x, y, z, qw, qx, qy, qz] | None,
                         "target_joints": dict[str, float] | None,
                         "world_update": dict | None}}
    response = {"trajectory": list[list[float]],
                "joint_names": list[str],        # on success
                "success": bool,
                "status": str}

The trajectory rows are ``[time_from_start_seconds, q0, q1, ..., qN]`` -
the time column lets the client / runner schedule waypoints precisely.
``joint_names`` names the joint each of ``q0 .. qN`` belongs to, in column
order: the planning group's own vocabulary, read from the trajectory message
rather than guessed from the robot the client drives.

Notes for forks:

* The ``world_update`` payload is intentionally schema-free here. A
  production sidecar should validate it against a known schema (e.g.
  expect ``{"depth_topic": str, "stamp": int, "frame_id": str}``) before
  pushing to the planning scene.
* ``api_token`` validation is left as an exercise - the reference
  implementation accepts any client. Enable it by checking
  ``request.get("api_token")`` against an env-var / file secret - by
  then ``request`` is known to be a map.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from strands_robots.utils import require_optional, require_optionals

# These imports deliberately happen inside ``main`` so this file can be
# imported and statically analysed without ROS 2 sourced. Top-level
# imports of ``rclpy`` / ``moveit_py`` would crash on dev boxes that
# only have the strands-robots client installed.

logger = logging.getLogger("moveit2.zmq_node")

#: Remedy for a sidecar launched in a shell where ROS 2 / MoveIt 2 are not
#: importable. A ``system_install=`` text, not a pip line: ``rclpy``,
#: ``moveit`` (moveit_py) and ``moveit_configs_utils`` are not published on
#: PyPI, and the ``[moveit2]`` extra deliberately carries only the client side
#: (pyzmq + msgpack), so a pip command here would report success and change
#: nothing.
ROS_SIDECAR_INSTALL_HINT = (
    "rclpy and moveit_py are not published on PyPI - they ship with a system ROS 2 + MoveIt 2 "
    "install (apt / RoboStack / conda).\n"
    "Source a distro in the shell that launches the sidecar, e.g.:\n"
    "  source /opt/ros/jazzy/setup.bash   # or your distro\n"
    "and install MoveIt 2's Python bindings for it, e.g.:\n"
    "  sudo apt install ros-jazzy-moveit-py ros-jazzy-moveit-configs-utils\n"
    "The [moveit2] extra installs only the client-side pyzmq + msgpack; it does not provision ROS 2."
)

#: What the refusals are for, so every gate in this module names one thing.
_SIDECAR_PURPOSE = "the MoveIt2 ZMQ sidecar"


class MissingRosModuleError(ImportError):
    """A ROS 2 / MoveIt 2 module the sidecar imports lazily is not importable.

    Raised only where this module gates such an import, which is what tells the
    gate's refusal - the remedy in :data:`ROS_SIDECAR_INSTALL_HINT`, reported as
    one error line and exit status 2 - apart from an ``ImportError`` raised by
    the planner the gate admitted. The latter is a failure to diagnose, not an
    install to perform, and keeps its traceback and exit status 1.

    ``ImportError.name`` cannot draw that line: a binding whose name moved
    (``from moveit.planning import MoveItPy`` against a MoveIt 2 that renamed
    it) raises ``ImportError(name="moveit.planning")`` too - the same value the
    gate for that module carries. Measured: exit status 2 and a single line with
    no traceback for a construction failure, where the remedy is not the answer.

    Subclasses ``ImportError``, so a fork catching ``ImportError`` around the
    seams this module invites it to replace still catches it.
    """


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoveIt2 ZMQ sidecar for strands-robots MoveIt2Policy",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for the ZMQ REP socket. Default 0.0.0.0 inside "
        "the container (the host-level firewall / docker port mapping is "
        "the security boundary).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Bind port. Default 5556 - matches MoveIt2Policy's default.",
    )
    parser.add_argument(
        "--planning-group",
        default="arm",
        help="Default MoveIt2 planning-group name. Per-request overrides win.",
    )
    parser.add_argument(
        "--robot-name",
        default="panda",
        help="Robot name inside the MoveIt config package - it names the "
        "description ``MoveItConfigsBuilder`` loads (``config/<robot-name>"
        ".urdf.xacro``). Default matches --moveit-config-package.",
    )
    parser.add_argument(
        "--moveit-config-package",
        default="moveit_resources_panda_moveit_config",
        help="MoveIt config package for the robot to plan for. Default is the "
        "panda config MoveIt 2 itself ships, so the reference deployment "
        "plans out of the box.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for the sidecar. ROS 2 spinner logs are independent.",
    )
    return parser.parse_args(argv)


def _planning_pipeline_params(moveit_config: dict[str, Any]) -> dict[str, Any]:
    """Add the two parameters ``MoveItPy`` needs and the config builder omits.

    ``MoveItConfigsBuilder.to_moveit_configs().to_dict()`` describes the
    pipelines a launch file *could* start: ``planning_pipelines`` is a flat
    list of names. ``MoveItCpp`` reads the ones it should load from the nested
    ``planning_pipelines.pipeline_names`` parameter instead, and refuses to
    construct without it ("Failed to load planning pipelines from parameter
    server"), so the list is re-keyed here. Only the default pipeline is
    loaded: the config package advertises every pipeline MoveIt 2 knows, and
    loading one whose plugin is not installed is fatal to construction.

    ``plan_request_params`` is what a no-argument ``PlanningComponent.plan()``
    reads, and the builder never writes it. Both are written only when absent,
    so a fork can supply either through its own ``moveit_cpp`` YAML.

    Args:
        moveit_config: The dict from ``to_moveit_configs().to_dict()``.

    Returns:
        The same dict, with the two parameters filled in.
    """
    pipeline = moveit_config.get("default_planning_pipeline") or "ompl"
    if not isinstance(moveit_config.get("planning_pipelines"), dict):
        moveit_config["planning_pipelines"] = {
            "pipeline_names": [pipeline],
            "default_planning_pipeline": pipeline,
        }
    moveit_config.setdefault(
        "plan_request_params",
        {
            "planning_attempts": 1,
            "planning_pipeline": pipeline,
            "planning_time": 5.0,
            "max_velocity_scaling_factor": 1.0,
            "max_acceleration_scaling_factor": 1.0,
        },
    )
    return moveit_config


def _build_moveit_py(args: argparse.Namespace) -> Any:
    """Construct the ``moveit_py`` runtime.

    Kept in its own function so a fork can mock / replace the planner
    initialisation without rewriting the ZMQ loop.
    """
    try:
        require_optional("moveit.planning", system_install=ROS_SIDECAR_INSTALL_HINT, purpose=_SIDECAR_PURPOSE)
        require_optional("moveit_configs_utils", system_install=ROS_SIDECAR_INSTALL_HINT, purpose=_SIDECAR_PURPOSE)
    except ImportError as e:
        raise MissingRosModuleError(str(e), name=e.name) from None

    # Outside the gate above on purpose: an ImportError from here is a MoveIt 2
    # whose binding moved, not a MoveIt 2 that is missing, and its traceback is
    # the only thing that says which.
    from moveit.planning import MoveItPy
    from moveit_configs_utils import MoveItConfigsBuilder

    builder = MoveItConfigsBuilder(robot_name=args.robot_name, package_name=args.moveit_config_package)
    moveit_config = _planning_pipeline_params(builder.to_moveit_configs().to_dict())
    moveit_py = MoveItPy(node_name="strands_robots_moveit2_sidecar", config_dict=moveit_config)
    logger.info(
        "MoveItPy initialised for %r from %r; planning groups: %s",
        args.robot_name,
        args.moveit_config_package,
        list(moveit_py.get_robot_model().joint_model_group_names),
    )
    return moveit_py


def _tip_link(moveit_py: Any, planning_group: str) -> str:
    """The link a Cartesian goal for ``planning_group`` is expressed for.

    Read from the robot model rather than hardcoded: the group's end-effector
    parent, falling back to its last link for a group that declares no
    end effector. A name that is not in the model is not refused by
    ``set_goal_state`` - it plans and fails with "Unable to construct goal
    representation", which is why this is derived and not a constant.

    Args:
        moveit_py: The ``MoveItPy`` runtime.
        planning_group: Group the goal is for.

    Returns:
        The link name to pass as ``pose_link``.
    """
    group = moveit_py.get_robot_model().get_joint_model_group(planning_group)
    return group.eef_name or list(group.link_model_names)[-1]


def _start_state(moveit_py: Any, planning_group: str, joint_state: list[float]) -> Any:
    """Build the ``RobotState`` the request's ``joint_state`` describes.

    The client sends the robot's own proprioception, which is the only start
    state available when the robot is not a ROS 2 robot: nothing publishes
    ``/joint_states``, and MoveIt's own current state is then the description's
    default pose - for the panda a self-collision the
    ``CheckStartStateCollision`` adapter aborts every plan on.

    ``joint_state`` is an unnamed vector on the wire, so it is read in order
    onto the group's active joints. A robot that publishes more joints than the
    group plans over (a Panda publishes its two fingers; ``panda_arm`` has
    seven joints) is read from the front and the trailing values are logged as
    ignored rather than dropped silently - a payload in a different order shows
    up in that log and in the plan's own first waypoint. Too few values is
    refused: there is no configuration to plan from.

    Args:
        moveit_py: The ``MoveItPy`` runtime.
        planning_group: Group whose active joints the values are read onto.
        joint_state: Joint positions, in the group's joint order.

    Returns:
        A ``RobotState`` at the described configuration.

    Raises:
        ValueError: If the payload carries fewer values than the group has
            active joints.
    """
    require_optional("moveit.core.robot_state", system_install=ROS_SIDECAR_INSTALL_HINT, purpose=_SIDECAR_PURPOSE)
    from moveit.core.robot_state import RobotState

    model = moveit_py.get_robot_model()
    joints = list(model.get_joint_model_group(planning_group).active_joint_model_names)
    if len(joint_state) < len(joints):
        raise ValueError(
            f"joint_state carries {len(joint_state)} values but planning group {planning_group!r} "
            f"plans over {len(joints)} joints ({', '.join(joints)})"
        )
    if len(joint_state) > len(joints):
        logger.info(
            "joint_state carries %d values for the %d joints of %r; read the first %d, ignored %s",
            len(joint_state),
            len(joints),
            planning_group,
            len(joints),
            joint_state[len(joints) :],
        )
    state = RobotState(model)
    # Assigned as a mapping on purpose: ``set_joint_group_positions`` segfaults
    # on a RobotState that has not been given values yet (moveit_py 2.12).
    state.joint_positions = dict(zip(joints, (float(v) for v in joint_state), strict=False))
    state.update()
    return state


def _plan(
    moveit_py: Any,
    *,
    planning_group: str,
    joint_state: list[float] | None,
    target_pose: list[float] | None,
    target_joints: dict[str, float] | None,
    world_update: dict[str, Any] | None,  # noqa: ARG001 - reserved, see schema TODO above
) -> dict[str, Any]:
    """Run a single ``moveit_py`` plan and serialise the result.

    Returns the wire-format response dict. Every ``moveit_py``
    interaction is guarded, so a failing plan comes back as a structured
    ``{"success": False, "status": ...}`` response instead of raising:
    the REP loop owes its peer exactly one reply, and an escaping
    exception would end the sidecar with that reply never sent.

    ``moveit_py`` is a C++ binding whose exception types are not
    enumerable from Python, so each guard catches broadly and reports the
    stage that failed in ``status``:

    * ``unknown_planning_group`` - the group name does not resolve.
    * ``start_state_error`` - the start state is not usable: a
      ``joint_state`` whose value count does not match the group's active
      joints, or - when the request carries none - a current robot state
      that is not readable (no ``/joint_states`` yet, monitor not warmed up).
    * ``missing_goal`` - neither goal field was supplied.
    * ``invalid_goal`` - the goal was rejected: a joint the group does
      not have, an unresolvable pose link, or a ``target_pose`` that is
      not 7 values.
    * ``planner_exception`` / ``planner_returned_empty`` - planning ran
      and failed. ``planner_returned_empty`` also covers a plan that
      serialised to no waypoint, or to waypoints carrying no joint
      position: a plan that commands nothing is a planning failure, not a
      successful plan. Those two carry a ``:detail`` suffix naming which
      of the two it was.
    * ``trajectory_error`` - the result did not serialise.
    """
    require_optional("geometry_msgs.msg", system_install=ROS_SIDECAR_INSTALL_HINT, purpose=_SIDECAR_PURPOSE)
    from geometry_msgs.msg import PoseStamped

    try:
        component = moveit_py.get_planning_component(planning_group)
    except Exception as e:
        return {"trajectory": [], "success": False, "status": f"unknown_planning_group:{e}"}

    try:
        if joint_state is None:
            component.set_start_state_to_current_state()
        else:
            component.set_start_state(robot_state=_start_state(moveit_py, planning_group, joint_state))
    except Exception as e:  # noqa: BLE001 - report the failing stage structurally
        logger.exception("Reading the start robot state failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"start_state_error:{e}"}

    try:
        if target_joints is not None:
            component.set_goal_state(joint_values=target_joints)
        elif target_pose is not None:
            x, y, z, qw, qx, qy, qz = target_pose
            pose = PoseStamped()
            pose.header.frame_id = moveit_py.get_robot_model().model_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            component.set_goal_state(pose_stamped_msg=pose, pose_link=_tip_link(moveit_py, planning_group))
        else:
            return {
                "trajectory": [],
                "success": False,
                "status": "missing_goal:expected_target_pose_or_target_joints",
            }
    except Exception as e:  # noqa: BLE001 - report the failing stage structurally
        logger.exception("Setting the goal state failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"invalid_goal:{e}"}

    try:
        plan_result = component.plan()
    except Exception as e:  # noqa: BLE001 - report planner failure structurally
        logger.exception("Planning failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"planner_exception:{e}"}

    if not plan_result:
        return {"trajectory": [], "success": False, "status": "planner_returned_empty"}

    # ``trajectory_msg.joint_trajectory.points`` is a list of
    # ``trajectory_msgs/JointTrajectoryPoint``. Each has
    # ``time_from_start`` (Duration) + ``positions`` (list[float]).
    try:
        trajectory_msg = plan_result.trajectory.get_robot_trajectory_msg()
        # The message names the joint each position column belongs to. Sent
        # with the rows so the client keys them by name instead of by the
        # position a column happens to hold in whatever roster it has.
        joint_names = [str(name) for name in trajectory_msg.joint_trajectory.joint_names]
        rows: list[list[float]] = []
        for point in trajectory_msg.joint_trajectory.points:
            t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            rows.append([float(t)] + [float(q) for q in point.positions])
    except Exception as e:  # noqa: BLE001 - report the failing stage structurally
        logger.exception("Serialising the planned trajectory failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"trajectory_error:{e}"}

    # ``not plan_result`` above only sees a falsy plan object. A truthy plan can
    # still serialise to zero waypoints, or to waypoints holding only the time
    # column, and reporting either as success=True hands the client a plan that
    # moves no joint. The kind is unchanged, so a client already matching
    # ``planner_returned_empty`` needs no change to handle these.
    short = [i for i, row in enumerate(rows) if len(row) < 2]
    if not rows or short:
        detail = "no_waypoints" if not rows else f"{len(short)}_of_{len(rows)}_waypoints_carry_no_joint_position"
        logger.warning("The plan serialised to nothing commandable (%s); reporting it as a planning failure.", detail)
        return {"trajectory": [], "success": False, "status": f"planner_returned_empty:{detail}"}

    return {"trajectory": rows, "joint_names": joint_names, "success": True, "status": "ok"}


def main(argv: list[str] | None = None) -> int:
    """ZMQ REP loop entry point. Returns the desired process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Lazy imports - see module docstring for rationale. Each absence is
    # refused with the install that supplies the module, before any socket is
    # bound: an operator who launched the sidecar in an unsourced shell reads
    # the remedy, not a traceback ending in "No module named 'rclpy'".
    try:
        require_optionals(
            ("msgpack", "zmq"),
            extra="moveit2",
            purpose=_SIDECAR_PURPOSE,
            pip_install={"zmq": "pyzmq"},
        )
        require_optional("rclpy", system_install=ROS_SIDECAR_INSTALL_HINT, purpose=_SIDECAR_PURPOSE)
        import msgpack
        import rclpy
        import zmq
    except ImportError as e:
        logger.error("%s", e)
        return 2

    rclpy.init()
    try:
        moveit_py = _build_moveit_py(args)
    except MissingRosModuleError as e:
        # moveit_py / moveit_configs_utils absent: the remedy is the message.
        # Only the gate raises this, so a construction failure that happens to
        # be an ImportError still falls to the branch below with its traceback.
        logger.error("%s", e)
        rclpy.shutdown()
        return 2
    except Exception as e:
        logger.exception("Failed to construct MoveItPy: %s", e)
        rclpy.shutdown()
        return 1

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    logger.info("MoveIt2 ZMQ sidecar listening on tcp://%s:%d", args.host, args.port)

    try:
        while True:
            try:
                raw = socket.recv()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt; shutting down.")
                break

            try:
                request = msgpack.unpackb(raw, raw=False)
            except Exception as e:  # noqa: BLE001
                socket.send(msgpack.packb({"error": f"malformed_request:{e}"}, use_bin_type=True))
                continue

            # ``unpackb`` decodes any valid msgpack value, not just a map: the
            # single byte ``0x2a`` is the integer 42, and a string, list, nil or
            # bool decode just as cleanly. Reading ``endpoint`` off one of those
            # raises before the dispatch guard below exists to answer it, so it
            # is rejected here in the same class as bytes that do not decode at
            # all - either way the peer did not send a request. This is also
            # what makes the ``api_token`` check the fork notes suggest safe to
            # add: nothing reads a key off ``request`` until it is a map.
            if not isinstance(request, dict):
                socket.send(
                    msgpack.packb(
                        {"error": f"malformed_request:expected a msgpack map, got {type(request).__name__}"},
                        use_bin_type=True,
                    )
                )
                continue

            endpoint = request.get("endpoint", "")
            data = request.get("data") or {}

            # REQ/REP is lockstep: this peer is blocked until we reply, so
            # every request must produce exactly one response. Dispatch is
            # guarded because forks are invited to edit the handlers above
            # (start-state override, pose frame, world_update schema) and an
            # exception escaping here would close the socket with the reply
            # never sent - taking the sidecar down for every other client
            # too, not just the one that sent the bad request.
            try:
                if endpoint == "ping":
                    response = {"status": "ok"}
                elif endpoint == "reset":
                    # Reference implementation has nothing to reset; forks
                    # with stateful planners (RRT-Connect cache, etc.)
                    # should plug seed handling here.
                    seed = (data.get("options") or {}).get("seed")
                    logger.info("reset called (seed=%r); reference impl is a no-op", seed)
                    response = {"status": "ok"}
                elif endpoint == "plan":
                    response = _plan(
                        moveit_py,
                        planning_group=data.get("planning_group", args.planning_group),
                        joint_state=data.get("joint_state"),
                        target_pose=data.get("target_pose"),
                        target_joints=data.get("target_joints"),
                        world_update=data.get("world_update"),
                    )
                else:
                    response = {"error": f"unknown_endpoint:{endpoint}"}
                payload = msgpack.packb(response, use_bin_type=True)
            except Exception as e:  # noqa: BLE001 - answer the peer, keep serving
                logger.exception("Handling endpoint %r failed: %s", endpoint, e)
                payload = msgpack.packb({"error": f"internal_error:{type(e).__name__}:{e}"}, use_bin_type=True)

            socket.send(payload)
    finally:
        socket.close()
        context.term()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
