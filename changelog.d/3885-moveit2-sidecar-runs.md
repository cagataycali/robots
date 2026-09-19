### Fixed: the MoveIt2 reference sidecar starts, and plans from the state the client sends

`zmq_node` could not reach its own socket for any configuration. It built
`MoveItConfigsBuilder(robot_name="moveit2_sidecar")`, which looks for a
`moveit2_sidecar_moveit_config` package that cannot exist, and its two config
flags passed keyword arguments the builder has no parameter for
(`robot_description(package=)`) or a package name where a file path goes
(`moveit_cpp(file_path=)`) - so the documented launch command died in
`PackageNotFoundError`, and naming a real config package could not help. Past
that, `MoveItCpp` refuses to construct without the nested
`planning_pipelines.pipeline_names` parameter the config builder does not write
("Failed to load planning pipelines from parameter server"), and the
construction log called `get_planning_component_names()`, which `MoveItPy` does
not have. The config package is now `package_name=`, with `--robot-name`
naming the description inside it; both default to the panda config MoveIt 2
ships, the same one `docker-compose.yml` passes, so the reference deployment
plans out of the box.

A plan that ran could not be returned either: the waypoints were read from
`plan_result.trajectory.joint_trajectory`, and a `moveit.core` `RobotTrajectory`
carries no such attribute - every successful plan came back to the client as
`trajectory_error:`. The ROS message is reached through
`get_robot_trajectory_msg()`.

The start state was MoveIt's own, with the request's `joint_state` received and
logged as unused. Nothing publishes `/joint_states` when the robot is not a ROS 2
robot - the documented simulation deployment - so that state is the description's
default pose, which for the panda is a self-collision the
`CheckStartStateCollision` adapter aborts every plan on. The request's state is
now the start state, read in order onto the joints the planning group plans over,
with the trailing values a robot publishes but the group does not plan (the two
Panda fingers) logged as ignored; too few values is refused naming the group's
joints. The Cartesian goal's frame and link come from the robot model rather than
the `"base_link"` / `"end_effector_link"` literals, which MoveIt 2's own panda
description does not have and which `set_goal_state` accepts before failing with
"Unable to construct goal representation".

None of this failed a test because the doubles were written from the wire
protocol rather than from the binding; they now carry the `moveit_py` 2.12 shapes.
Panda in MuJoCo against the page's own fence: the flange goes from 283.3 mm to
4.6 mm from the commanded `target_pose` over 500 steps.
