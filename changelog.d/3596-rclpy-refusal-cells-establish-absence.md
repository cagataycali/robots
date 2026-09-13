### Tests: an rclpy refusal cell makes rclpy absent instead of assuming it is

`rclpy` ships with a ROS 2 distribution rather than from PyPI, so on the hosts
this suite usually runs on it is simply missing and `require_optional("rclpy")`
raises without being asked to. Five cells took that for granted, and on a host
with a distro sourced they did not merely fail - four of them ran on past the
probe they were checking the guard for, into
`RosTelemetryBridge.__init__`'s `rclpy.init()` and `create_node()`. Measured
with ROS 2 Jazzy sourced: `rclpy.ok()` False before and True after, a live
`strands_robots` node on domain 11, and nothing destroying either, because the
cell asserts `ROS_DOMAIN_ID` and returns.

The fifth carried a `sys.meta_path` finder and was documented as holding
"whether or not the interpreter running the suite happens to have a ROS 2 distro
sourced". It does not: an import consults `sys.modules` first and reaches the
finders only when it misses, so the finder is bypassed once anything in the
session has imported `rclpy`, and the refusal under test never runs. That cell
passes alone and failed in the full suite.

All five now use `tests._blocked_module.blocked("rclpy")`, which does both halves
- `sys.modules[name] = None` and dropping `require_optional`'s memo - and
restores both, so they hold wherever they run. Each sat beside a `cyclonedds`
sibling that already established its own absence this way.

`tests/test_rclpy_refusal_cells_establish_the_absence.py` reports the next one.
Its surfaces are derived rather than listed: the classes whose `__init__` calls
`require_optional("rclpy", ...)`, plus their subclasses, so a new rclpy-probing
bridge is covered the day it lands. Scoping to `__init__` is what keeps a
`Robot` built for an unrelated missing dependency out of it - `Robot` probes
rclpy from `_check_ros2_bridge_deps`, reached only for `ros2_bridge=True`.
