### Fixed: a refused `MuJoCoSimEngine` argument no longer strands a ROS 2 node

`MuJoCoSimEngine.__init__` builds the optional ROS 2 bridge part-way through
construction, and a comment above that call already stated the rule it works to:
"a refusal about a constructor argument should not leave a ROS 2 node behind it"
- which is why the `default_width` / `default_height` guards sit above it. One
argument still did not: `mesh=`, whose documented `TypeError` for a value the
engine cannot `.stop()` was raised after `_init_ros_bridge` had created the
`strands_sim` node and called `rclpy.init()`.

`MuJoCoSimEngine(ros2_bridge=True, mesh=True)` therefore refused having already
built a live node and initialized the rclpy context, and `__init__` raising
returns no object - so the `cleanup()` that calls `_shutdown_ros_bridge` is
unreachable by construction. A corrected retry cannot release it either: the
leaked bridge recorded `_owns_context`, so the retry's bridge finds the context
already up, declines ownership, and on teardown destroys only its own node,
leaving the first alive and the context initialized with nobody left to shut it
down. Measured with a recording rclpy double: one undestroyed node and
`rclpy.ok()` true after the refusal, two nodes and `[False, True]` destroyed
after the retry, and `rclpy.shutdown()` never called.

The handle is now resolved alongside the resolution pair above the bridge, so
every argument this constructor can refuse is answered before anything is built.
This is the same placement `HardwareRosBridge` converged on for `joint_limits`,
the same order the hardware `Robot` already gets for free by initializing its
bridge last, and the discipline `NewtonSimEngine` states outright ("state that
teardown touches must be set before any fallible construction step"). The guard
moved rather than tightened: a stoppable client is still accepted, stored on
`.mesh`, and stopped by `cleanup()`. The MuJoCo backend is the only one that
both builds a bridge mid-construction and can refuse an argument after it.
