### Fixed: the ROS 2 bridge QoS depth is checked where the caller gives it

`RosTelemetryBridge` and `HardwareRosBridge` take a `qos_depth` and hand it to rclpy's
`create_publisher` / `create_subscription` as the `qos_or_depth` argument, which becomes
`QoSProfile(depth=value, history=KEEP_LAST)`. Every sibling constructor parameter of that pair is
measured at the boundary - `domain_id` against the RTPS port map, `spin_period` against the positive
finite domain, `enable_commands` against the boolean domain, `joint_limits` against a finite numeric
pair, each with the reason written down - and this one was handed through unchecked. It is also the
only one whose consumer is not reached from the constructor: publishers are built lazily, on the
first `publish_joint_states` / `publish_image` for a robot.

So a bad depth reported a successful construction. Measured against rclpy on ROS 2 Jazzy, `0` and
`False` were *accepted*: rclpy warns "A zero depth with KEEP_LAST doesn't make sense; no data could
be stored. This will be interpreted as SYSTEM_DEFAULT" and builds the endpoint with the middleware
default, so the declared depth was silently not the depth in force - reported by a `UserWarning`
that `warnings.warn` shows once per location, so a second bridge in the same process said nothing at
all. `True` was accepted as a silent depth of 1: one frame of history on a stream the caller asked
to buffer.

Every other spelling raised from inside rclpy, naming neither the parameter nor the bridge. `-1`
gave `ValueError: history depth must be greater than or equal to zero`; `2.5`, `10.0`, `"10"`,
`None`, `nan`, `inf` and `np.int64(10)` gave `TypeError: Expected QoSProfile or int`; and `2**31`
and above failed the pybind11 conversion into the QoS profile instead. On the telemetry-only path
none of those surfaced until the first frame was published, mid-run. On the command path
`create_subscription` runs in the constructor, so the same mistake raised there - after the
process-wide `ROS_DOMAIN_ID` write and after the node had been created.

The depth is now measured against `positive_count_error`, the shared domain for a discrete count a C
API consumes directly rather than coerces, which is exactly why the strict-`int` requirement is the
right one here: rclpy refuses `np.int64(10)` outright even though it names a perfectly good depth.
The floor of 1 closes `0` and `False`, and the domain's boolean refusal closes `True`. Only the
transport's own ceiling is added on top, beside the transport that has it, in the manner of
`_transport_port_error` in `strands_robots.tools.use_rosbridge`: the depth is stored in
`rmw_qos_profile_t` through a pybind11 binding taking a signed 32-bit integer, so
`MAX_QOS_HISTORY_DEPTH` is `2**31 - 1` - measured, not chosen, and pinned by the arithmetic so this
library cannot start refusing a depth the transport is happy to carry.

The guard is placed beside the `domain_id` check, ahead of both the process-wide `ROS_DOMAIN_ID`
write and the rclpy probe. That is what makes a refused depth leave the environment as it found it,
and makes the same caller mistake report identically on an install with the `[ros2]` extra and one
without it - so the refusal is checked on a minimal install too. An accepted depth still reaches
every endpoint verbatim: nothing coerces it, because a converted value is a depth on the wire the
caller never named.

Nothing observed any of this before. `qos_depth` appeared in no test, no doc and no example, and the
one rclpy double in the suite that reaches `create_publisher` takes the depth as `_depth` and
discards it, so no existing test could see which depth an endpoint was built with. The new tests
record it instead.
