### Fixed: a hardware bridge loop period that cannot pace a thread is refused rather than run

`HardwareRosBridge`'s `spin_period` and `HardwareRtpsBridge`'s `poll_period` name the
cadence of the background thread that services inbound joint commands, and the value is
handed to `threading.Event.wait`. Neither was validated, so a value that names no cadence
was accepted and simply run: `0`, a negative and `nan` all return from `wait` immediately,
turning the loop into an unbounded busy-spin, and `inf` raises `OverflowError` out of `wait`
and kills the loop thread, leaving a bridge that reported a successful construction with a
command surface that never delivers again. `True` was a silent one-second period and a
numeric string was quietly parsed, both by the `float()` conversion that stood in for a check.

Both now share `positive_finite_number_error` - the same domain a control frequency, a
rollout duration and a teleop rate already use - placed ahead of each surface's transport
probe, so the same caller mistake reports identically with and without the `[ros2]` extra.
On the rclpy bridge the guard also precedes the process-wide `ROS_DOMAIN_ID` write, so a
refused period no longer leaves the environment pointing at a domain that construction
never reached. The `float()` conversion stays *after* the guard rather than instead of it:
the shared domain accepts any real scalar and `Event.wait` rejects a `np.float32`, so the
conversion is what makes an accepted value consumable.
