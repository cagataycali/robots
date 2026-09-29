### Fixed: the native Feetech and Dynamixel drivers take a leader arm through `attach_teleop`

`Robot("so101_leader", mode="real")` refuses and points at
`Robot("<follower>", mode="real", port=...).attach_teleop("so101_leader", port=...)`,
but with `driver="strands"` that call raised
`AttributeError: 'FeetechDriver' object has no attribute 'attach_teleop'`.
`FeetechDriver` (and `DynamixelDriver`, which subclasses it) now composes
`TeleopMixin` like the hardware `Robot` and the MuJoCo simulation. `stop()` joins
a running teleop loop before it turns the torque off. Because the native bus
commands degrees, `attach_teleop` refuses a leader whose motor table reports
another unit (`use_degrees=False`, or `koch_leader`'s `RANGE_M100_100`) unless
`map_fn=` converts it.
