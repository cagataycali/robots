### Fixed: `pose_tool` reads a joint's degree scale from the arm's calibration

The tool carried a fixed per-joint degree table (`shoulder_lift` and
`wrist_flex` declared `+-90`, `elbow_flex` `+-150`, all across a full turn of the
encoder) while `FeetechDriver` reads the travel `lerobot-calibrate` measured for
that arm, so the package's two Feetech stacks quoted one servo two ways: a joint
moved twice as far as it was asked to and reported half the angle it moved, and
`0 percent closed` commanded the gripper two thousand counts past its closed
stop. Graded against the installed LeRobot on one physical SO-101 follower's
records, all 18 readings and all 18 encoded targets disagreed, the worst by 52.6
degrees.

Both directions of the conversion and the bounds a target is refused against now
come from `FeetechBus`, and `pose_tool(calibration=...)` takes the path of that
arm's JSON exactly as `FeetechDriver(calibration=...)` does; omitting it reads
and commands the servo's whole rotation. A target outside the travel is refused
rather than clamped onto the mechanical limit, so a joint that cannot honour a
target is reported as uncommanded instead of driven to its end stop under a
success message echoing the value asked for.
