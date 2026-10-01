### Fixed: Isaac refuses a joint write outside the joint's range, and stops reporting success once the physics diverged

`set_joint_positions` checked only that each value was finite. A value outside
the joint's limits was written: on so100, `{"Elbow": 50}` (degrees sent as
radians) was reported as set and 60 steps later every joint was NaN, and
`{"Rotation": 2.5}` on a [-1.92, 1.92] joint snapped back and kicked the wrist.
The write is now refused, naming the joint, its range and unit, with the
degree reading when that is what fits - the MuJoCo backend's message - and
nothing is written.

Once an articulation's joint state is no longer finite, `step` and
`send_action` return an error naming the robot and the non-finite joints, with
`reset()` as the remedy, instead of "Stepped 1x" and "Action applied" - so a
rollout no longer runs its whole horizon on a robot that is not being simulated.
