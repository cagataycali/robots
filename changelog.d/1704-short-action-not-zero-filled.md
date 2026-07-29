### Fixed: an action vector shorter than the actuator list no longer fabricates zero commands

Both LeRobot providers map a policy's flat action vector onto actuator names by
index. When the vector was shorter than the actuator list - a 6-DOF checkpoint
pointed at a 7-actuator robot, an embodiment declaring a gripper the checkpoint
never learned - the unmatched actuators were filled with `0.0` and sent anyway.

Every action space on that path is absolute position: a LeRobot `<motor>.pos`
follower and a MuJoCo position actuator both read `0.0` as "travel to zero", not
as "hold". So the padding was a command. A 4-of-6 SO-arm action drove
`wrist_roll` and `gripper` to their zero targets at servo speed, closing the
gripper on whatever the arm was holding. The library's own diagnostic described
the opposite outcome - "the unmatched actuator(s) are zero-filled and will not
move" - so the documented intent was already that those actuators hold.

An unmatched actuator is now omitted from the action dict, which is what makes
it hold. Measured on two position-controlled joints parked at 0.8 rad, a
one-value action left the unmatched joint at 0.8 rad instead of driving it to
0.0. `pad_short_actions=True` restores the previous zero-fill for a consumer
that needs a fixed-width action dict, and the dim-mismatch warning now reports
whichever consequence is in effect rather than one that was never true.

`LerobotLocalPolicy` and `LerobotAsyncPolicy` each carried their own copy of the
index mapping, so the rule now lives once in
`strands_robots.policies.align_action_values` alongside `resolve_chunk_length`,
with a parity test pinning that the two providers agree.
