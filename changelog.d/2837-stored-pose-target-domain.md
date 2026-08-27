### Fixed: a stored pose is held to the travel its joints are driven within

`load_pose` read a pose off disk and drove its positions straight into
`move_multiple_motors`. `MotorController.degrees_to_position` clamps every
target into the joint's configured `range` before scaling it onto the 12-bit
`Goal_Position` register, so a stored value outside that range was not refused
but silently rewritten to the mechanical limit -- while the envelope reported
`Moved to pose '<name>'` and echoed `target_positions` verbatim from the
artifact. Measured on `shoulder_pan` (configured `(-180, 180)`), a pose naming
`999` drove `Goal_Position` 4095, which is +180 degrees, and reported `999`; a
stored `NaN` reached the same end stop, because `min(max_deg, nan)` returns
`max_deg`, and `NaN` round-trips through the pose file as a JSON literal.

The identical value supplied as an argument was already refused:
`_joint_target_error` holds `position`, `delta` and the values of `positions` to
each joint's travel. So one tool answered two ways for one number depending on
whether it arrived in the call or in a file, driving the same servo through the
same conversion. `load_pose` now routes its stored positions through that same
authority, restating no bound of its own, and refuses before the port is opened
for the reason the argument check gives -- the arm must not travel to an end stop
on a target that could not be honored. The refusal names the pose as well as the
joint and the bound, because a stored value is corrected in a file rather than in
the next call.

The deferral this replaces could not fire, which corrects the record left when
the argument guard landed: `PoseManager.validate_pose` consults the pose's own
optional `safety_bounds` and answers `No safety bounds defined` when the field is
absent, and the single `store_pose` call site never supplies it -- so every pose
this tool writes passed that check unconditionally. The arm's travel is a
property of the arm, not an annotation a pose file has to carry.

`reset_to_home` stays out of scope and is now pinned by reading its literals
rather than by assertion: it supplies its own targets and every one of them is
inside its joint's travel. A stored motor with no configured `range` still has no
travel to be held to and is left to the mover that cannot address it, while
finiteness -- which needs no range -- still applies.
