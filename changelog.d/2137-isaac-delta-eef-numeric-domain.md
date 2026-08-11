### Fixed: the Isaac delta-EEF controller accepted `inf` for every scale it bounds, and any value at all for a gripper reference

`IsaacDeltaEEFController.__init__` bounded `pos_scale`, `rot_scale` and
`damping` with an order comparison (`> 0`), and bounded the `joint_limits`
clip table with another (`lower > upper`). An order comparison cannot reject
`inf` -- `inf > 0` is `True` -- and it cannot see a `nan` row either, since
`nan > nan` is `False`. `gripper_open` and `gripper_close` had no numeric
bound at all. Every such value was accepted at construction and then made
`compute_joint_targets` return `nan` for all nine joints, which
`send_action`'s action-value domain refuses one action at a time while
naming the *joint* it was handed -- so a bad scale read as a per-action
embodiment problem rather than as the configuration it was. A non-finite
`gripper_close` was worse than loud: approach and hold actions were
unaffected and it surfaced only at the first grasp.

Each numeric knob now clears the shared scalar domain the rest of the
library uses -- `positive_finite_number_error` for the two scales and the
damping lambda, `finite_number_error` for the two gripper references (a
joint reference of either sign is usable) -- and the limits table is checked
for finiteness before it is checked for order. A refused value is reported
against the parameter it came from, before either injected kinematics read
is called. The guards also make the constructor's own message reachable for
a non-numeric value, which previously escaped as a bare `float()` conversion
error.
