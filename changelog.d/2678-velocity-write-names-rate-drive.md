### Fixed

`set_joint_velocities` now names the joints whose actuator is commanding a
different rate. The write lands and `qvel` reads back what was asked for, so
`status` was `success` and the report was the plain `Set 1/1 joint velocities` --
but on a joint driven by a `<velocity>` actuator that `ctrl` **is** the joint's
own rate, so the drive was already commanding a velocity of its own and the next
step resolved the disagreement in the drive's favour. Measured on a one-hinge bar
with gravity off and `kv=2`: writing `+2.0` rad/s while the drive commanded
`-3.0` left the joint at `-0.2532` rad after 72 steps, and writing the same
`+2.0` while the drive commanded `+2.0` left it at `+0.2391`. Two opposite
outcomes, and one byte-identical report.

The sibling `set_joint_positions` had already settled the shape of that answer
for a position servo, naming the joint and the setpoint that overrides it. Only
the velocity half had no equivalent, and its remedy is not the same one:
`hold=True` moves a servo's setpoint, while a rate drive's `ctrl` already is the
rate, so the report points at `send_action`.

`scene_ops.joint_rate_drive_map` is the velocity counterpart of the existing
`joint_drive_map`, reading the same compiled fields: an affine bias with no
position feedback but negative velocity feedback (`biasprm = [0, 0, -kv]`), a
stateless `dyntype`, and a joint transmission resolved through
`actuator_joint_id`. Measured against every MuJoCo actuator shortcut, `<velocity>`
is the only one that clears all of them -- a torque motor, a position servo,
`<intvelocity>` (whose `ctrl` is integrated into a pose), `<damper>` (whose `ctrl`
scales damping rather than commanding a rate) and `<cylinder>` are all excluded,
and a `<general>` spelling `<velocity>` longhand classifies identically, which is
the point of reading the compiled fields rather than the element. Slot 2 has to
be negative rather than merely non-zero: positive velocity feedback is
anti-damping, and a written rate against it diverges rather than settling on any
commanded value.

A tendon rate drive is not reported. Its `ctrl` moves a tendon length rate, not
the joint's, so there is no rate in the joint's units for the write to disagree
with -- which is why the transmission is resolved through `actuator_joint_id`
rather than read raw.

Report only: no verdict, no accepted input and no written value moves, and a
drive that happens to be commanding the written rate keeps the plain report
byte-for-byte.
