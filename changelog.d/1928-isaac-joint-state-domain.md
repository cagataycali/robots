### Fixed: `set_joint_positions` applies the same joint-state domain on every backend

The kinematic state writers bypass the actuators and write joint state directly.
The MuJoCo backend refuses a value the engine cannot honor; the Isaac backend
accepted all of them, so the accepted domain depended on which engine the caller
happened to be driving. On Isaac, `positions={"shoulder": True}` wrote a
1-radian target, a `nan` / `inf` was written into the articulation (PhysX
surfaces that only from a *later* step, as an "Illegal BroadPhaseUpdateData -
non-finite bounds" error), an unresolvable joint name was skipped so a typo
wrote nothing - or half a pose - while the caller was told the pose had been
applied, and a vector of the wrong length resized the articulation's
joint-position array instead of being refused. All eleven reported
`status="success"`.

A non-numeric value was worse still: on the main thread it raised `ValueError`
past the structured-error contract, and from a worker thread the write is queued
and the pump's queued-action handler swallows that failure as best-effort - so
the same call returned `status="success"` and failed with no channel at all. The
verdict depended on which thread called.

The value half of the contract now lives on the engine base as
`SimEngine._coerce_joint_state_map`, promoted from the MuJoCo backend with its
message text unchanged, so the two backends cannot drift apart again. Isaac
applies it, plus the structural checks its sibling already enforced: the list
form's length must match the robot's joint count, every mapping key must name one
of the robot's joints, and the mapping must not be empty. Validation runs
synchronously, before the write is queued, so a rejected value is reported to the
caller rather than swallowed on the pump thread. Every usable call is unaffected.
