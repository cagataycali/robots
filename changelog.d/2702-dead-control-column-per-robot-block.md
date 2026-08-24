### Fixed: `verify-dataset` grades a dead control column per robot, not per whole vector

`verify_dataset`'s dead-control-column check exists to catch a recording whose `action` or
`observation.state` column was written as zeros because the writer's keys never resolved to the
declared columns - correct episode counts, correct pixels, no control signal. It fired only when
the ENTIRE vector was identically zero, and a multi-robot recording cannot present that shape.
`start_recording` declares one column per robot per joint and prefixes each with its robot's
instance name (`alice__shoulder_pan`), so a resolution that succeeds for one robot and fails for
another leaves that robot's whole block of columns zero while the other robot's columns carry
real measurements. The vector as a whole varies, so the check saw nothing and the report read
`[PASS]`.

The recorder documents this mechanism itself, in the comment that remaps the prefixed keys in
`strands_robots/simulation/mujoco/simulation.py`: without the remap, `add_frame` "looks up the
prefixed schema keys, finds nothing, and writes all-zero state/action vectors silently". That is
the per-robot failure, and the gate written to catch it could only see the all-robot case.

The vector is now split into the blocks `meta/info.json` declares - column indices grouped by the
per-robot prefix - and a block that is wholly zero is reported the same way a wholly-zero vector
is, naming the robot: `feature 'action' is identically zero for every 'bob' column across episode
0 (60 frame(s))`. A zero SUBSET of one robot's block is deliberately left alone, because a
gripper parked at zero for a whole episode is a measurement rather than a writer fault. The
separator is the doubled underscore `start_recording` writes, not a single one: a real arm's
joints are `shoulder_pan` / `gripper`, and splitting on `_` would turn one arm into four blocks
and flag its parked gripper.

The change is a strict extension. A wholly-zero vector reports the message it reported before,
once, rather than one per block; a dataset whose `names` are absent, non-string or of a different
width than the stats it is being asked about keeps the whole-vector rule, which is every
single-robot recording; and `stats_vectors_checked` is unchanged, so the same work yields a finer
verdict. Over the LeRobot datasets on one machine - 1314 roots, graded on both sides - 58 moved
from pass to fail, no dataset moved from fail to pass, every problem string reported before was
still reported, and every added string was the new per-block finding.

This complements rather than duplicates the recorder fix in #2699. That change makes an undriven
robot's `observation.state` columns measurements from now on, and states that the matching action
columns are deliberately unchanged pending a schema decision. Neither can repair a dataset already
on disk, and a recording made after it still carries an undriven robot's `action` block as zeros -
which is exactly what this check now reports, without deciding the schema question: on a two-arm
recording made on current `main`, the `action` block is named and the `observation.state` block is
not, because that one now varies.
