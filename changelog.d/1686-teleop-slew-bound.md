### Fixed: a teleop stream cannot command a joint faster than it can travel

`InputReceiver` bounded each inbound teleop frame on four axes - who sent it,
how fresh it is, how densely frames may arrive (`STRANDS_MESH_INPUT_MAX_HZ`)
and how large a single value may be (`MAX_INPUT_VALUE_ABS`) - but every one of
those judged a frame in isolation. Nothing bounded the distance between
consecutive commands for the same joint, so a stream inside all four caps could
still reverse a joint full-scale on every frame. Measured on a MuJoCo Panda
follower at 50 Hz, half the permitted frame rate: 60 of 60 reversals applied,
`rejected` and `rate_dropped` both zero, each frame commanding 90 units/s -
roughly 14x the no-load speed of the Feetech STS3215 servos on an SO-100 class
arm, which is the overcurrent / gear-strip trajectory the rate cap exists to
prevent in the time domain.

Frames are now also bounded on per-joint speed, via
`security.input_frame_slew_violation` and `STRANDS_MESH_INPUT_SLEW_ABS`
(default `8pi` units/second, above what a leader arm's own servos can produce,
so only a synthetic stream trips it). An over-speed frame is refused and
counted in the new `slew_rejected` stat, matching how every other guard on this
path behaves; the commanded value is never silently altered.

The baseline each command is measured against is kept *per joint* and merged on
every apply, because the shape of the frames is the sender's choice: a stream
that interleaves single-joint frames would otherwise erase the baseline of the
joint it is about to reverse, so every frame would arrive with no reference and
the bound would never fire. Measured on the same follower, that stream applied
60 of 60 frames at 45 units/s with `slew_rejected` at zero; it is now refused.
The baseline is pruned of entries old enough that no permissible command could
exceed the bound from them, which cannot change a verdict and keeps a mapping
keyed by sender-chosen joint names bounded.

Because the bound is a speed measured from each joint's own last applied
command, the allowance grows while that joint is not moving: a refused stream
resumes on its own once the commanded pose is reachable safely, with no resync
handshake, and a joint that pauses while others move is not over-refused when it
starts again. The interval charged to a move is floored at the minimum
inter-apply interval the rate cap guarantees, so a batched delivery - whose
intermediate commands are superseded before an actuator can act on them - is not
mistaken for a high-speed command, and the two guards compose rather than
contradict.
