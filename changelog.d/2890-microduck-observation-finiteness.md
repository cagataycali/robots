### Fixed: a non-finite observation component is refused instead of reaching the ONNX graph

`strands_robots.policies.microduck.build_observation` held its two floating-base
blocks to a width, and a width was the only thing it held anything to. Every
value it reads passed into the vector the ONNX graph consumes without anyone
asking whether it was a number: `base_ang_vel` and `base_quat` from the caller's
observation dict, the per-joint `<joint>` and `<joint>.vel` scalars from the same
dict, the previous action, and the command.

All six paths reported success at the documented `48 + len(command)` width, so a
poisoned observation is shaped exactly like a healthy one and nothing downstream
can screen it by shape. What happened next depended on the checkpoint. A graph
that tolerates the value answered with a number nobody could trace. A graph that
propagates it - which is what any real one does - returned a `nan` action that
`get_actions`' own finiteness guard then refused **as** `'the ONNX action'`,
reporting the checkpoint's graph for a `nan` the caller's observation supplied.

The builder now refuses it first, and the message names the offending block; a
joint block names the joint. The check runs once, on the assembled vector, which
is the single place all six input paths meet - guarding each input separately
would be six checks that drift apart and would still miss whatever a later block
adds. It is a plain `numpy.isfinite` rather than the shared vector domain for a
measured reason: at that point the value is a `float32` 1-D array the builder
itself made, so none of the spellings that domain exists to judge can reach it,
the two agree on everything that can, and the shared domain costs more than the
whole build (40.68 us against a 37.63 us build) where the `isfinite` pass costs
2.00 us end to end (+5.31%).

The width axis is unchanged: a block that is both wrong-width and non-finite is
still reported by width, which is the more basic mistake. A large finite reading
is still a reading - the guard refuses `nan`/`inf`, not magnitude.
