### Changed: the simulation tool's description says nothing in it reaches hardware

The `so100_sim`-style tool description told the model "sim and real with zero
code changes", which reads as if `destroy`, `apply_force` or `set_joint_positions`
could be the same call on a robot. No action of this tool has a hardware branch;
the description now says so, marks `destroy` as irreversible, and states that
`register_urdf` overrides a robot name for this process only (nothing is written
to disk).
