### Fixed: `set_joint_velocities` refuses a value MuJoCo would answer by resetting the world

`velocities={"Rotation": 1e300}` on so100 returned `success`; the next `step`
printed MuJoCo's `WARNING: Nan, Inf or huge value in QVEL ... The simulation is
unstable` to stderr and reset every joint and object to its initial state. The
write now applies mj_step's own test (finite and below `mjMAXVAL` on `qvel` and
the `qacc` it produces) under a checkpoint, and refuses with the state unchanged.
Ordinary velocities are unaffected.
