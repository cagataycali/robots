### Fixed: the mock policy clips to the range that holds its command

`MockPolicy.set_sim_context` learned its clipping bounds from
`actuator_ctrllimited` alone, which is one of the two ways MuJoCo holds an
actuator's command to a range. An actuator whose MJCF authors neither
`ctrlrange` nor `inheritrange` compiles to `ctrlrange (0, 0)` with
`ctrllimited=0`, and for a position servo its `ctrl` IS the joint target, so
the driven joint's limits bound the pose. Every so101 actuator is in that
second case, so the mock learned no bounds there at all and commanded its jaw
past the joint's `-0.1745` limit - the out-of-range warning the method exists
to prevent. It now reads
`simulation.mujoco.scene_ops.effective_ctrl_range`, the rule `set_gripper`
and the engine's own warning already share.
