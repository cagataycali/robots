### Fixed: a Newton joint observation carries the velocity a locomotion policy reads

`get_observation` on the Newton backend emitted joint positions only, while the
MuJoCo backend pairs each position with a `<joint>.vel` reading. A
velocity-feedback policy reads that pair by name, so the shipped Microduck
walking policy raised `KeyError: 'left_hip_yaw.vel'` on its first tick and
`run_policy` returned `steps_used=0`. The velocity was already read -
`get_robot_state` reports it from `joint_qd` via the per-joint DOF index - so
the observation now reads the same index beside each position, and
`set_obs_noise`'s `joint_vel_std` governs those entries instead of
`joint_pos_std`.
