### Changed: `set_obs_noise` is one implementation, and clearing it says so on every backend

MuJoCo, Isaac and Newton each carried their own `set_obs_noise` plus the passes
that apply it. They now all inherit
`strands_robots.simulation.obs_noise.ObservationNoiseMixin`. The copies had
drifted on the result: on MuJoCo and Newton an all-zero call stored a zero
config and replied `Sensor noise: joint_pos_std=0.0, ...`, and neither echoed
the config back. Every backend now answers the way Isaac did: a `json` block
with the stored stds and seed, and an all-zero call clears the config and
replies `Sensor noise cleared.` Observations are unchanged either way.
