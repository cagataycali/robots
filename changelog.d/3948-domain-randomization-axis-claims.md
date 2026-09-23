### Fixed: the domain-randomization guide names only the quantities the physics axis samples

`randomize_physics` writes `geom_friction` and `body_mass`; the guide described it
as scaling joint damping too, in the signature fence and in the Categories table,
while `dof_damping` (and `jnt_stiffness`, `dof_armature`, `dof_frictionloss`)
measure a zero delta and no `damping_range` exists in the signature to ask for
more -- so a reader closed the sim2real gap on the one dynamics parameter a
position-controlled arm needs most, and nothing reported that the axis had not
moved it. `examples/12_domain_randomization.py` also told the reader the sensor
noise applies "until reset"; a configuration outlives `reset()` and is dropped
only by another `set_obs_noise` or by `destroy`, which is what every other
surface says, so a per-episode collection loop read its later episodes as clean
while they were still noisy.
