### Fixed: `WBCConfig` refuses a value it cannot honor instead of driving the robot with it

`WBCConfig.__post_init__` rejected impossible dimensions and accepted every
numeric *value*. Each field is read verbatim into the SONIC PD law that writes
`data.ctrl` or into the observation the network sees, so an unusable one became a
wrong torque rather than an error. Measured on a real MuJoCo `unitree_g1` driven
by the real `compute_targets` -> `pd_control` chain: `action_scale=0.0` returned
`status="success"` having discarded the network output entirely (5.7 Nm peak
instead of 21.5, base sagging to 0.309 m), `kps=[-150.0]*15` returned success at
461.9 Nm with the feedback driving every joint away from its target, and
`action_scale=nan` failed with a message blaming the embodiment. A `None` or a
numeric string raised a bare `TypeError` from the per-tick `float()` in
`compute_targets`, i.e. mid-rollout - the failure this module's docstring says it
exists to convert into a construction-time message.

`action_scale` must now be a finite number `> 0`, the `kps`/`kds` components
finite and `>= 0` (a zero gain is a pure-damping joint and stays first-class, a
negative one is not), and `default_angles`, `cmd_scale`, `rpy_cmd`,
`obs_scales`, `height_cmd` and `freq_cmd` finite. The existing dimension checks
also became total: a per-joint field carrying no readable length now names the
field instead of raising `len()`'s bare `TypeError`, and a NumPy vector of the
right width is accepted rather than raising the ambiguous-truth `ValueError`.
