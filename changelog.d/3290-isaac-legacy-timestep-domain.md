### Fixed: the legacy Isaac timestep shortcut is graded before it is converted

`IsaacSimulation(default_timestep=...)` is a second way to set
`IsaacConfig.physics_dt`, and it wrote the field through `float(...)`. That
conversion is exactly what the shared timestep domain cannot undo - its boolean
arm exists because "`float(True)` is `1.0`, so the boolean is unrecoverable once
coerced" - so `default_timestep=True` stored a one-second physics step, 120x the
default, and `create_world()` reported `status="success"` on it, while the
canonical `IsaacConfig(physics_dt=True)` spelling of the same value is refused
there. `numpy.True_` behaved the same. A one-second `dt` is not a coarse
simulation: replayed in MuJoCo, a 0.60 m fall completes between two consecutive
observations - no trajectory for a policy to act on - and the contact resolves
45x worse, the box settling 4.92 mm inside the floor. `nan` and `inf` were
stored to be refused a call later under `physics_dt`, the knob the caller never
spelled; `False`, `0`, `-inf` and `-0.002` were refused by the field's own
`<= 0` test after conversion, so `False` was reported as `0.0`; and `[0.002]`
raised `TypeError` out of `float()`, naming neither the parameter nor the class.
All eleven unusable values now raise `ValueError` naming `default_timestep`, on
the same `SimEngine._validate_timestep` domain `create_world` applies to the
effective dt. Conversion happens only once the value is admitted, so nothing
that worked stops working - including a numeric string, which that domain
accepts.
