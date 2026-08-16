### Fixed: the remaining simulation input validators refuse a boolean instead of coercing it to 1.0

#1837 settled the actuator command and #1838 the runtime state writers. Six
input validators in `strands_robots/simulation/` were not routed through the
shared predicate those two established (`utils.is_boolean`), and they failed in
two distinct ways.

**Two had no boolean check at all.** `_coerce_finite_vector` and
`_normalize_gravity` coerced with a bare `float()`, so a boolean arrived as a
silent `1.0` under `status="success"`:

```
set_gravity(True)              -> success   gravity [0, 0, +1.0] - pointing *up*
set_gravity([True, 0, 0])      -> success   gravity [1.0, 0, 0]
raycast(origin=[True, 0, 0])   -> success   cast from x = 1.0
set_obs_noise(std=True)        -> success   a noise sigma of 1.0
randomize(mass_range=(True, 2.0)) -> success   a scale range of (1.0, 2.0)
```

`_coerce_finite_vector` is one chokepoint for seven call sites, so the gate lands
in front of `raycast`'s origin and direction, `set_geom_properties`' size and
friction, an rgba colour, and each ray of a `multi_raycast` batch.

The domain it contradicted was already written down: `utils.finite_vector_error`
refuses a bool component because `float(True)` would "silently write `1.0` where
a coordinate, extent or colour channel belongs" - and its docstring then defers a
colour to the rgba coercion in `simulation.mujoco.physics`, which is the helper
that accepted one. A caller who could not place a body at `True` could still cast
a ray from `True`.

**Two documented the refusal and did not implement it.** `_validate_timestep` and
`_validate_mass` both already stated that a bool is "rejected explicitly since
`True` would act as a silent 1-second step" / "1 kg body", and implemented it as
`isinstance(value, bool)`. `numpy.bool_` is not a `bool` subclass, so the guard
held for a hand-typed literal and vanished for the spelling computed code
produces (`gripper > 0.5`):

```
set_timestep(np.True_)              -> success   dt = 1.0 s
set_body_properties(mass=np.True_)  -> success   a 1 kg body
```

A `dt` of one second is not a mis-sized step, it is a different simulation.

All six now route through `utils.is_boolean`, which covers a python `bool`, a
`numpy.bool_` scalar and a 0-d boolean array. Each refusal names the parameter,
says "not a bool" rather than describing a generic non-number, and carries the
reason; the vector surfaces carry a reason naming a coordinate, extent or colour
channel rather than the radians / rad/s / newtons that belong to the joint
writers. Because every check runs before any write, a refused value leaves
`opt.gravity`, `opt.timestep`, `body_mass` and the geom fields untouched.

The gate keys on the **type, not the value**: `1`, `1.0`, `np.uint8(1)`,
`np.int64(1)`, `np.float64` and a 0-d numeric array remain accepted at every
surface, so `set_timestep(1.0)` is still a legal request and `set_gravity(1)`
still means 1 m/s^2. `nan` / `inf`, wrong component counts and non-numeric values
keep their own distinct messages - the bool gate is additive.

One pre-existing assertion changed rather than being deleted.
`test_numpy_bool_scalar_still_rejected` passed by accident: `numpy.bool_` is not
`numbers.Real`, so it missed `_normalize_gravity`'s scalar branch, fell through to
`len()` and surfaced as `"'gravity' must be a 3-element list of numbers (len() of
unsized object)"` - a component-count complaint about a value with no components,
while a plain `True` was accepted outright on the same code. Its conclusion still
holds and is now reached deliberately, so the assertion moved onto the bool
wording and the docstring records why.

A structural guard (`TestTheBooleanDomainIsStructurallyClosed`) now enumerates the
input-validation coercions under `strands_robots/simulation/` and fails when one
coerces caller input with a bare `float()` while neither consulting the shared
predicate nor being listed as out of domain with a reason. This was the third pass
over one decision because nothing enumerated the surfaces; the guard is what makes
it the last.
