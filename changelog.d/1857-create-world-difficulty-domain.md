### Fixed: `create_world(difficulty=...)` accepts one domain on every backend

`difficulty` scales a heightfield terrain's peak elevation - the curriculum knob
a locomotion trainer ramps across resets. Every backend read it through a bare
`float(difficulty)`, so the three disagreed on which values are numbers at all:
`None` and `[0.5]` raised `TypeError`, which escapes the `{"status": "error"}`
tool-result contract because the callers catch only `ValueError`; a non-numeric
string surfaced `float()`'s own message on MuJoCo (naming neither the parameter
nor the surface) and escaped outright on Newton and Isaac; `bool` was accepted
asymmetrically, with `True` passing as a silent `1.0` - a full-height terrain
indistinguishable from the default - while `False` was refused; and a numeric
string was silently honored as a scale.

The domain is now owned by `simulation.terrain.validate_difficulty`, the raising
binding over the shared `utils.positive_finite_number_error`, and all three
`create_world` implementations report through that one binding. A value one
backend refuses can no longer be honored by another, and a rejected curriculum
step is reported instead of aborting the ramp. The accepted domain is unchanged:
a positive finite real still compiles into the heightfield exactly as before.
