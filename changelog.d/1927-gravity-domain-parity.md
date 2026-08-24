### Fixed: every gravity surface applies the one shared gravity domain

`SimEngine._normalize_gravity` is the shared domain for a gravity argument, and
two of the five public gravity surfaces carried a local copy of it instead:
`NewtonSimEngine.set_gravity` and `IsaacSimulation.create_world`. Both copies
drifted from the shared rule in both directions.

A boolean was applied as a magnitude. `bool` is an `int` subclass, so a local
`float()` coercion accepted it: `set_gravity(True)` wrote `[0, 0, +1.0]` onto the
Newton world and rebuilt the model under `status="success"` - gravity pointing
*up* - while `set_gravity(False)` switched gravity off. A boolean in any
component of a vector did the same, and `create_world(gravity=True)` configured
`+1.0` on the Isaac physics context the same way.

A value the other backends honour was refused. Both copies keyed on
`isinstance(gravity, (int, float))` / `(list, tuple)` rather than on a real
number plus a length, so `np.float32(-9.81)` was refused by both and a NumPy
gravity *vector* was refused by Isaac - values MuJoCo accepts. Newton reported
the first as `'gravity' must be a 3-element list of numbers (object of type
'numpy.float32' has no len())`, naming a NumPy internal rather than the
parameter.

Both surfaces now normalize through the shared domain, which also removes the
two copies. Isaac keeps its own Z-alignment constraint and applies it to the
normalized components, so a non-Z-aligned NumPy vector is now refused for being
off-axis rather than for its type. Measured over twelve values, Newton and Isaac
disagreed with the MuJoCo reference on 9 of 24 verdicts before this change and
on 0 after it. A wrong-length gravity on Isaac now reports the shared sentence
(`must be a 3-element list [x,y,z]`) that the other two backends already emit.
