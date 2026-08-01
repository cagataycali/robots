### Fixed: `add_object` validates its `mass` on every simulation backend

`SimEngine._validate_mass` is a static method on the base class all three
backends inherit, and its docstring already states why the domain exists: a mass
outside `(0, inf)` "does not merely mis-size one object - it poisons the whole
world on the next step", since the solver shares one state vector. The MuJoCo
backend has called it from `add_object` since object mass was hardened there; the
Newton and Isaac backends inherited it and never called it. 15 of 18 measured
cells diverged from the domain the base class defines.

Newton stored the value verbatim, so `nan` / `inf` / `True` reached
`builder.add_body(mass=...)` unchanged. A negative mass silently took the
`obj.mass <= 0` static path, returning an immovable body on a value only `0` is
documented to mean. A non-number was stored and then raised
`TypeError: '<=' not supported between instances of 'str' and 'int'` out of both
readers of that comparison - the solver rebuild and `list_objects` - so one bad
`add_object` left an already-registered object that made a later, unrelated scene
query raise.

Isaac read the value exactly once, at `float(mass)` while assembling the result -
after the prim was constructed, added to the scene, appended to the cleanup
registry and entered in `_objects`. `mass="heavy"`, `[0.1]` and `None` therefore
raised past the envelope `add_object` documents as its only failure channel with
the object already on the stage, and retrying under the same name with a usable
mass was refused as a duplicate: the name was permanently taken. The check now
precedes prim construction, so a refused mass constructs nothing and leaves the
name reusable.

Isaac's success log line formatted the caller's raw `mass` with `%.3f` while the
result reported the resolved one. For a static object the raw value is documented
as ignored and is never coerced, so a non-numeric mass made the logging call raise
`TypeError: must be real number` - latent, since it only fires when INFO logging
is enabled. It now logs the resolved value, which is always a float and is the
number the result reports, so the two cannot disagree.

One documented difference is preserved and pinned: Newton documents `mass=0` as
an alternative spelling of `is_static=True`, so a zero mass is a mode rather than
a small mass and is not validated as a dynamic one. Isaac documents no such
spelling and converges on the MuJoCo contract, refusing `0` with `is_static=True`
named as the remedy. A static object's mass is read by nobody on any backend, so
it is not validated there either.
