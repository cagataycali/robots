### Fixed: a runtime property setter's value survives the next scene recompile

`world._model` is derived state - every scene mutation recompiles the scene spec
over it - but the MuJoCo runtime setters wrote only the model. Any later
`add_object` / `add_camera` / `add_robot` therefore discarded the value the
setter had already reported as applied:

```python
sim.set_body_properties(body_name="crate", mass=5.0)          # success
sim.set_geom_properties(geom_name="crate", friction=[0.2, 0.1, 0.0005])
sim.set_gravity(gravity=[0, 0, -1.62])                        # success

sim.add_object(name="marker", shape="sphere", size=[0.03])    # unrelated
# before: mass 0.1 kg, friction [1, 0.5, 0.001], gravity -9.81 - all restored
```

All six settable properties reverted: a body's `mass`, a geom's `color`,
`friction` and `size`, and the world's `gravity` and `timestep`. "Ephemeral" was
not a contract a caller could program against, because none of the methods that
recompile documents that it does so - the moment a value disappeared was
unpredictable from the outside, and a scene configured for lunar gravity quietly
went back to 9.81 m/s^2 mid-experiment.

Each setter now records its value in the spec as well as the model, so the
durable value is the one it reported. The compiled entity id indexes the spec's
element list directly, which is what lets an unnamed geom be reached at all -
most geoms in a robot scene carry no name - and the recorded id is verified
rather than assumed, because writing a size or friction onto the wrong geom is
worse than not recording it. A change that cannot be recorded is refused before
either representation is touched, rather than reported as applied.

A mass change is recorded as the uniform density change `set_body_properties`
documents: mass and inertia are both linear in density at fixed geometry, so one
ratio reproduces exactly the inertial the setter reported. Which element carries
it depends on how the body derives that inertial - a body declaring an explicit
`<inertial>` (every menagerie robot link) carries its own mass and inertia, while
a body integrating both from its geoms needs the geoms' mass or density moved
instead, since assigning `mass` on such a body element is silently ignored by the
compiler. A body with no mass of its own - the world body, which declares no
inertial and owns no geom - has nothing to scale and is now refused rather than
accepting a write that could never take effect.

Out of scope: `randomize` applies multiplicative domain randomization across
every geom and body, so making its writes durable would compound them into the
persistent scene on each call - a different contract that needs its own decision.
`remove_robot` replaces the spec with one rebuilt from the object/robot/camera
registry, which by construction drops every spec-level edit, not just these.
