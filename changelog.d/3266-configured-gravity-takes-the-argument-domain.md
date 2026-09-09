### Fixed: a configured Isaac gravity is honoured or refused, never silently reduced

`IsaacSimulation.create_world` has two owners for one value - its own `gravity=`
argument and `IsaacConfig.gravity`, the field it falls back to - and only the
argument was validated. The field was read straight into
`PhysicsContext.set_gravity`, which takes a signed scalar, so the z-component
was picked out of whatever the field held and applied. Every verdict therefore
depended on where the caller happened to spell the value.

`IsaacConfig(gravity=(0.0, -9.81, 0.0))` - the exact off-axis vector the
argument path documents refusing, because the physics context cannot aim
gravity off the Z axis - configured **zero gravity** while the result reported
`[0.0, -9.81, 0.0]` as if applied, so a world that was asked for lateral
gravity ran with none and said so nowhere. `nan` and `inf` reached
`set_gravity` unexamined; `("a", "b", "c")` reached it as the string `"c"`; a
four-component vector was read at index 2 and was also silently zero gravity;
and a two-component one raised `IndexError` past the method's
`{"status": "error"}` contract. The same values passed as `gravity=` were each
refused by name.

`create_world` now resolves config-or-argument into one effective value before
validating - the way `effective_timestep` immediately above it already does -
so both owners take the shared gravity domain (three finite, non-boolean
components) and this backend's Z-alignment constraint, and the source name is
passed to the domain so the message names the owner to fix
(`'IsaacConfig.gravity'` rather than `'gravity'`). With the value normalized on
every path, the downstream scalar-or-vector branches that produced the
index-2 read are gone.

Two consequences worth stating. A real scalar set on the field is now accepted
as the z-component, as `gravity=-9.81` always was, instead of raising a
`TypeError` about a float not being iterable - one domain, one answer. And
`gravity=None` remains the argument's spelling of "unstated, read the field",
while `IsaacConfig(gravity=None)` is a stated non-vector and is refused.
