### Fixed: a composed model's keyframes carry the robot's namespace

A robot model that pulls another model in with MJCF `<attach>` reaches the scene
attach carrying the inner model's keyframes as MuJoCo *pending* keyframes -
absent from `spec.keys` until a compile materialises them - so
`spec.attach(child, prefix="<robot>/")` cannot rename them, and MuJoCo warns
that they "will not be namespaced correctly".

They then landed in the world under their bare source names. That cost a caller
a second instance of that robot outright: the bare name repeats, so `add_robot`
was refused with "repeated name 'home' in key" (measured on the registry's
LeKiwi, which mounts the SO-ARM100 arm on its base and inherits its `home` and
`rest` - so a two-LeKiwi fleet could not be built), and it left the name every
other robot's keyframes carry, `<robot>/<key>`, absent from the model.

`add_robot` now renames those keys after the attach's recompile, the first
moment they are reachable, and only the keys that attach introduced -
keyframes the scene already held keep their names. MuJoCo's warning about the
namespacing this repairs joins the attach-noise roster, which both suppression
layers now read.
