### Fixed: a scene object's rotational inertia is integrated from its shape

`add_object` declared a dynamic object's mass on the body together with a
hard-coded inertia diagonal (`[0.001, 0.001, 0.001]`). `body_mass` was
therefore correct and the object fell exactly as it should, so only its
*rotational* dynamics were wrong - and wrong by orders of magnitude in a
direction that flipped with size. For a 100 g cube the true diagonal is
`m/6 * a**2`: at 1 cm the constant is 600x too large, at 5 cm 24x too large,
and for a 30 cm 1 kg crate it is 15x too small. A single constant also cannot
represent an anisotropic body at all, so every cylinder, capsule and non-cubic
box lost the `Izz != Ixx` that decides which way it topples.

The mass is now declared on the geom, so MuJoCo's compiler integrates the
inertia tensor over the shape the caller asked for. Given the torque that
should turn it a quarter turn in one second, a 6 cm matchstick previously
rotated 0.6 degrees and a 40 cm plank spun 2448 degrees; both now turn 86
degrees.

Because the inertia now scales with the shape, MuJoCo's "mass and inertia of
moving bodies must be larger than mjMINVAL" invariant becomes shape-dependent:
a mass above `mjMINVAL` can still integrate to an inertia below it on a very
small geom. `add_object` keeps its numeric mass pre-check (reproducing the
per-shape bound would mean reimplementing the compiler's integration) and the
residual case now reports the compiler's own reason, which names both mass and
inertia, instead of the generic `"spec recompile refused."` that left the
actionable text in the log.
