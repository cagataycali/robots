### Fixed: a body declaring a placement that is not finite is refused, not measured around

`load_mjcf_scene_objects` composes a `SceneObject` from three placements: the top-level body's `pos`
becomes the object's `position`, its orientation becomes `quat`, and each nested body's `pos` becomes
the running offset every geom below it is measured from. `_geom_aabb` refuses a non-finite value on
all four of a *geom*'s parse paths. The *body* placements the same bound is composed from were read
straight through `_parse_xyz` / `_parse_orientation`, whose contract is to fall back to a documented
default on an unreadable attribute and to return whatever a readable one parsed to.

The nested case is the one nothing can detect. A non-finite offset makes every bound below it
non-finite, and the running `min`/`max` in `_recursive_collision_aabb` orders a NaN as neither
smaller nor larger than anything, so the whole subtree disappears from the union while the walk still
reports that it found analytic geometry:

    table, leg body pos="0 0 -0.37"  ->  size (0.8, 0.8, 0.76)
    table, leg body pos="0 0 nan"    ->  size (0.8, 0.8, 0.04)
    table with the leg deleted       ->  size (0.8, 0.8, 0.04)

A 4 cm slab where the file declares a 76 cm table, byte-identical to the same fixture with the leg
deleted, under a successful load - and with every field the loader reports finite, which is what
separates it from the geom spellings: a consumer cannot screen for it. The other placements fail more
loudly but no more correctly. A non-finite top-level `pos` or orientation is reported verbatim as the
object's own, and `pos="0 0 inf"` on a nested body reports an infinite extent, because `inf - inf` is
a NaN the outer accumulator drops in turn.

Reachability is wider here than for the geom spellings. That finding is scoped to fixtures, because a
non-finite geom makes the inertia MuJoCo derives non-finite and MuJoCo refuses a body with a free
joint. A body's own `pos` is not an input to that derivation, so MuJoCo compiles a non-finite body
placement with or without a free joint - movable task objects as well as the tables and cabinets
whose footprint a manipulation scene is planned against.

The disposition follows the sibling rather than inventing one: refuse, naming the body and the
attribute the file used. The finiteness test moves into `_refuse_non_finite_placement`, the single
owner the geom locator and the new body locator both delegate to, so neither can drift into
tolerating what the other refuses; the guard's own docstring already named this accumulator as the
step that drops a non-finite value, while the accumulator's own input went ungraded.
