### Fixed: `add_object(shape="cube")` and `shape="cuboid"` build a box on every backend

The README quickstart and the `add_object` docs call a box a cube, but only
Isaac accepted `"cuboid"` and no backend accepted `"cube"`. Both words now
build a `"box"` on MuJoCo, Isaac, Newton and mjlab through one shared table
(`strands_robots.simulation.models.SHAPE_ALIASES`), and the object is listed as
a box. mjlab's refusal for an unknown shape now also names the closest one.
