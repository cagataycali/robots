### Fixed: Isaac `randomize(randomize_colors=True)` changes the colour objects render in

`_randomize_colors` wrote each object's `displayColor` primvar, which RTX reads only
for a prim with no bound material - but `add_object(..., color=)` binds one
(`/World/Looks/visual_material`), so the result reported new colours while every
frame kept the old ones: a red cube reported recoloured to (0.18, 0.31, 0.82)
rendered (119, 37, 37) before and after on Isaac Sim 6.1. Each object now gets its
own `UsdPreviewSurface` (`/World/Looks/strands_randomized_<name>`) whose
`diffuseColor` is the sampled colour, bound stronger than its descendants; a later
randomization updates the same material, and recolouring one object no longer
depends on (or changes) a material another object shares. The same cube now renders
(43, 66, 127).
