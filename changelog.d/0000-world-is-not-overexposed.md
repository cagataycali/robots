### Changed: the default world is a checker ground under toned lighting, not white on white

`Simulation()` now builds a checker-textured ground and dims the headlight (diffuse 0.5, ambient 0.15), so a
camera sees edges and colour instead of a blown-out white arm on a white plane: the default scene's mean pixel
drops from 234 to 93 of 255 and gains a texture. `Simulation(ground=)` chooses the floor: `True`/`"checker"`
textured, `"plain"` a flat grey plane, `False` none; an unknown value is refused by name. `light_scale` now has
headroom in both directions.
