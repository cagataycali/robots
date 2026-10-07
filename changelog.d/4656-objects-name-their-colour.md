### Fixed: an object's colour reaches the agent, not only the renderer

`add_object(color=...)` used to change only the rendered pixels. The tool
description, the `add_object` reply, `list_objects` and `get_body_state` named
an object by its identifier alone, so "pick up the red one" could only resolve
when the name itself said red. Each now names the colour the object renders
with (`'cube_a' (red)`, `- cube_a: box at [...], 0.1kg, red`), and
`get_body_state` adds a `color` field to its JSON. The colour is read off the
compiled model, so a recolour by `set_geom_properties` or `randomize` is what
is reported. Newton's `list_objects` names the colour too.
