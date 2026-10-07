### Fixed: `list_objects` returns a json listing on MuJoCo and Newton, as on Isaac

`list_objects` on the MuJoCo and Newton backends answered with a text block
only, while `list_bodies` on the same backend and `list_objects` on Isaac also
return a `{"json": ...}` block. It now returns the text block first (unchanged,
so `content[0]["text"]` readers keep working) and then
`{"json": {"objects": {name: {shape, is_static, mass, position}}}}`, with the
same live position the text reports. An empty scene returns
`{"json": {"objects": {}}}`, as Isaac does.
