### Fixed: `add_object` / `add_camera` / `add_robot` refuse a name that cannot address the entity they create

These three methods claim a name twice - as the world-registry key and as the MJCF
element name - and three kinds of value silently broke the link between those layers.

A non-`str` was hashable often enough to be registered (`7` and `True` both are) and
only then reached the spec build, where pybind11 raised
`TypeError: add_body(): incompatible function arguments`. That escaped the result dict
these methods document, and it landed *after* the registry write, so the world was left
holding a key for an entity with no body in the model.

An empty name is MuJoCo's own sentinel for an unnamed element, so the entity compiled
anonymously and nothing could address it afterwards: `get_body_state(body_name="")`
reported the body as missing while it simulated, `render(camera_name="")` routed to the
free camera by an explicit token check - handing back an image from a camera the caller
never placed - and the recording schema dropped the anonymous camera rather than declare
an empty feature key, so a two-camera scene silently recorded one.

A name containing a NUL left the two layers disagreeing: MuJoCo compares names only up
to the NUL, so `"a\0b"` compiled as `"a"` and answered to that, while the registry kept
the full string.

All three are now refused through the normal error result, before the registry write, by
a shared `strands_robots.utils.entity_name_error` so the creators' accepted domain cannot
diverge. Nothing further is constrained, because nothing further was broken - `"a/b"`,
`"a b"` and `"a-b"` each compile under the name given and remain accepted. `add_robot`
still derives a label from the model when no name is supplied, exactly as documented.
