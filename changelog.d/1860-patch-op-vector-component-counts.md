### Fixed: `patch_scene_mjcf` holds a fixed-width op field to the library's component count

The patch ops write the same buffers `add_object`, `add_camera` and `move_object`
write, so a component count either surface refuses has to be refused by the
other. Only the components were checked; the count was left to MuJoCo, which
could not deliver that in either direction.

`set_body_pos` / `set_body_quat` assign the field as a spec attribute
(`body.pos = ...`) rather than passing it as a constructor keyword, and pybind11
reports a width mismatch there by dumping its C++ overload table and the
receiving object's address - naming neither the op nor the field, so the one
thing the caller needs is the one thing absent from it. The sibling ops writing
those same two fields through a keyword reported cleanly, so a wrong-length pose
had two very different answers depending on which op issued it.

In the other direction a three-component `rgba` was refused outright, though it
is the RGB `add_object(color=...)` accepts and completes with an opaque alpha:
one backend, two surfaces, one `geom_rgba` buffer, opposite verdicts on the same
colour.

Each numeric field now has a decided domain - `pos` exactly 3 finite components,
`quat` 4, `rgba` 3 (RGB, completed) or 4, `size` the count its shape consumes -
and a parity test pins that every field an op declares has one, so a field added
to an op cannot reach the compiled model undecided. A key that is present carries
a value, so an explicit `None` is refused rather than read as an omission asking
for the default.
