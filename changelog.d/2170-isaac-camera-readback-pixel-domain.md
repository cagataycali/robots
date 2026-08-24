### Quality: drive the pixel-count floor on Isaac's two camera readback surfaces

`positive_count_error` is the shared pixel floor for `width`/`height`, and its
docstring names the whole family it backs - `add_camera` plus the render family
(`render`, `get_frame`, `get_camera_params`) on every simulation backend - with
the invariant that the same camera configuration cannot be refused on one
backend and accepted on another. Four Isaac methods apply it; only two were ever
driven. No test called `IsaacSimulation.get_frame` or `get_camera_params` at
all, so on the two raw in-process readback surfaces neither the refusal nor the
success path had ever run: a regression that kept the guard call and dropped the
`raise` would have satisfied the existing structural sweep.

`get_frame`'s `Raises:` entry also named only one of its two `ValueError`
causes - the native-resolution mismatch - while its sibling `get_camera_params`
named both, so a caller reading the narrower docstring could not discover the
floor its code enforces. Both `Args:` entries now state it as well.

The new module drives both surfaces over the shared probe set and asserts each
refusal is the shared domain's verdict verbatim, pins that the guard precedes
every RTX handle read, keeps the Isaac-specific native-resolution check as the
over-reach control, and covers the values the readbacks exist to return - the
`(rgb, depth)` shapes and dtypes and the documented USD-prim-to-OpenGL basis
correction. A structural check requires any future public surface taking
`width`/`height` to apply the floor or forward the value to a sibling that does.
No Isaac Sim runtime or GPU is needed: the guard runs before the handle is read
and the handle itself is duck-typed.
