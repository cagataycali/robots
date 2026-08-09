### Fixed: a Newton `"gl"` viewer window size is on the same floor as a frame's

`NewtonSimEngine.open_viewer` sized its OpenGL window from `width` / `height`
without applying a domain, so a pixel count the same engine refuses for a frame
was forwarded verbatim into `ViewerGL`. Measured against a recording stand-in
for the viewer, 12 of 13 values were refused by `render(...)` and accepted here:
`0`, `-4`, `-1`, `2.7`, `640.0`, `True`, `False`, `"big"`, `"640"`, `nan`, `inf`
and `[640]`.

The consequence is the one the neighbouring `port` guard already states in its
own comment: the engine holds a single viewer slot, so
`open_viewer("gl", width=0, height=0)` returned `status="success"`, built a
zero-pixel window and filled the slot -- after which the obvious recovery,
calling `open_viewer` again with a usable size, was answered
`"Viewer already open (gl)."` under `status="success"` and built nothing. The
caller was left with a window they had not asked for and no way to replace it.

Both dimensions now go through `positive_count_error`, the floor `add_camera`
and the render family already share, applied on the `"gl"` branch alone (the
`"viser"` and `"null"` viewers are never handed a size, exactly as `port` is
checked only on the branch that binds it) and before the lock, so a refused size
constructs nothing and the slot stays free.
