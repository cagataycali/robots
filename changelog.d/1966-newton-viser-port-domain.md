### Fixed: a viser dashboard port that cannot address a port is refused where it is named

`NewtonSimEngine.open_viewer(viewer="viser", port=...)` accepted any value,
forwarded it verbatim into `ViewerViser(port=...)`, and interpolated it into the
dashboard URL it reports back - so `port=0`, `-1`, `65536`, `10**9`, `8080.5`,
`nan`, `inf`, `True`, `'8080'`, `None` and `[8080]` each came back as
`status="success"` advertising an address such as `http://localhost:nan` or
`http://localhost:[8080]` for the caller to browse.

Two things followed. The engine holds a single viewer slot, so a value that did
not raise inside `ViewerViser` filled it, and the obvious recovery - calling
`open_viewer` again with a usable port - was then refused as
`"Viewer already open"`. And a value that did raise was reported as
`"Viewer launch failed: <exc>"` from the surrounding catch-all, which implicates
the viewer rather than the port the caller got wrong.

The port is now validated through the shared `utils.tcp_port_error` domain on the
`"viser"` branch, before the lock and before any viewer is constructed, so a
refusal builds nothing and leaves the slot reusable. The `"gl"` window and the
`"null"` sink bind nothing and continue to ignore the port, matching how the
policy providers validate one only when they dial it. `0` is refused rather than
read as an ephemeral-bind request, because unlike `PolicyServer` - which accepts
`0` and reads the assigned port back onto `.port` - this surface advertises the
port that was requested.
