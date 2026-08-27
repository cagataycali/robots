### Fixed: a background's alignment numbers are checked instead of coerced

`PanoramaBackground` and `GsplatBackground` handed every scalar number a caller
supplies to a bare `float()`, which accepts `nan` and `inf`. A non-finite
`rotation_deg` built the rotation matrix every output ray is turned by, so each
world direction became `nan`, the equirectangular lookup sampled nothing, and
`render()` returned a uniformly black backdrop -- `rgb.mean() == 0.0` with a
single distinct value -- while reporting no error at all. That last part is what
made it costly: in app contexts the photoreal background sits inside a catch-all
that demotes it to a procedural fallback, and a silent black frame never raises,
so the fallback could not fire.

The gsplat numbers feed the fitted `world_from_gs`. A non-finite `up_sign`,
`yaw_deg`, `radius`, `floor_z` or `backdrop_radius` produced a 4x4 with
non-finite cells, so every gaussian was placed nowhere; `min_opacity` was worse
than nowhere, because `nan > 0` is `False` and that skips the opacity filter the
value asks to apply.

The posture flags beside these were already closed on `boolean_flag_error`, a
domain documented as being for a flag that selects a posture rather than scaling
a quantity. The quantities were the other half, and they now take
`finite_number_error`, the shared signed domain for a physical quantity a caller
supplies verbatim -- the same rule `_shadow_plane_z_error` states one module
over for a plane height. `up_sign` and `clip_below` carry a documented `None`
sentinel, so the domain applies to the number and the sentinel passes through.

Bounds are deliberately not imposed: whether `min_opacity` belongs in `[0, 1]`
and `floor_pct` in `[0, 100]` is a separate question, and `numpy.percentile`
already refuses a non-finite `floor_pct` on its own.
