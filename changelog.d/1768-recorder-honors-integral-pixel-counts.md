### Fixed: a plain-MP4 recording started at an integral-float pixel count now records

`start_cameras_recording(width=640.0)` returned `status="success"`, announced the
recording, and then captured nothing: `stop_cameras_recording` also reported
success with `0 frames  0.0 KB  (N errors)` and no MP4 on disk. The identical
call with `width=640` recorded normally, so the only difference between a
working recording and an empty one was a value the `start` call accepted.

The recorders validate their four frame/pixel-count options against the shared
`positive_whole_number_error` domain, which deliberately admits any real scalar
with an integral value so an `fps` read out of a config float or a resolution
probed off a camera as `np.int64` can be honored. `fps` and
`max_frames_per_camera` were honored - the capture loop only divides by and
compares against them - but `width`/`height` were forwarded verbatim to
`render`, whose `_validate_render_dims` requires a true `int` and therefore
refused every frame. Passing the guard is a promise the value *can* be honored,
not that it is already in the form its consumer needs.

Both plain-MP4 entry points (`start_cameras_recording` and
`start_cameras_recording_synchronous`) now normalize an accepted pixel count to
plain `int` before the capture loop uses it, so `640.0` and `np.int64(640)`
record at 640 pixels. This is the normalization every other pixel-count surface
in the library already performs after validating on the same domain -
`VideoConfig.from_dict`, which is why `run_policy(video={"width": 640.0})` wrote
a correct MP4 while the recorder sharing its domain wrote none; `HybridCompositor`,
whose comment states that "a np.int64 must not leak through"; and `mjpeg_frames`,
whose per-component check exists so `size` and "the recorders' `width`/`height`
cannot diverge".

The accepted domain itself is unchanged: the guard still runs first, so a
non-integral `12.5`, a non-positive `0`/`-64`, `nan`/`inf`, a `"64"` string and
`True` are refused exactly as before rather than silently truncated, and a
`width`/`height` of `None` keeps its "use the camera's own resolution" meaning.
