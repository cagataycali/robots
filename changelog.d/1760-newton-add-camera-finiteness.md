### Fixed: a Newton camera pose or field of view that is not finite is refused

The Newton backend's `add_camera` validated neither the contents of
`position`/`target` nor `fov`, so a camera configuration the MuJoCo backend
refuses was accepted here and surfaced far from the call that caused it.

`position`/`target` were coerced with a bare `float(v)`, which caught a
non-numeric element but passed a `nan`/`inf` component - and a `bool`, reading
`True` as the coordinate `1.0`. Nothing downstream caught it either: the
degenerate-orientation check compares `abs(pos[i] - tgt[i]) < 1e-9`, which is
`False` for a `nan`, and `_look_at_quat` then divides the view vector by a `nan`
norm, so `render`/`get_frame` returned a frame from an all-NaN camera quaternion
under `status="success"`.

`fov` was coerced by a bare `float(fov)` inside the lock, so a non-numeric value
raised a `ValueError` straight through the structured tool-result contract, and
`nan`/`inf`/`0`/`>=180` registered a degenerate camera under a success result:
`get_camera_params` derives the pinhole intrinsics
`0.5 * h / tan(radians(fov) / 2)`, which is `nan` for a `nan` fov and raises
`ZeroDivisionError` for `0`.

Both now use the shared domains MuJoCo's `add_camera` already applies. The pose
goes through `coerce_pose_vector`, whose contract is that a pose either backend
entry point refuses must be refused by the other. The fov interval moved out of
the MuJoCo method into a shared `camera_fov_error` beside `pose_vector_error`, so
the two backends call one definition and cannot drift apart.
