### Fixed: an Isaac camera pose or field of view that is not finite is refused

`IsaacSimulation.add_camera` validated none of its numeric inputs, so a camera
configuration the MuJoCo and Newton backends refuse was accepted here and
surfaced far from the call that caused it -- or not at all.

`position`/`target` were copied with a bare `list(position)`, which reads no
element, so a `nan`/`inf` component, a non-numeric element, a `bool` (read as the
coordinate `1.0`), a wrong-length vector and an empty one were all registered
under `status="success"` and handed to the USD camera prim and
`set_camera_view`. A NumPy pose -- the natural product of pose arithmetic -- also
leaked `np.float64` into the agent-visible status text and the `json` payload.

`fov` was coerced with a bare `float(fov)` from outside the method's try block,
so a non-numeric value raised a `ValueError` straight through the structured
tool-result contract, and `nan`/`inf`/`0`/`>= 180` registered a camera the RTX
pipeline cannot use: the pinhole relation
`focal_length = horizontal_aperture / (2 * tan(radians(fov) / 2))` raises
`ZeroDivisionError` for `0` -- a type absent from that try block's except tuple,
so it escaped the tool call -- and yields `nan` for a `nan` fov and 7.3e-16 mm
for `180`, both of which `set_focal_length` accepts.

The pose now goes through the shared `coerce_pose_vector` and the field of view
through `camera_fov_error`, the domains the other two backends' `add_camera`
already apply, with the identical-eye-and-target refusal they already carry. A
camera configuration one backend refuses is now refused by all three.
