### Fixed: the Isaac backend reads `add_camera(fov=)` as the vertical FOV, matching MuJoCo and Newton

`add_camera(fov=)` is documented as one shared surface across backends, but the
Isaac backend applied it to the *horizontal* axis (deriving the focal length
from the horizontal aperture) while MuJoCo and Newton pass it to MuJoCo's
`fovy` and `get_camera_params` falls back to `fovy`. The same call therefore
framed a different scene on Isaac than on the other two. Isaac now derives the
vertical aperture from the horizontal one and the image aspect ratio and maps
`fov` onto the focal length on the vertical axis, so the intrinsics come out
`fx == fy == height / (2*tan(fov/2))` -- exactly the square, vertical-FOV
intrinsics MuJoCo reports for the same call.

This changes framing on the Isaac backend: **every Isaac render and every
dataset recorded from an Isaac camera changes** (a `640x480` `fov=60` camera's
horizontal field widens from 60 deg to ~75 deg, and objects sit at different
pixel offsets). Re-record Isaac datasets whose camera framing matters, or pass
the horizontal angle you previously relied on converted to its vertical
equivalent (`fovy = 2*atan(tan(fovx/2) * height/width)`).
