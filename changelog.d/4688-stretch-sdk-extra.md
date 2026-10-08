### Fixed: `pip install 'strands-robots[stretch]'` installs the Stretch SDK

`[stretch]` was not a declared extra, so pip installed nothing and exited 0, and
the `StretchDriver` refusal named a different command
(`pip install hello-robot-stretch-body`) from the one every other native driver
with a PyPI SDK names. `[stretch]` now pulls `hello-robot-stretch-body>=0.7.0`,
and the refusal and the driver docs cite it. It is not a member of `[all]` (the
wheel pulls open3d, pyrealsense2 and jupyter) and cannot be installed next to
`[sim-mjlab]`: the Stretch tools pin trimesh below mjlab's floor.
