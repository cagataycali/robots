### Fixed: the Isaac move_to GPU test expects the frame move_to solves for

The GPU `move_to` suite registered a separate MJCF as the IK model, which `move_to` now ignores in favour of the loaded URDF, and asserted that MJCF's site as the end-effector frame. It now asserts the description's own frame (`jaw_link`).
