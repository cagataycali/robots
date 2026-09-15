### Fixed: `add_camera` / `remove_camera` accept `camera_name`

`render`, `render_depth` and `get_camera_params` spell "which camera" as
`camera_name`; `add_camera` and `remove_camera` spell it `name`. An agent that
had just rendered from `camera_name="wrist"` and then sent
`remove_camera {"camera_name": "wrist"}` was refused with
`Unknown parameter 'camera_name'. Valid: ['name']`.

The dispatcher now accepts `camera_name` for `name` on a camera action whose
`name` IS the camera - one that names cameras through no other parameter - the
camera twin of the existing `name`/`robot_name` courtesy. `render` spells it
`camera_name` and `start_cameras_recording` spells it `cameras`, so neither is
rewritten; that action's `name` is the output filename tag, and a camera bound
into it recorded every camera in the scene under that tag. `name` remains the
documented spelling, and `camera_name` on a non-camera action is still
unknown.
