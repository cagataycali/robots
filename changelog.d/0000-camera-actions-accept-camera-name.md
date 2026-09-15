### Fixed: `add_camera` / `remove_camera` accept `camera_name`

`render`, `render_depth` and `get_camera_params` spell "which camera" as
`camera_name`; `add_camera` and `remove_camera` spell it `name`. An agent that
had just rendered from `camera_name="wrist"` and then sent
`remove_camera {"camera_name": "wrist"}` was refused with
`Unknown parameter 'camera_name'. Valid: ['name']`.

The dispatcher now accepts `camera_name` for `name` on an action whose name
says `camera` and whose method has `name` but no `camera_name` of its own -
the camera twin of the existing `name`/`robot_name` courtesy. `render`'s own
parameter is untouched, `name` remains the documented spelling, and
`camera_name` on a non-camera action is still unknown.
