### Fixed: `remove_object` drops the cameras mounted on the body it removes

A camera added with `add_camera(parent_body=<object>)` expresses its pose in that
body's frame, so MuJoCo makes it a child element of the body and the recompile
that follows `remove_object` deletes it. The `world.cameras` registry did not
follow, so the call reported success while leaving an entry naming a camera the
renderer could no longer resolve: `list_cameras()` offered it, and `render` /
`get_camera_params` refused it with `Camera 'watch' not found. Available:
['default', 'fixed', 'plate_cam', 'watch']` -- naming the missing camera as an
available alternative to itself. `eject_body_from_scene` now drops those entries
with a warning naming the camera and the removed body, the same treatment
`eject_robot_from_scene` already gives a camera whose parent belonged to the
robot being removed ("stale entries would linger in the registry and confuse
observation code", in that path's own words). Cameras mounted on other bodies
and world-fixed cameras are untouched.
