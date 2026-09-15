### Fixed: every camera name the simulation lists can be rendered through it

`list_cameras` documents its return value as the names `render` accepts, and it
offers two spellings for a robot's own camera: `add_robot` registers the cameras
a robot's MJCF declares under their short name (`wrist`) while the compiled model
holds them namespaced (`arm0/wrist`). Each camera surface resolved that name for
itself and they disagreed - `get_observation` published a frame under the short
key, `render` / `render_depth` / `get_frame` / `get_camera_params` refused it as
"Camera 'wrist' not found" by a message that listed `wrist` as available, and
`start_cameras_recording` accepted it, reported success and wrote a clip of zero
frames. One private seam now owns the mapping for all of them, so either
spelling addresses the camera on every surface, the way a body name may already
be bare or namespaced. Names that already resolved are unaffected: the model
lookup is tried first and the registry is consulted only for a name it refuses.
