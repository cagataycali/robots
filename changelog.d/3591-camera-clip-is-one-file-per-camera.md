### Fixed: a namespaced camera's MP4 clip is a file in the output directory

`start_cameras_recording` / `start_cameras_recording_synchronous` named each clip
`<tag>__<camera>.mp4`, interpolating the camera name raw. A camera name may carry
one namespace scope -- `arm0/wrist` is how a wrist camera on a namespaced robot is
spelled, and `add_camera` accepts it -- and in a file name that separator is a
directory separator. The clip for `arm0/wrist` therefore landed at
`<output_dir>/<tag>__arm0/wrist.mp4`: one level below the directory the caller
named, in a directory nothing reported creating, under a name the recording tag
was missing from. The result's own two lines composed into a path that did not
exist, reporting `output_dir: <output_dir>` beside `-> wrist.mp4`.

The clip name now applies `camera_schema_key`, the same `/` -> `__` collapse that
names the dataset column recorded from that camera, so the clip is a file in
`output_dir` and is spelled like its `observation.images.*` feature. That mapping
is not injective, so `camera_clip_name_collision_error` refuses a recording whose
cameras would name one clip -- `arm0/wrist` and `arm0__wrist` -- before a frame is
captured, where `camera_schema_key_collision_error` already refuses the same pair
for the dataset sink. Without it the fix would have traded a misplaced clip for a
silently overwritten one.
