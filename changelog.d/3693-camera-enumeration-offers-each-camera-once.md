### Fixed: a MuJoCo camera enumeration offers each camera once, not once per name

A robot camera answers to two names - `add_robot` registers the short alias
(`wrist`) while the compiled model holds it namespaced (`arm0/wrist`) - and
`_camera_id` resolves both to one camera. `_active_camera_list` enumerated a
scene by name without resolving identity, and its callers spend one unit of work
per name: `render_all` returned two pixel-identical frames of one camera under
two labels, and `start_cameras_recording` / its synchronous peer ran a second
encoder to write a second MP4 of the same view (a 3-camera scene produced 5
clips).

Two sibling guards in the same call path already refuse that outcome -
`camera_clip_name_collision_error` for two cameras naming one clip, and
`name_list_error` for a caller repeating a name, "because a repeated name opened
a second encoder on the one output path" - but neither could see this one, since
two spellings of one camera are two distinct names naming two distinct clips.
The enumeration now resolves identity through `_camera_id` and keeps the first
spelling of each camera, so a caller's own ordering and spelling survive and the
scene-wide list keeps the namespaced model name.
