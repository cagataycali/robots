### Fixed: `cameras=` is validated as a list of distinct camera names on every surface

Seven public methods accept a `cameras` subset - MuJoCo's `render_all`, its two
plain-MP4 recorders (`start_cameras_recording`,
`start_cameras_recording_synchronous`) and every backend's `start_recording` -
and none validated its shape, so one parameter had four failure modes and two of
them escaped the structured `{"status": "error"}` contract these methods
document. `cameras="wrist"` was read as five cameras, one per letter, and
reported as five unknown cameras rather than as one mis-typed parameter;
`cameras={"wrist": ...}` was accepted with its values silently discarded;
`cameras=[3]` and `cameras=3` raised a bare `TypeError` out of the rendering
surfaces and dead-ended in a generic "Dataset init failed" on the dataset ones.

A repeated name failed in opposite directions depending on which surface received
it: `render_all(["wrist", "wrist"])` returned two image blocks for one camera,
`start_cameras_recording` reported "2 camera(s)" and opened a second encoder on
the one output path - so the camera was rendered and appended twice per capture
tick, and `stop_cameras_recording` reported two artifacts (25 frames each) for a
single 25-frame file - while `start_recording` silently collapsed it and declared
one camera column for the two requested.

Every surface now resolves the value through
`strands_robots.utils.name_list_error`, the shared domain that already governed
the policy `image_keys` parameters, so a mistake reports the same way wherever it
lands and names the parameter. `cameras=None` keeps its "every camera" meaning
and an empty sequence keeps each surface's existing verdict.
