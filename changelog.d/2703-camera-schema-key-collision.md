### Fixed: a scene whose cameras share one dataset column name is refused

A LeRobot feature name cannot contain `/` -- it addresses nested features -- so a camera's
namespace separator is recorded as `__`, and `arm0/wrist` is written into
`observation.images.arm0__wrist`. That collapse is not injective, so `arm0/wrist` and
`arm0__wrist` are two cameras in the scene and one column in the dataset. Nothing downstream
could say which of them the column was declared for, and the three ways of asking each failed
differently. Recording every camera declared the key twice and was refused by
`DatasetRecorder.create` as a repeated `camera_keys` entry, which is a true refusal but reads as
the caller having named one camera twice. `cameras=` naming both reported success and recorded
one: the two requested names are distinct, so they pass `name_list_error` and only their keys
collide, and the surviving column carried the other camera's frames -- or, when the two render at
different sizes, the FIRST `add_frame` was rejected and the episode was lost, after
`overwrite=True` had already wiped the dataset being replaced. `cameras=` naming one of them
succeeded, with the column's contents decided by the spelling used: the same
`observation.images.arm0__wrist` is the robot's wrist view when asked for as `arm0/wrist` and the
other camera when asked for as `arm0__wrist`.

`docs/recording.md` already promised the opposite -- "a column is never quietly populated from the
wrong camera" -- and that guarantee needs the scene's cameras to have distinct column names.

The ambiguity belongs to the scene rather than to one way of recording it, so it is refused once,
on the shared `camera_schema_key_collision_error` domain that every backend's `start_recording`
consults. The refusal names both cameras, the key they share and the rename that resolves it. It
runs after the dataset-stack probe, because reading the scene's cameras is an engine call and that
block is reachable on an install with no engine at all, and ahead of any session state or dataset
target, so a refusal leaves nothing set and nothing on disk. Scoping with `cameras=` is not
treated as a workaround: whichever of the pair won the column, the column would be named after the
other one.

The collapse itself now has a single owner, `strands_robots.utils.camera_schema_key`, called by
the three backends that declare a schema and by `DatasetRecorder.add_frame`, which renames an
observed frame onto its declared column. Those four have to agree on one mapping for a frame to
reach the column declared for it, and the guard has to agree with them or it grades a different
question. The mapping is idempotent, which is what lets a caller keep naming a camera in either
form.

Nothing changes for a scene whose camera names stay distinct after the collapse, which is every
scene that does not spell `__` in a camera name: a namespaced robot camera still records under its
collapsed key, and a blank name is still skipped rather than colliding with every other blank one.
