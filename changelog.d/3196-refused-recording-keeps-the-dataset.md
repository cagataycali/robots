### Fixed: a refused `start_recording` keeps the dataset it was replacing

`overwrite=True` is the one posture that deletes a real LeRobotDataset without
asking (`DatasetRecordingMixin._prepare_dataset_target`), and it is what the
`run_policy` agent tool records with. Every camera-shaped refusal
`start_recording` makes was already ahead of that deletion -- the `cameras=`
name-list domain, the boolean postures, the fps domain, and the scene-level key
collision, whose guard is documented as running "before any dataset is created,
resumed or wiped".

One was not. A single unknown name in `cameras=` was refused from inside the
schema-declaration block, roughly a hundred lines after the deletion, so the
call reported an error having already removed the dataset it was replacing.
Measured against a real single-episode recording:

| | `total_episodes` | `total_frames` | MP4s |
|---|---|---|---|
| after recording | 1 | 20 | 1 |
| after `cameras=["camera1", "camrea2"], overwrite=True` | -- | -- | -- |

The directory was gone. The refusal itself is right, and its remedy is what
makes the ordering costly: "Add them with `add_camera(...)` before recording, or
omit `cameras=` to record all of them" asks for a retry against the data the
same call destroyed. All three backends that declare a dataset schema carried
it, and `run_policy` cannot pre-flight the name on the caller's behalf -- the
available cameras are known only to the engine, which is why the tool forwards
`dataset_cameras` verbatim to a call that hardcodes `overwrite=True`.

The hazard was already written down for the sibling failure mode:
`camera_schema_key_collision_error`'s docstring notes that a colliding pair
surfaces "as the FIRST `add_frame` being rejected, after `overwrite=True` has
already wiped the dataset being replaced", and that guard was placed ahead of
the deletion for exactly that reason. This is the same ordering, applied to the
last refusal that still followed it.

Nothing between resolving the target and building the recorder reads or writes
the dataset directory -- the schema is read from the scene -- so the deletion is
deferred to the last statement before the recorder is built. That is a move, not
a rewrite: the executable statement count of all three backends is unchanged
(136 / 176 / 194 before and after), and the four documented outcomes still
happen, just later. `overwrite=True` still replaces an existing dataset rather
than appending to it, `overwrite=False` still resumes, a pre-existing empty root
is still cleared, and a non-empty non-dataset directory is still reported rather
than clobbered.

The regression is graded two ways, because the guard was correct and only
reached too late. A behavioural cell records a real episode, drives the refused
call, and reopens the dataset through `LeRobotDataset` to assert its frames,
metadata and per-camera video survived; a structural cell derived from the tree
asserts no backend refuses between the deletion and the recorder, so a backend
or a refusal added later is graded on arrival. `overwrite=False` is a passing
control that locates the damage in the destructive posture specifically, and
deleting the deletion outright fails the "still replaces" cell -- the two ways
this could be over-corrected.
