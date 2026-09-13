### Fixed: a declared camera every frame leaves empty is refused by name, not by LeRobot

`DatasetRecorder.add_frame` graded a frame against the schema in one direction
only. An observed camera the schema does not declare is dropped - that absorbs
LeRobot's `Extra features` branch, so an extra debug view cannot fail the write.
The mirror branch of the same LeRobot check, `Missing features`, is not
survivable: `validate_frame` rejects a frame that omits a declared feature, and
because camera names do not change between steps it rejects every frame of the
episode. Nothing is recorded.

The recorder treated that case as tolerable ("LeRobot tolerates absent columns
and the episode simply won't have that camera's data") and split its diagnostic
on the wrong condition - whether *any* observed stream matched. So declaring
three cameras and streaming `front`, `top`, `wrist_cam` against a declared
`wrist` recorded zero frames and logged nothing, and declaring none of the
streamed names logged that the dataset "will have no video" when in fact the
recording had already ended. Only the fake dataset the unit tests inject
tolerated the absent column; the real one never did.

A declared `observation.images.*` column with no image is now refused at
`add_frame` by `unrecordable_camera_columns_error`, the camera sibling of the
existing state and action column refusals, naming the empty column, the observed
stream that was dropped and the `camera_key_map` that reconciles them. The
refused set is exactly the set LeRobot already refuses, so no frame that used to
record stops recording: an extra camera alongside a full set of declared ones is
still dropped in silence.
