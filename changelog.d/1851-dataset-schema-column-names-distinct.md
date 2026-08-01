### Fixed: `DatasetRecorder.create` refuses schema column names it cannot honor

`camera_keys`, `joint_names` and `action_names` declare the recorded dataset's
column names, and neither way of getting that wrong was reported. A single name
passed as a bare string is iterable per character, so `joint_names="gripper"`
declared seven columns (`g`, `r`, `i`, `p`, `p`, `e`, `r`) and every one recorded
`0.0` for the whole episode - `add_frame` reads each declared name out of the
observation and none of those names is in it - while `create`, `add_frame`,
`save_episode` and `finalize` all succeeded. A repeated name collapsed where it
keys a dict (`camera_keys=["front", "front"]` declared one camera column for the
two requested) and doubled where it indexes a position
(`joint_names=["j1", "j2", "j2"]` recorded `j2` twice and the joint the caller
meant not at all).

Each list is now refused on the shared name-list domain already applied by every
backend's `start_recording`, the plain-MP4 recorders and every provider's
`set_robot_state_keys`. The check runs before the lazy lerobot import, so the
same mistake reports identically with or without the dataset extra, and before
the on-disk target is touched, so a refused `overwrite=True` call leaves an
existing dataset intact instead of removing it first. `None` and `[]` still mean
"not supplied" and the schema is derived as before.
