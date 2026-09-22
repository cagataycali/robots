### Fixed: an embodiment child inherits its parent's unit frame

`_extends` in `embodiments.json` carried only the key mapping (`obs_rename`,
`state_keys`, `action_keys`, `dim_policy`) and dropped `state_units`,
`action_units`, `gripper_index`, `gripper_joint_range` and `joint_mids`, so a
child of an SO-arm parent packed `observation.state` and emitted actions in the
native frame while its parent declared `degrees` -- with no refusal. A child now
inherits every field the parent declares, read off `dataclasses.fields`, so a
field added later cannot silently fail to inherit. Keys the child declares still
win.
