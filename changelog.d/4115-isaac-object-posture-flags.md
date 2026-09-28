### Fixed: Isaac posture flags are checked, not read by truthiness

`IsaacSimulation.set_object_kinematic`, `set_object_collision` and
`create_world(ground_plane=)` read a posture flag by truthiness. Every non-empty
string is truthy, so `"false"`/`"no"`/`"off"`/`"0"` selected the *ON* posture
while spelling its refusal - `set_object_kinematic("drop", "false")` pinned the
body kinematic and `create_world(ground_plane="false")` added the plane - and
`None`/`0`/`[]` took the other branch without being a declared spelling. Each
flag now passes through `boolean_flag_error`, the shared boolean domain the other
posture-flag surfaces already use, and a non-boolean is refused before the write.
