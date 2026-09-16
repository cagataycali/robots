### Fixed: `load_scene` keeps an exported robot registered, and says what it dropped

`export_xml` → `load_scene` (the documented round trip) used to leave the arm
in the scene but unregistered: robot-scoped actions refused with "No robots
registered" and `add_robot` under the same name collided in MuJoCo, reported
only as "Failed to inject robot into scene". `load_scene` now carries every
registered robot whose namespaced joints are all in the loaded model (ids
re-resolved), names anything the swap did discard (`REPLACED the live world:
dropped robot(s) […]`) with the `add_robot(name=…, data_config=…)` /
`add_object` / `add_camera` calls that put it back, and returns a json block
of carried / dropped names. A refused robot attach now reports MuJoCo's
reason (with a hint when a subtree of that name is already in the scene).
