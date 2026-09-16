### Fixed: `load_scene` keeps an exported robot registered, and says what it dropped

`load_scene` replaces the live world, so every registered robot, object and
camera was discarded silently - the result said only "Scene loaded from
table.xml / Bodies: 3" and the loss surfaced two calls later as "No robots
registered in the simulation". `export_xml` -> `load_scene` (the documented
round trip) was worse than silent: the arm stayed in the scene but unregistered,
robot-scoped actions refused, and `add_robot` under the same name was refused as
a bare "Failed to inject robot into scene." with MuJoCo's "repeated name" reason
left in the log - a dead end.

`load_scene` now carries over every registered robot whose namespaced joints are
all in the loaded model (ids re-resolved, mesh back-reference re-pointed at the
new world), and names anything it did discard - `REPLACED the live world: dropped
robot(s) [...]` - with the `add_robot(name=..., data_config=...)` / `add_object` /
`add_camera` calls that put it back, plus a json block of the carried and dropped
names. An object or camera the loaded file still carries under the same name is
reported as untracked-but-present rather than given an `add_object` that MuJoCo
refuses, and the "No robots registered" warning is only given when a robot was
actually dropped. A refused robot attach now reports MuJoCo's own reason, with a
hint when a subtree of that name is already in the scene.
