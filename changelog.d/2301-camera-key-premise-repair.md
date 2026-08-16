### Fixed

Repaired the regression test whose premise the camera-ownership fix invalidated,
which had left `main` failing CI. `test_key_naming_no_compiled_camera_is_absent`
manufactured a camera key with no compiled camera by adding a camera-bearing
robot, adding a second robot, and removing the first -- a sequence that stranded
the entry only because `add_robot` registered one robot's cameras against
another. `SimCamera` now pins that `origin_robot` never names a robot outside the
camera's namespace, so removing a robot removes its cameras and only its cameras
and nothing is stranded. The contract under test is unchanged: a registry key the
compiled model cannot answer for yields no image rather than the free camera's
overview, so the entry is now registered directly instead of routed through a
path the registry contract forbids, and the answerable keys are asserted to still
report so an omitted key cannot be mistaken for a broken render.
