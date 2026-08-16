### Fixed: a body-mounted camera no longer breaks `remove_robot`

`SpecBuilder.build` adds the scene's cameras but deliberately does not attach
robots, so a camera mounted on a robot body (`parent_body`, e.g. a wrist camera)
was added before its parent existed. The parent lookup failed and `ValueError`
escaped `remove_robot` -- a method documented to return a status dict -- so no
scene carrying a wrist camera could remove a robot at all, and the message told
the caller to "Pass the fully-qualified body name" for the very name
`add_camera` had already accepted.

Body-mounted cameras are now deferred and mounted by the new
`SpecBuilder.add_deferred_cameras` once every surviving robot is attached, so
such a camera keeps its parent body, its local pose and its tracking across the
rebuild. A camera mounted on the robot being removed has no mount point left, so
it is dropped from the registry with a warning -- the same treatment that
robot's own URDF cameras already get -- instead of aborting the removal.
