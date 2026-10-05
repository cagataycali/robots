### Fixed: the sim tool description names the objects and cameras already in the scene

`Robot("so100")` followed by `add_object(name="red_cube", ...)` and
`add_camera(name="front", ...)` - the README quickstart - left the tool
description naming only the robot and its joints, so an agent asked to pick up
the red cube spent its first call on `list_objects` to learn the cube existed.
The MuJoCo and Isaac descriptions now add one sentence, "The scene also holds 1
object(s) 'red_cube' and 1 camera(s) 'front'; ...", built by
`strands_robots.simulation.base.scene_contents_sentence`. The free `default`
camera every session has is not named, and an empty scene adds nothing.
