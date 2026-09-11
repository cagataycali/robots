### Fixed: the Newton backend's scene listings report where things are

`list_robots_info()` and `list_objects()` reported `SimRobot.position` /
`SimObject.position` -- the vector `add_robot` / `add_object` was *asked* for,
which the solver never writes back. Both now read the pose out of the live
`body_q` state under the engine lock, matching what the MuJoCo backend's
listings already report. A robot with several root bodies has no one base pose
to measure, so its line reports the requested transform and is labelled as
such; a static object keeps its record, which is where it is, because Newton
bakes a static shape into the world instead of giving it a body.
