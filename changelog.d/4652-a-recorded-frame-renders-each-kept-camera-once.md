### Fixed: a MuJoCo recording renders each kept camera once per frame, and none when it keeps none

`step()` under an open recording rendered every scene camera for every robot
on every recorded frame, then dropped the pixels a `start_recording(cameras=[])`
session never keeps; `run_multi_policy` rendered the cameras again for each robot
after the first. A frame now renders the scene's cameras once, from the first
robot's observation, and not at all for a session scoped to no camera. The
recorded datasets are unchanged.
