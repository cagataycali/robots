### `add_robot("so101")` keeps the registry entry it loaded

The deprecated name-as-registry-key fallback resolved the model from the
instance name but recorded no `data_config` on the robot, so everything keyed
on it forgot which registry entry the model came from: `set_gripper` reported
"the registry carries no gripper metadata for this robot" for an entry that
has it, a recording declared `robot_type` from the instance name,
`list_robots_info` printed `Config: direct`. `Robot("so101")` passes
`data_config` and worked on the same scene, so the plain Python-API shape was
the one that failed. The robot's `data_config` now records the registry entry
its model was built from whichever argument named it; a `urdf_path` add still
records none, and the deprecation hint is unchanged.
