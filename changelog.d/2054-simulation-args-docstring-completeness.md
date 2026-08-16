### Docs: document the simulation parameters that carry an enforced domain but no `Args:` entry

Six public simulation surfaces accepted parameters their docstring never
mentioned, so a caller could be *refused for a parameter they could not look
up*: `MuJoCoSimEngine.__init__`'s `ros2_domain` raises unless it is an `int` in
`[0, 232]`, `add_object`'s `material` is refused for any key outside
`MATERIAL_KEYS`, and `PolicyRunner.run` / `.evaluate`'s `control_substeps`
raises unless it is a positive integer - each reported a name with no entry to
read. `SimEngine.get_observation`'s `skip_images` was documented on the Newton
backend that implements the method but not on the ABC that declares it.

Every entry is now written from the authoritative source (`_init_ros_bridge`
for the ROS 2 knobs, `spec_builder.MATERIAL_KEYS` for the material vocabulary,
`_control_substeps` for the substep contract), and a new guard compares every
public simulation method's signature against its own `Args:` block in both
directions, so a parameter can no longer be added - or removed - without the
docs following. The guard honours combined entry labels (`width/height:`,
`width, height:`), which two recorder entry points already rely on.
