### Docs: the `Args:`-completeness guard now compares the whole package, and the four surfaces outside the simulation subtree that needed it

The guard that compares a signature against its own `Args:` block was rooted at
`strands_robots.simulation`, which is the subtree the six surfaces it was written
for happened to live in - nothing about the drift is simulation-specific. Rooted
at the package it compares 629 parameters over 172 surfaces instead of 345 over
82, and it reported four more surfaces accepting a parameter their docstring
never mentions (and none in the other direction, anywhere in the package).

`DatasetRecorder.create` now documents `camera_dims`, `video_width` and
`video_height` - the three parameters that decide the shape every camera column
is declared with, including that `camera_dims` is `(height, width)` while the
pair it falls back to on the same call is width-then-height. Both
`Cosmos3Policy.get_actions` and `LerobotLocalPolicy.get_actions` now say what
they do with `**kwargs`: Cosmos 3 forwards it on the `diffusers` backend and
drops it on `service`, and the LeRobot-local provider never reads it, honouring
only the `inference_kwargs` bound at construction. `LiberoAdapter.__init__` now
documents `state_gripper_joint_names`, whose length is the width of
`state.gripper` and is not arity-checked - unlike the `state_gripper_signs`
vector it has to agree with, which is, so a mismatch is warned about and the
signs silently skipped.
