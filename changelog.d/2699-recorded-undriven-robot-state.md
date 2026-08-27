### Fixed: an undriven robot's declared `observation.state` columns are recorded as measurements

`start_recording` declares `observation.state` over EVERY robot in the scene, prefixing each
column with its robot's name (`alice__shoulder_pan`). A single-policy rollout's recording hook
supplies only the driven robot's observation, so every column belonging to another robot was
absent from the frame and took `DatasetRecorder.add_frame`'s `0.0` fill - in the same column,
with the same dtype, as a measurement, for every frame of the episode, under `status="success"`.

Nothing downstream can tell that zero from a reading. A policy trained on the dataset learns the
other robot is permanently at its zero pose, and anything replaying or analysing the episode
reads the same. The disagreement is unbounded: an undriven robot keeps whatever pose it was
placed in, and contact with the driven robot moves it further. In a two-`so101` scene where the
undriven arm sat at `[0.4653, -0.3447, 0.3548, 0.2475, 0.3476, 0.2970]` rad and reached
`1.5042` rad on its first joint during the episode, all six of its declared columns were `0.0`
on disk with zero span.

`run_multi_policy`'s synchronized loop already had this right - it builds one merged observation
per step covering every robot it drives. The three single-policy hooks did not, so which rollout
entry point recorded an episode decided whether its state columns were measurements. They now
resolve the columns they do not drive through one shared owner, `undriven_robot_state`, which
reads each robot's scalar state through the engine's own
`get_observation(robot_name=..., skip_images=True)` - the same call the driven robot's columns
come from, at the same step - and merges it under the hook's existing prefix, with the driven
robot's own reading winning any collision. Camera arrays are not collected: a frame is keyed by
its camera rather than by a robot, and cameras are scoped by `start_recording(cameras=...)`.

A bystander read the engine cannot serve is reported and skipped rather than ending the episode.
The driven robot's columns are the rollout's primary product, so losing a whole episode of them
to a bystander's read failure is strictly worse than the fill this replaces.

The matching *action* columns are deliberately unchanged. No command was issued to a robot this
rollout does not drive, so no value there is truthful, and settling it needs a schema decision
about whether two independently-driven robots may share one dataset at all. A state column has
no such ambiguity - the robot is in the scene and its joint positions are readable - so the two
are separable and only the answerable half is changed here.
