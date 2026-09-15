### Fixed: `MockPolicy` commands inside each actuator's ctrlrange, so hello world prints only its expected line

`examples/01_sim_hello_world.py` promised one line of output and printed four:
three engine warnings that the mock's ±0.5 rad sinusoid was `outside its
ctrlrange` for the SO-100 `Pitch` `[-3.32, 0.174]`, `Jaw` `[-0.174, 1.75]` and
`Elbow` `[-0.174, 3.14]`, then `run_policy status: success`. The same three
lines opened examples 03, 06, 07 and 17. The warning was right - MuJoCo clamped
the value, so the commanded trajectory was not the one reproduced - but the
policy it warned about is the one every first-run path uses.

`MockPolicy` now opts into the engine's existing `set_sim_context` hook (called
right after `set_robot_state_keys`) and reads the ctrlrange of every driven
ctrl-limited actuator, then clips its sinusoid to it. An unlimited actuator or
a key with no actuator keeps the full sinusoid; a context that is not a model
leaves the policy exactly as configured and never fails the rollout.
