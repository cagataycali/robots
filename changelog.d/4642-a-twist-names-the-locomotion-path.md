### Fixed: a velocity twist sent to a simulated robot names where a twist goes

`send_action({"vx": 0.1, "vyaw": 0.0})` on a simulated robot is still refused,
because a world is commanded by joint. The refusal now adds one sentence when
every refused key is a twist key (`vx`, `vy`, `vyaw`): an intent-level driver
takes it on real hardware (`mode='real'`), and in sim a locomotion policy takes
it as `run_policy(..., policy_kwargs={'target_velocity': [vx, vy, vyaw]})`.
MuJoCo, Newton and Isaac share the sentence. A refusal naming any other key is
unchanged.
