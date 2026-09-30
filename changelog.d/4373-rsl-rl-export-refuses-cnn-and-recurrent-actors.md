### Fixed: exporting an Isaac Lab CNN or recurrent actor is refused, and `create_policy("rl")` no longer truncates an observation

`rsl_rl.convert_checkpoint` (behind the `isaaclab` trainer's `export`) kept the
`mlp.` layers of every rsl_rl actor and dropped the rest, so an Isaac-Cartpole-Camera
`CNNModel` exported as a 1,600-input MLP without its image encoder, and a recurrent
AnymalD `RNNModel` exported "OK" without its LSTM; both then deployed a network that
computes something else. It now reads `actor.class_name` from the run's
`params/agent.yaml` and the actor's key prefixes, and refuses anything but an
`MLPModel` (keys `mlp.`, `obs_normalizer.`, `distribution.`), naming the class and
the parts it would drop and pointing at `play()` or Isaac Lab's exported policy.
`RLCheckpointPolicy` read the first N values of any longer `policy_obs` vector - an
actor reading 4 values answered for 100,000 - and reported a short one as N
"missing" keys; a vector whose length is not the width the actor reads is now
refused with both numbers.
