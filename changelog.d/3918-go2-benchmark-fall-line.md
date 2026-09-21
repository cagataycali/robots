### Fixed: the Go2 benchmarks score a folded stance as the fall it is

The three shipped Unitree Go2 locomotion specs (`go2_walk_forward`,
`go2_strafe_left`, `go2_turn_left`) terminated an episode on
`base_below_z(0.18)`, a line no collapse reaches: a Go2 whose legs fold under
gravity comes to rest at 0.203 m with the trunk still level, so `base_tipped`
stays quiet too and a dead robot was scored as a healthy full-horizon episode
(mock policy: 0 failures over 1000/1000 steps, reward 500.2). The 0.18 m line
was a biped-style "half the standing height" applied to a 0.32 m stance figure
that is not this asset's - its own `home` keyframe stands at 0.27 m. The fall
line is now grounded in the fold (0.22 m, 17 mm above it and 50 mm below the
stance) and the `base_height` regularizer in the keyframe (0.27 m), so the same
rollout is scored a failure at step 21 instead of step 1000.
