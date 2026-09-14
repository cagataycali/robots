### Fixed: an evaluation reports the episodes whose success predated the policy

Both evaluation routes sample the success criterion only AFTER an applied action,
so an episode whose criterion already holds at reset succeeds on its first step
whatever the policy commands, and `success_rate` / `pass_hat_k` report a hard
`1.0` for it. Nothing said so. `uncommanded_eval_error` already refuses the other
route to the same harm - there the policy never commands, here the criterion never
needed it - and the `success_measured=False` guard already warns that a *missing*
criterion reports a hard `0.0` "regardless of what the policy does". The
`1.0` direction was unguarded.

It needs no unusual spec: the threshold only has to sit on the wrong side of the
initial state, which is what a lift predicate does whenever the object rests on a
support and the height is measured from the floor. Measured on an SO-101 pedestal
scene, `body_above_z(cube, z=0.05)` against a cube already resting at `0.0549` m
reported `success_rate 1.0` and `pass_hat_k` `1.0` through `k=3` over three
episodes for a policy commanding its own current pose - the cube moved `0.78` mm,
the rest-jitter floor, and each episode ended after one control step of a
sixty-step budget. A second policy differing by 1160x in commanded base travel
produced a byte-identical report.

`eval_policy` and `evaluate_benchmark` now sample the criterion once per episode
at reset and report `episodes_successful_at_reset` (int) beside
`reset_success_warning` (the qualifying text, `None` when the count is zero), with
`success_at_reset` on each per-episode record. Every reported figure is left as
measured and nothing is refused: domain randomisation draws initial states per
episode, so a partial count is a fact about those draws rather than a broken spec,
and the count is what distinguishes them. That is the posture `_warn_unresolved`
states for a criterion that degenerates to a constant - surface the corruption
without changing a returned value.
