### Fixed: a benchmark reports the episodes whose failure predated the policy

`evaluate_benchmark` samples a spec's `failure` clause only AFTER an applied
action, so one that already holds at reset ends every episode on its first step
whatever the policy commands, and `success_rate` reports a hard `0.0` beside
`success_measured: true` - the same number an honest policy failure reports,
reached without the policy. Nothing said so. This is the third route to that hard
`0.0` the module already reasons about (the others are a missing criterion and a
`success` clause pinned to a constant, both already warned about) and the only one
reachable through a clause that is entirely well-formed, so no compile-time or
name-resolution check can see it: whether it holds is a fact about the initial
state.

It costs more than the mirrored `success` case, because the eval loop reads
`is_failure` BEFORE `is_success`: the episode is scored a failure with the success
criterion never consulted, so a success the run had already reached on that step is
discarded. Measured on an SO-101 pedestal scene, one policy and one scene scored
`success_rate 1.0` with `pass_hat_k` `1.0` through `k=3` over nine control steps
per episode, and `0.0` with `pass_hat_k` `0.0` after one step per episode, the only
difference being a `body_below_z(cube)` failure threshold moved from `0.01` m to
`0.10` m against a cube already resting at `0.055` m. All four existing guards read
clean in both runs (`success_measured: true`, `uncommanded_error: null`,
`episodes_successful_at_reset: 0`, `reset_success_warning: null`), and the run
scoring `0.0` wrote a one-frame rollout video. A "the object fell" height above
where the object already rests does it, as does a `base_below_z` collapse line
above the robot's spawned stance - which is why the shipped humanoid benchmarks
tune that line to each biped's own measured standing height.

`evaluate_benchmark` now samples the failure clause at the same pre-episode probe
the success clause already uses and reports `episodes_failed_at_reset` (int) beside
`reset_failure_warning` (the qualifying text, `None` when the count is zero), with
`failure_at_reset` on each per-episode record. Every reported figure is left as
measured and nothing is refused, the posture `episodes_successful_at_reset` already
takes: domain randomisation draws initial states per episode, so a partial count is
a fact about those draws rather than a broken spec. `eval_policy` takes a
`success_fn` and has no failure criterion to sample, so it reports neither.
