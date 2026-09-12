### Fixed: an evaluation measured with success_fn reports its reliability figure too

`pass_hat_k` shipped on one of `PolicyRunner.evaluate`'s two routes. `evaluate`
delegates to `_evaluate_with_spec` when given a `spec` and otherwise runs the
`success_fn` route itself, and only the first payload carried the key - so
`eval_policy(success_fn=...)`, the public evaluation surface, reported
`success_rate=0.4, n_success=2, episodes_completed=5` and no reliability figure
at all. That is the gap the field was added to close ("two checkpoints at the
same mean, one consistent and one erratic, were indistinguishable in the
payload"), still open on the route most callers reach.

Nothing was missing to compute it: `pass_hat_k` is `C(c, k) / C(n, k)` over
`n_success` and `episodes_completed`, both of which that route already reports.
Unlike the reward fields beside it - `max_step_reward` and
`avg_max_step_reward`, which need the dense-reward term only a `spec` supplies,
and whose paragraph in `evaluate`'s `Returns:` is scoped "when `spec` is used" -
the reliability figure follows the success criterion rather than the reward
source. The `pass_hat_k` paragraph carried no such scope, so it read as a
promise on both routes and was one.

Absent rather than `0.0` when no success criterion was in force. Without one
every attempt counts as a failure for a reason unrelated to the policy, and a
row of zeros would read as measured unreliability instead of an unasked
question - the same rule that already makes `k` above `episodes_completed`
absent. The `spec` route is unaffected: a spec is a criterion, so it reports
`success_measured=True` unconditionally.

Pinned by `TestPassHatKIsReportedOnTheSuccessFnRoute` in
`tests/simulation/test_eval_reliability_and_peak_reward.py`, whose classes
exercised the `spec` route only. Four cells fail before the change; the fifth is
the control that the unmeasured case reports nothing, and it fires when the
`success_measured` gate is removed. The row is graded against the estimator
applied to the payload's own counts rather than against a literal, so it pins
the arithmetic and not the probe, and the split is asserted non-degenerate so an
all-zero or all-one row cannot pass.
