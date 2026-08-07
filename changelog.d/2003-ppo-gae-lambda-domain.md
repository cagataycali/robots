### Fixed: an unusable GAE-lambda is refused instead of letting the advantage trace diverge

`RLTrainSpec.lam` is the second factor of the advantage trace's decay: PPO's GAE
recursion carries it forward as `last_adv = delta + gamma * lam * (1 - done) *
last_adv`, so the trace decays by the *product* `gamma * lam`. The discount-factor
preflight bounds `gamma` to the closed interval [0, 1]; nothing bounded `lam`, so
the divergence that gate exists to refuse stayed reachable through the other
factor. With a `gamma` of `0.99` -- comfortably inside its accepted domain -- a
`lam` of `1.5` decays by `1.485`, and on this backend's own `compute_gae` over a
rollout of unit rewards the largest advantage grows from `235` at `T=12` to
`6.3e+16` at `T=96`: unbounded in the horizon rather than merely large, with the
run reporting success and writing a checkpoint. `lam=1e6` overflows it to `inf`.

`lam` outside the interval failed three further ways: far enough below zero the
trace diverges again (the decay is `|gamma * lam|`, so `lam=-2` reaches `1.0e+28`
by `T=96`) while merely below zero it collapses to the immediate reward and stops
accumulating future advantage at all; `nan`/`inf` poisoned every advantage and
surfaced only as a torch constraint error from the distribution sample, naming
neither the field nor the run, after the env, the networks and a full rollout had
been built; and `True` was a silent `lam` of one -- a Monte-Carlo estimator rather
than the bootstrapped trace requested -- because `bool` is an `int` subclass and a
bare comparison against the bounds accepts it.

`Trainer._gae_lambda_problems` now reports it, on the same closed-interval domain
the discount factor uses: both endpoints are legitimate and standard (`lam=1` is
the Monte-Carlo advantage, `lam=0` the one-step TD advantage), so the interval is
closed rather than a positivity test. It is a separate gate from the discount
factor's because only the on-policy backend estimates an advantage trace -- per
`TrainSpec` a backend must not report on a field it never reads -- and an AST
guard asserts every module that reads `spec.lam` routes through it, so a second
on-policy backend cannot ship without the domain.
