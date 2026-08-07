### Fixed: an RL discount factor the return cannot be discounted by is refused

`RLTrainSpec.gamma` weights every future reward in the return an RL algorithm
optimizes, and it is the one coefficient both backends read - PPO discounts the
GAE recursion with it (single-env and vectorized paths), FastSAC its target-Q
bootstrap. Neither preflight bounded it, and the arithmetic that consumes it
never judges it. A discounted return is a geometric series, so measured on the
real `compute_gae` over a 24-step rollout of unit rewards where the honored
`gamma=0.99` gives a largest advantage of 12.9: `gamma=1.5` gives 1.2e4 and
`gamma=5` gives 4.6e15, and doubling the horizon compounds again - divergence,
not merely a large number. A real PPO run with `gamma=1.5` returned
`status="success"` and wrote a loadable checkpoint. `gamma=-0.5` alternates the
sign of each future reward and collapses the largest advantage to the immediate
reward, 1.0. `gamma=nan` surfaced only once the update sampled the action
distribution, as `ValueError: Expected parameter loc ... of distribution Normal
... to satisfy the constraint Real()` - a torch message naming neither the field
nor the run, raised after the env, the networks and a full rollout were built,
which is the deep stack trace a read-only preflight exists to replace. `True`
was a silent `gamma` of one, because `bool` is an `int` subclass and a bare
comparison against the bounds accepts it.

`gamma` must now be a finite real number in the closed interval `[0, 1]` on both
backends. Both endpoints stay first-class: `gamma=1` is the undiscounted
episodic return and `gamma=0` a myopic agent. The check is a new shared gate
alongside the existing run-size / launch-topology / learning-rate / seed ones,
so a third RL backend that discounts a return with the field cannot ship without
it. This generalizes the shape the FastSAC preflight already used for its own
interval coefficient, `tau` in `(0, 1]`.
