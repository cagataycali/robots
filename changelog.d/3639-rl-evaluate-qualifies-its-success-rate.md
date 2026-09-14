### Fixed: the RL trainer's evaluate() says whether its success_rate measured the policy

`BaseRLAlgo.evaluate` publishes the same `success_rate` that `PolicyRunner.evaluate`
and `PolicyRunner.evaluate_benchmark` publish, and it carried none of the
qualifications those two report. `SimEnv.step` samples `success_fn` only after an
applied action, so a predicate that already holds at reset terminates the episode
on its first step whatever the policy commands; and with no `success_fn` at all
nothing can terminate, so every episode times out. Either way the rate is a
constant the policy did not earn, and the returned metrics said nothing - the
`success_at_reset_warning` helper that states this rule for the two simulation
routes names them as the only two.

Measured on SO-101 MuJoCo physics with a policy commanding its own current pose:
a `body_on(cube, target)` predicate against a cube whose spawn already rests on
the target reported `success_rate 1.0` over five episodes, each ending after one
control step of a forty-step budget. The same scene with the cube spawned off the
target reported `0.0`, as did the same scene with no predicate at all - two
identical numbers, one a measurement and one not.

`evaluate` now samples the predicate once per episode at reset and returns
`episodes_successful_at_reset` (int) and `success_measured` (bool), warning on
each. Every reported figure is left exactly as measured and nothing is refused:
domain randomisation draws initial states per episode through `reset_fn`, so a
partial count is a fact about those draws rather than a broken predicate, and the
count is what distinguishes them. The reset sample is diagnostic, so a predicate
reading state that only a first step establishes is not counted rather than fatal.

`success_at_reset_warning` gained a `reported` keyword so the warning names the
figures its caller actually publishes; this route has no `pass_hat_k`, and a
remedy naming a field absent from the result sends the reader looking for nothing.
Both existing callers keep their message unchanged.
