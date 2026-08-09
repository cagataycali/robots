### Fixed: `PolicyRunner.evaluate` refuses an `n_episodes` / `max_steps` that runs no evaluation

`n_episodes` and `max_steps` are the outer and inner bound of the episode loop
`evaluate` runs, and they applied no domain. Unlike a knob whose misuse degrades
an evaluation, these two removed it while still reporting one: `success_rate` is
`n_success / max(n_completed, 1)`, so the guard protecting the division also
turned "nothing ran" into a clean `0.0`, and `success_measured` is derived from
whether a criterion was *supplied* rather than ever evaluated.

So `n_episodes=0` and `max_steps=0` each returned `status="success"` with
`success_rate: 0.0` **and** `success_measured: true` over zero applied actions -
indistinguishable from a policy that genuinely failed every episode, and
`success_measured` is the field that exists precisely so a `0.0` cannot be
mistaken for a measurement. `max_steps=nan` reached the same result silently
(`steps < nan` is `False` on the first test), `2.7` and `True` truncated to a
horizon nobody typed, and `max_steps=inf` did not report a wrong number at all:
`while steps < max_steps` has no false case, so the first episode never
returned - an unbounded spend of one model inference per step.

The same method already refused the same mistake one path over:
`_evaluate_with_spec` checks the benchmark's horizon with
`positive_count_error(spec.max_steps, "max_steps", "evaluate_benchmark")` and a
message describing exactly what `evaluate` did with its own parameter, its
comment explaining the placement by asserting that every other bound of that
nested loop "is checked by the public entry point before it gets here". That
holds for the facade - `SimEngine.eval_policy` checks both - and not for this
layer, which is documented as drivable directly.

Both bounds now apply that same shared domain (`positive_count_error`) and raise
`ValueError` naming the parameter and the method, before `sim.reset()`,
`set_eval_seed` (which reseeds the process-global RNG) and the first inference,
so a refused eval costs nothing and leaves no global side effect. `n_episodes` is
unconditional, since it bounds the episode loop on both eval paths. `max_steps`
is checked only when it is the effective horizon: with a `spec` it is documented
as ignored (`spec.max_steps` wins and is refused at its own read), so refusing it
there would reject a value that changes nothing - the same scoping
`SimEngine.run_policy` applies to `duration`.
