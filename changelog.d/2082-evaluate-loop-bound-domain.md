### Fixed: `PolicyRunner.evaluate` refuses a loop bound it cannot evaluate

`n_episodes` and `max_steps` are the two bounds of `evaluate`'s own episode loop and had no
domain, while every public entry point above them already refuses a non-positive integer for
both. A bound outside that domain did not shorten the evaluation, it removed it while still
reporting one: `n_episodes=0` returned `status="success"` over zero episodes and `max_steps=0`
over episodes of zero length, each with `success_rate: 0.0` **and `success_measured: true`** -
the flag that exists so a `0.0` cannot be mistaken for a measurement - over zero applied
actions. `max_steps=inf` never returned, because `while steps < max_steps` has no false case,
and `2.7` / `True` truncated to a horizon nobody typed while `"3"` / `None` / a list leaked a
bare `'X' object cannot be interpreted as an integer` naming neither the parameter nor the
method. Both are now checked against the same shared count domain the facade and the benchmark
path use, raised as `ValueError` (this layer's contract for a direct caller) before
`sim.reset()`, before `set_eval_seed` touches the process-global RNG and before the first
inference. `max_steps` is validated only when it is the horizon actually read, so a `spec=`
call - which takes its horizon off the benchmark - is not refused for a value it never reads.
