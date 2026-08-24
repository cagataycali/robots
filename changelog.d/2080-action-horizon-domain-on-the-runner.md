### Fixed: `PolicyRunner.run` / `PolicyRunner.evaluate` refuse an `action_horizon` they cannot run

`action_horizon` is how many actions are consumed from one policy chunk before
re-querying. Every public entry point that accepts it - `SimEngine.run_policy` /
`eval_policy` / `evaluate_benchmark` and `MuJoCoSimEngine.start_policy` /
`run_multi_policy` - validates it through `SimEngine._validate_action_horizon`,
whose docstring gives the reason: an out-of-domain value "would otherwise be
silently clamped to 1 by `resolve_chunk_length`, hiding the caller's mistake
behind a rollout that does not run the requested horizon".

`PolicyRunner.run` and `PolicyRunner.evaluate` are the layer that consumes the
value and are documented as drivable directly, and they applied no domain. A
direct caller therefore got exactly the clamp those checks exist to prevent:
`0`, `-5`, `True`, `2.7` and `"8"` each ran a rollout to `status="success"` at a
re-query interval nobody asked for, while `nan` / `inf` / `None` / a list leaked
a bare `int()` conversion error out of the FIRST inference - surfaced by `run`
as "Policy failed: cannot convert float NaN to integer", naming neither the
parameter nor the method, and propagating uncaught out of `evaluate`.

Both surfaces now apply the same shared domain the entry points delegate to
(`strands_robots.utils.positive_count_error`) and raise `ValueError` naming the
parameter and the method, before any inference is queried or action applied.
That matches the two sibling knobs of the same signature: `control_substeps`,
whose own docstring calls its raise "the guarantee for callers driving
`PolicyRunner` directly", and `control_frequency`. The check is unconditional,
exactly as the entry point's is - an RTC policy ignores the horizon because it
owns its own re-query interval, but that is a property of the policy rather than
of the caller's request.
