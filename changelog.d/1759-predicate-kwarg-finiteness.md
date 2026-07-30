### Fixed: a benchmark spec predicate kwarg that is not a finite number is refused

A numeric kwarg in a benchmark spec is coerced with a bare `float(...)` inside
its predicate factory and then closed over, so a `nan`/`inf` threshold or weight
compiled clean and only surfaced in the evaluated result.

In a `dense_reward` term it made the episode's `cumulative_reward` and the
evaluation's `avg_reward` `nan`/`inf`, reported under `status="success"` - a
poisoned score handed to whatever consumes it. In a `success`, `failure` or
`stop_when` clause every comparison against `nan` is `False`, so the clause was
unsatisfiable and the rollout burned its whole step budget reporting an honest
miss. That second one is the failure mode a typo'd body name is already probed
against the live sim to prevent; a non-finite threshold reached it by a route
that was not checked. A non-numeric value escaped as a bare "could not convert
string to float" naming neither the predicate nor the field it came from.

`make_predicate` now holds every kwarg the factory annotates as numeric to the
same finite domain the sim setters use. The check lives there rather than in the
spec compiler because it is the only choke point every predicate call passes
through: `staged_reward` compiles its per-stage `reward` and `advance_when` calls
by calling back into `make_predicate`, so a guard in the compiler left nested
stage calls unchecked. A stage's own `bonus` is not a factory param and carries
the same domain directly.

The domain is read from the factories' parameter annotations, the same mechanism
that already classifies a predicate as bool- or float-valued from its return
annotation, so a predicate added later is covered by declaring its params rather
than by remembering to extend a list.
